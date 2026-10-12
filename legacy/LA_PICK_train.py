"""
Archived legacy training script for Left Atrium (LA) dataset.
Moved to legacy/ as part of the MedSSL-BHSD refactoring plan.
"""
from asyncore import write
import os
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb
import time

from test_util import test_calculate_metric, test_calculate_metric_LA
from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/userdisk1/qjzeng/semi_seg/preprocess/', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='exp', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=5000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='trained samples')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--lambda_', type=float, default=0.2, help='balance loss')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()

def random_mask(img, mask_ratio):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    img_mask = torch.ones(batch_size, channel, img_x, img_y, img_z).cuda()
    label_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    for i in range(batch_size):
        w = np.random.randint(0, 96 - patch_pixel_x)
        h = np.random.randint(0, 96 - patch_pixel_y)
        z = np.random.randint(0, 96 - patch_pixel_z)
        label_mask[i, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
        img_mask[i, :, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return img_mask.long(), label_mask.long()

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path):
    state = {'net': net.state_dict(), 'opt': optimizer.state_dict()}
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
