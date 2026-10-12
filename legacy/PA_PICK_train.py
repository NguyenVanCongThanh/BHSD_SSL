"""
Archived legacy training script for Pancreas (PA) dataset.
Moved to legacy/ as part of the MedSSL-BHSD refactoring plan.
"""
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn

from test_util import test_calculate_metric
from skimage.measure import label
from torch.utils.data import DataLoader
from utils import losses, ramps
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/userdisk1/qjzeng/semi_seg/preprocess/', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='exp', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=5000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=60, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=12, help='trained samples')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--lambda_', type=float, default=1.0, help='balance loss')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
args = parser.parse_args()
