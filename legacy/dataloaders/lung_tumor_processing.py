"""
Archived legacy Lung tumor data processing script from original PICK repository.
"""
import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd
import os
import nibabel as nib

output_size = [96, 96, 96]

def normalize(data):
    normalized_data = (data - data.min()) / (data.max() - data.min())
    return normalized_data

def save_to_h5(img, mask, filename):
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('image', data=img)
        hf.create_dataset('label', data=mask)
