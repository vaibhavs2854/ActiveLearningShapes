import numpy as np
import torch
import torchvision
from time import time
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.autograd import Variable
import os
import glob
import cv2
from tqdm import tqdm

from matplotlib.pyplot import imsave, imread
import matplotlib.pyplot as plt
import sys
import matplotlib.gridspec as gridspec

import copy

"""Largest Contiguous Region Helper Methods"""
def bfs_flood_fill(mask,i,j,val):
    queue = []
    queue.append((i,j))
    while len(queue)>0:
        i,j = queue.pop(0)
        if i<0 or j<0 or i>=mask.shape[0] or j>=mask.shape[1]:
            continue
        if not mask[i,j]==1:
            continue
        mask[i,j] = val
        queue.append((i+1,j))
        queue.append((i-1,j))
        queue.append((i,j+1))
        queue.append((i,j-1))
    return mask

def get_all_regions(mask):
    val = 2
    count_dict = {}
    modifying_mask = mask.copy()
    while 1 in modifying_mask:
        tuples = np.nonzero(modifying_mask==1)
        i,j = tuples[0][0],tuples[1][0]
        modifying_mask = bfs_flood_fill(modifying_mask,i,j,val)
        count_dict[val] = len(np.where(modifying_mask==val)[0])
        val += 1
    return count_dict, modifying_mask

def largest_contiguous_region(mask):
    if 1 not in mask:
        return mask
    count_dict,modifying_mask = get_all_regions(mask)
    max_val = 2
    max_ = count_dict[max_val]
    for key in count_dict.keys():
        if count_dict[key] > max_:
            max_val = key
            max_ = count_dict[key]
    return np.where(modifying_mask==max_val,1,0)

def get_int(mask):
    return np.where(mask>0.2,1,0)

"""END"""
