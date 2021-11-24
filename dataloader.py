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

#from floodfill import largest_contiguous_region

class get_data(Dataset):
    def __init__(self,image_filepaths,image_transform):
        super().__init__()
        self.image_filepaths = image_filepaths
        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_filepaths)

    def __getitem__(self,idx):
        filepath = self.image_filepaths[idx]
        #print("Filepath: " + filepath)
        arr_and_mask = np.load(filepath)
        copy_arr_mask = arr_and_mask.copy()
        arr = copy_arr_mask[0,:,:].copy()
        mask = copy_arr_mask[1,:,:].copy()
        
        #mask = largest_contiguous_region(mask)
        
        image = (self.image_transform(arr))
        #image = our_transform(image)
        #print(image.shape)
        #print("OUTSIDE")
        mask_label = (self.image_transform(mask))
        
        patient_id = '/'.join(filepath.split("/")[-2:])[:-4]
        #patient_id = filepath.split("/")[-1][:-4]
        #mask_label = our_transform(mask_label)
        return image,mask_label.long(),patient_id
    
def get_DataLoader(train_images_directory,batch_size,num_workers):
    #find all files inside train_images_directory, assign to train_images_filenames
    train_images_filepaths = []
    for root, dirs, files in os.walk(train_images_directory):
        for file in files:
            if file.endswith(".npy"):
                train_images_filepaths.append(os.path.join(root,file))
                
    transforms_arr = [transforms.ToTensor(),transforms.Resize((256,256))]
    transform = transforms.Compose(transforms_arr)
    
    trainset = get_data(train_images_filepaths,transform)
    trainloader = DataLoader(trainset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    return trainloader