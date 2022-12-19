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
import pickle

from PIL import Image

def display_image_annotation(filepath,annotations,png=False):
    ncols, nrows = 3, len(filepath)
    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(9, 3*len(filepath)+1)
    fig.tight_layout()
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig,hspace=0,wspace=0)
    
    anno_opts = dict(xy=(0.05, 0.05), xycoords='axes fraction', va='bottom', ha='left',color='cyan',fontweight='extra bold',fontsize='8')

    f_axes = []
    for row in range(nrows):
        f_axes.append([])
        for col in range(ncols):
            f_axes[-1].append(fig.add_subplot(spec[row, col]))

    for ax_num, ax in enumerate(f_axes[0]):
            if ax_num == 0:
                ax.set_title("Image", fontdict=None, loc='left', color = "k")
            elif ax_num == 1:
                ax.set_title("Segmentation", fontdict=None, loc='left', color = "k")
            elif ax_num == 2:
                ax.set_title("Overlay", fontdict=None, loc='left', color = "k")

    for row in range(nrows):
        if png:
            #image = np.array(Image.open(filepath[row][0]))
            #mask = np.array(Image.open(filepath[row][1]))
            image = (cv2.imread(filepath[row][0]))/255
            mask = (255 - cv2.imread(filepath[row][1]))/255
            image_and_mask = np.stack([image,mask])
        else:
            image_and_mask = np.load(filepath[row])
        f_axes[row][0].imshow(image_and_mask[0],cmap='gray')
        f_axes[row][0].set_axis_off()
        
        f_axes[row][1].imshow(image_and_mask[1],cmap='gray')
        f_axes[row][1].set_axis_off()

        heatmap = cv2.applyColorMap(np.uint8(255*(1-image_and_mask[1])), cv2.COLORMAP_AUTUMN)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        if png:
            img = 0.6 * image_and_mask[0] + 0.3*heatmap
        else:
            img = 0.6 * np.stack([image_and_mask[0],image_and_mask[0],image_and_mask[0]],axis=-1) + 0.3*heatmap
        f_axes[row][2].imshow(img)
        f_axes[row][2].set_axis_off()
        
        f_axes[row][0].annotate(annotations[row],**anno_opts)

    #plt.show()


from PIL import Image

medstudent_filepath = "/usr/xtmp/mammo/image_datasets/MedStudentSegmentations_dst/Missing_27_dst/"
batches = [1]

for batch in batches: #loop through each batch
    png_filepaths = []
    full_filepath = medstudent_filepath + f"Batch_{batch}" + "/"
    #scan through each batch to find seg
    for root, dirs, files in os.walk(full_filepath):
        for file in files:
            if file.endswith(".png"):
                filepath = os.path.join(root, file)
                tail_filepath = '/'.join(filepath.split("/")[-2:])
                png_filepaths.append(tail_filepath)   
    #Find each pair of normal/fa and make numpy arrays using Image.open
    for png_filepath in tqdm(png_filepaths):
        if(png_filepath[-6:-4]=='fa'):
            png_image_filepath = full_filepath + png_filepath[:-7] + png_filepath[-4:]
            png_mask_filepath = full_filepath + png_filepath
            display_annotation_filepath = [png_image_filepath,png_mask_filepath]
            display_image_annotation([display_annotation_filepath],[png_filepath],png=True)
            #make a subfolder under each batch at the level of irreg/oval that contains all images
            save_folder = full_filepath + "Fides/"
            if not os.path.exists(save_folder + png_filepath.split("/")[0]):
                os.makedirs(save_folder + png_filepath.split("/")[0])
            save_path = save_folder + png_filepath
            plt.savefig(save_path)
            plt.clf()
            plt.close()