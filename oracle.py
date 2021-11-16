
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



def display_image(patient_ids,source_path):
    ncols, nrows = 3, len(patient_ids)
    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(9, 3*len(patient_ids)+1)
    fig.tight_layout()
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig,hspace=0,wspace=0)


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
        filepath = source_path + "/" + patient_ids[row] + ".npy" #doesn't change cuz of generalizing filepath
        image_and_mask = np.load(filepath)
        f_axes[row][0].imshow(image_and_mask[0],cmap='gray')
        f_axes[row][0].set_axis_off()
        f_axes[row][1].imshow(image_and_mask[1],cmap='gray')
        f_axes[row][1].set_axis_off()

        heatmap = cv2.applyColorMap(np.uint8(255*(1-image_and_mask[1])), cv2.COLORMAP_AUTUMN)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]

        img = 0.6 * np.stack([image_and_mask[0],image_and_mask[0],image_and_mask[0]],axis=-1) + 0.3*heatmap
        f_axes[row][2].imshow(img)
        f_axes[row][2].set_axis_off()

    plt.show()


#picks uniform scores to return to oracle

def query_oracle(oracle_results,patient_scores,im_dir,query_method="uniform",query_number=10):
    if query_number==0:
        print("Why are you asking for 0 queries?")
        return oracle_results
    if query_number>len(patient_scores):
        print("Query too big for number of patients")
        return oracle_results
    oracle_queries = []
    if query_method=="uniform":
        step = len(patient_scores)//(query_number-1)
        for i in range(0,len(patient_scores),step):
            if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
                oracle_queries.append(list(patient_scores.keys())[i])
    elif query_method=="random":
        indices = random.sample(len(patient_scores), query_number)
        for i in indices:
            if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
                oracle_queries.append(list(patient_scores.keys())[i])
    elif query_method=="best":
         for i in range(query_number-1,-1,-1):
             if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
                oracle_queries.append(list(patient_scores.keys())[i])
    elif query_method=="worst":
        for i in range(query_number):
            if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
                oracle_queries.append(list(patient_scores.keys())[i])
    else:
        print("You entered an unsupported query method.")
        return oracle_results
    display_image(oracle_queries,im_dir)
    #ensures that the oracle puts in the correct number of annotations
    ensure_correct_input_flag = True
    while ensure_correct_input_flag:
        oracle_input = input("If any part of an image segmentation is incorrect, label the entire image as incorrect.\nEnter labels for images per row without separation. 1 is a good label, 0 is a bad label. 2 is skip: ")
        raw_split = np.array(list(oracle_input))
        split = [int(i) for i in raw_split]
        ensure_correct_input_flag = (len(split)!=len(oracle_queries))
        if(ensure_correct_input_flag):
            print("\nYou have entered the wrong number of inputs. Please try again.")
    for index,i in enumerate(oracle_queries):
        oracle_results[i] = split[index]
    return oracle_results


def calculate_dispersion_metric(patient_scores,oracle_results):
    num_ones = np.sum(np.array([oracle_results[i] for i in oracle_results.keys() if oracle_results[i]==1]))
    #assuming that patient_scores is ordered
    num_twos = (0.5)*np.sum(np.array([oracle_results[i] for i in oracle_results.keys() if oracle_results[i]==2]))
    num_zeros = len(oracle_results.keys()) - num_ones - num_twos
    tupled_patient_scores = []
    for i in list(patient_scores.keys()):
        if i in list(oracle_results.keys()):
            tupled_patient_scores.append((i,patient_scores[i]))
    tupled_patient_scores = sorted(tupled_patient_scores,key = lambda x:x[1])   
    metric = 0
    for index,i in enumerate(tupled_patient_scores):
        if oracle_results[i[0]]==1:
            for j in range(index+1,len(tupled_patient_scores),1):
                if oracle_results[tupled_patient_scores[j][0]]==0: #red
                    metric += 1
    return 1 - metric/(num_ones*num_zeros)


def save_oracle_results(oracle_results,im_dir,save_dir):
    all_save_paths = []
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    for patient in oracle_results.keys():
        #patient is the patient id
        #oracle_results[patient] is 0/1
        if(oracle_results[patient]==1):
            #good segmentation - save into another folder
            if(patient.startswith("/")):
                save_path = save_dir+ patient[1:] + ".npy"
            else:
                save_path = save_dir + patient + ".npy";
            load_path = im_dir + patient + ".npy"
            shape_type = load_path.split("/")[-2]
            im = np.load(load_path)
            save_dir_dir = save_dir + shape_type + "/"
            if not os.path.exists(save_dir_dir):
                os.makedirs(save_dir_dir)
            np.save(save_path,im)
            all_save_paths.append(save_path)
    print("Done with saving this iteration of oracle results")
    return all_save_paths





