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

from floodfill import largest_contiguous_region

from floodfill import *
from dataloader import *
from model import *
from oracle import *
from unet import *

def intersection_over_union(output_mask,ground_mask):
    ground_mask = get_ints(ground_mask).squeeze(1)
    summed = ground_mask + output_mask
    twos = summed - 2
    num = 256*256 - np.count_nonzero(twos)
    denom = np.count_nonzero(summed)
    outputs = np.div(num,denom)
    return np.mean(outputs)

#TODO: Check if patID has shape subdir. If not, you're going to have to find them manually.
def calculate_iou(patID,threshold,ground_truth_dir,segmentation_dir):
    ground_truth_path = ground_truth_dir + patID + ".npy"
    segmentation_path = segmentation_dir + patID + ".npy"
    ground_truth = np.load(ground_truth_path)
    segmentation = np.load(segmentation_path)
    binarized = get_binary_mask_threshold(segmentation,threshold)
    return intersection_over_union(binarized,ground_truth)

#Mimics same input/output as ask_oracle from oracle.py. 
#Only difference is that it asks for two directories as it needs both the unbinarized seg dir and the ground truth dir
def ask_oracle_automatic(oracle_results, oracle_results_thresholds,oracle_queries, ground_truth_dir,segmentation_dir, iou_threshold=0.9):
    for patID in oracle_queries:
        max_iou = -1
        max_threshold = 0
        #calculate iou over a variety of thresholds
        thresholds = [0.01,0.05,0.1,0.15,0.2]
        for threshold in thresholds:
            iou = calculate_iou(patID,threshold,ground_truth_dir,segmentation_dir)
            #Check if proposed segmentation is very close to ground truth (Starting off at 0.2)
            if iou > max_iou:
                max_iou = iou
                max_threshold = threshold
        if(max_iou > iou_threshold):
            #return a 1 with correct threshold
            oracle_results[patID] = 1
        else:
            oracle_results[patID] = 0
        oracle_results_thresholds[patID] = max_threshold
    return oracle_results,oracle_results_thresholds


def query_oracle(oracle_results,oracle_results_thresholds,patient_scores,ground_truth_dir,segmentation_dir,query_method="uniform",query_number=10,threshold=0.2):
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
                oracle_queries.append((list(patient_scores.keys())[i],threshold))
    elif query_method=="random":
        indices = random.sample(len(patient_scores), query_number)
        for i in indices:
            if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
                oracle_queries.append((list(patient_scores.keys())[i],threshold))
    elif query_method=="best":
         for i in range(query_number-1,-1,-1):
             if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
                oracle_queries.append((list(patient_scores.keys())[i],threshold))
    elif query_method=="worst":
        for i in range(query_number):
            if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
                oracle_queries.append((list(patient_scores.keys())[i],threshold))
    elif 'middle' in query_method:
        #find the number of elements closest to 0.5
        split_val = float(query_method.split('=')[-1])
        middle_index = int(len(patient_scores.keys())/(1/split_val))
        left_bound = 0 if middle_index - int(query_number/2) < 0 else middle_index - int(query_number/2)
        indices = list(range(middle_index,middle_index+int(query_number/2))) + range(left_bound,middle_index)
        for i in indices:
            if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
                oracle_queries.append((list(patient_scores.keys())[i],threshold))
        print("Debugging for middle index: "  + str(middle_index) + " " + str(patient_scores[middle_index]))
    elif "percentile" in query_method:
        percentile = float(query_method.split('=')[-1])
        near_index = int(len(patient_scores.keys()) * percentile)
        indices = list(range(near_index - int(query_number/2), near_index)) + list(range(near_index, near_index + int(query_number/2)))
        for i in indices:
            if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
                oracle_queries.append((list(patient_scores.keys())[i],threshold))
    else:
        print("You entered an unsupported query method.")
        return oracle_results

    ask_oracle_automatic(oracle_results,oracle_results_thresholds,oracle_queries,ground_truth_dir,segmentation_dir)
    
    return oracle_results, oracle_results_thresholds