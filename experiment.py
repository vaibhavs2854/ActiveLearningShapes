#Python Library imports
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
import argparse

#Backend py file imports
from floodfill import *
from dataloader import *
from model import *
from oracle import *
from unet import *
from manual_oracle import query_oracle_automatic,ask_oracle_automatic

#grab arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int)
args = parser.parse_args()
random_seed_number = args.random_seed

random_seed_number = 1

#set random seeds
torch.manual_seed(random_seed_number)
torch.cuda.manual_seed(random_seed_number)
np.random.seed(random_seed_number)
random.seed(random_seed_number)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True


#write a custom dataloader that only uses x images from the training dataset
#size is the number of datapoints we're using in the dataloader
def from_manual_segmentation_dataloader_with_size(manual_seg_dir,batch_size,num_workers,size):
    filepaths = []
    for root,dirs,files in os.walk(manual_seg_dir):
        for file in files:
            if file.endswith(".npy"):
                filepaths.append(os.path.join(root,file))
    #sort by patient id
    filepaths = sorted(filepaths, key=lambda x:x.split("/")[-1])
    filepaths = filepaths[0:size]
    new_unet_dataloader = unet_dataloader(filepaths,batch_size,num_workers)
    return new_unet_dataloader


def intersection_over_union(output_mask,ground_mask):
    ground_mask = get_ints(ground_mask).squeeze(1)
    summed = ground_mask + output_mask
    twos = summed - 2
    num = 256*256 - torch.count_nonzero(twos)
    denom = torch.count_nonzero(summed)
    outputs = torch.div(num,denom)
    return torch.mean(outputs)

def evaluate_metric_on_validation(model,validation_dir):
    model.eval()
    transforms_arr = [transforms.ToTensor(),transforms.Resize((256,256))]
    image_transform = transforms.Compose(transforms_arr)
    ious = []
    segmentation_filepaths = []

    for root, dirs, files in os.walk(validation_dir):
        for file in files:
            if file.endswith(".npy"):
                segmentation_filepaths.append(os.path.join(root,file))
    
    for filepath in tqdm(segmentation_filepaths):
        arr_and_bin_output = np.load(filepath)

        arr = arr_and_bin_output[0,:,:].copy()
        bin_output = arr_and_bin_output[1,:,:].copy()

        mask = image_transform(bin_output)[0,:,:]
        arr = exposure.equalize_hist(arr) #add hist equalization to 
        image = image_transform(arr)        
        
        image = image.float()
        image = convert_to_3channel(image).cuda()
        image = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (image)

        unet_seg = model(image)
        unbinarized_unet_seg = F.softmax(unet_seg[0],dim=0)[1,:,:]
        unet_seg = get_binary_mask(unbinarized_unet_seg).cpu()
        iou = intersection_over_union(unet_seg,mask)
        ious.append(iou)
    return np.average(np.asarray(ious))


#Returns the original image, ground truth label, binarized UNet segmentation output, and iou of the ground truth and binarized seg output.
#Takes in the image filepath and the model for segmenting.
def grab_iou_of_image(image_filepath,model):
    model.eval()
    transforms_arr = [transforms.ToTensor(),transforms.Resize((256,256))]
    image_transform = transforms.Compose(transforms_arr)

    arr_and_bin_output = np.load(image_filepath)

    arr = arr_and_bin_output[0,:,:].copy()
    bin_output = arr_and_bin_output[1,:,:].copy()

    mask = image_transform(bin_output)[0,:,:]
    arr = exposure.equalize_hist(arr) #add hist equalization to 
    image = image_transform(arr)        
    
    image = image.float()
    image = convert_to_3channel(image).cuda()
    image = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (image)

    unet_seg = model(image)
    unbinarized_unet_seg = F.softmax(unet_seg[0],dim=0)[1,:,:]
    unet_seg = get_binary_mask(unbinarized_unet_seg).cpu()
    iou = intersection_over_union(unet_seg,mask)
    return image,mask,unet_seg,iou

def control_run():
    #Initialize filepaths 
    im_dir = "/usr/xtmp/vs196/mammoproj/Data/manualfa/train/" #manually labelled training segmentations
    manual_fa_train_dir = "/usr/xtmp/vs196/mammoproj/Data/manualfa/train/"
    model_save_dir = "/usr/xtmp/vs196/mammoproj/Code/SavedModels/ControlALUNet/0726/"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    sizes = []
    metrics = []
    for size in range(10,200,10):
        #initialize dataloader and model
        manual_fa_dataloader = from_manual_segmentation_dataloader_with_size(manual_fa_train_dir,10,2,size)
        unet_model = getattr(ternausnet.models, "UNet16")(num_classes=2,pretrained=True).cuda()

        #train model
        unet_model,loss_tracker,metric_tracker = unet_update_model(unet_model,manual_fa_dataloader,num_epochs=25)

        #save model
        model_save_path = model_save_dir + f"unetmodel_size{size}.pth"
        torch.save(unet_model,model_save_path)

        manual_fa_valid_dir = f"/usr/xtmp/vs196/mammoproj/Data/manualfa/manual_validation/"
        metric = evaluate_metric_on_validation(unet_model,manual_fa_valid_dir)
        sizes.append(size)
        metrics.append(metric)
        print(f"Done with size={size}. Metric={metric}")
    print("Done with all sizes")
    plt.scatter(sizes,metrics)
    sizes_np = np.asarray(sizes)
    metrics_np = np.asarray(metrics)
    data_save_dir = "/usr/xtmp/vs196/mammoproj/Code/ActiveLearning/0727controlrun/"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    sizes_save_path = "/usr/xtmp/vs196/mammoproj/Code/ActiveLearning/0727controlrun/sizes.npy"
    metrics_save_path = "/usr/xtmp/vs196/mammoproj/Code/ActiveLearning/0727controlrun/metrics.npy"
    np.save(sizes_save_path,sizes_np)
    np.save(metrics_save_path,metrics_np)
    plt.savefig("/usr/xtmp/vs196/mammoproj/Code/ActiveLearning/0727controlrun/control_graph.png")
    print("Done")


if __name__ == "__main__":
    model_save_path = "/usr/xtmp/vs196/mammoproj/Code/SavedModels/ControlALUNet/0726/unetmodel_size150.pth"
    model = torch.load(model_save_path)
    image_filepath = "/usr/xtmp/vs196/mammoproj/Data/manualfa/manual_validation/Round/DP_AAZA_161134.npy"
    image,mask,unet_seg,iou = grab_iou_of_image(image_filepath,model)
    print(f"IOU of this image is: {iou}")


#Evaluates BINARIZED segmentations from a eval_dir containing image and manual segmentation.
#Saves as [image,segmentation] np stack in save_dir.
#DID NOT NEED THIS CODE
def generate_segmentations(model,manual_seg_dir,save_dir):
    image_transform = None #FIX
    images_filepaths = []
    for root, dirs, files in os.walk(manual_seg_dir):
            for file in files:
                if file.endswith(".npy"):
                    images_filepaths.append(os.path.join(root,file))
        
    for image_filepath in tqdm(images_filepaths):
        #Load image from filepath
        image_and_manual_seg = np.load(image_filepath)
        arr = image_and_manual_seg[0,:,:].copy()
        loaded_image = arr.copy()
        
        #Preprocess image before feeding to model
        arr = exposure.equalize_hist(arr) #add hist equalization
        image = image_transform(arr)        
        image = image.float()
        image = convert_to_3channel(image).cuda()
        image = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (image)

        #Evaluate unbinarized unet seg
        unet_seg = model(image)
        unbinarized_unet_seg = F.softmax(unet_seg[0],dim=0)[1,:,:]

        #Save to save_dir.
        patID = '/'.join(image_filepath.split("/")[-2:])[:-4]
        image_save_path = save_dir + patID + ".npy"
        print(image_save_path)
        break
        #np.save(image_save_path,np.stack([loaded_image,unbinarized_unet_seg]))
    print(f"Saved unbinarized segmentations of {len(images_filepaths)} images to {save_dir}.")

#One run to generate unbinarized training segmentations from scratch.
#generate_segmentations(model,classifier_training_dir,"/usr/xtmp/vs196/mammoproj/Data/manualfa/unbinarized_train_seg/")


def plot_active_learning_training_metrics(all_patient_scores,oracle_results):
    pass

#Saves any correct segmentations found by the oracle, and additionally pickle dumps any structures 
def save_active_learning_results(run_id,iter_num,oracle_results,oracle_results_thresholds,im_dir):
    save_dir = "/usr/xtmp/vs196/mammoproj/Code/ActiveLearning/AllOracleRuns/Run_" + run_id + "/Iter" + str(iter_num)
    correct_segs_save_dir = save_dir + "CorrectSegmentations/"

    saved_oracle_filepaths = save_oracle_results(oracle_results,oracle_results_thresholds,im_dir,correct_segs_save_dir)
    fpath = save_dir + "saved_data_struct/"
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    saved_oracle_filepaths_filepath = save_dir + "saved_data_struct/Oracle_Filepaths.pickle"
    pickle.dump(saved_oracle_filepaths,open(saved_oracle_filepaths_filepath,"wb"))
    pickle.dump(oracle_results,open(save_dir + "saved_data_struct/Oracle_Results.pickle","wb"))
    pickle.dump(oracle_results_thresholds,open(save_dir + "saved_data_struct/Oracle_Results_Thresholds.pickle","wb"))
    

def active_learning_experiment(active_learning_train_cycles,query_cycles,run_id,iter_num):
    #ACTIVE LEARNING STAGE

    #File definitions and static setup
    classifier_training_dir = "/usr/xtmp/vs196/mammoproj/Data/manualfa/train/"
    ground_truth_dir = "/usr/xtmp/vs196/mammoproj/Data/manualfa/train/"
    segmentation_dir = "/usr/xtmp/mammo/image_datasets/data_split_july2021/square_ROI_by_shape_segmentations_unbin/train/" #Unbinarized train segmentations from something idk.
    oracle_results = dict()
    oracle_results_thresholds = dict()
    total_images_shown = 0

    #Begin loop over number of active learning/
    for active_learning_cycle in range(active_learning_train_cycles):
        #Variable Definitions
        dataloader = get_DataLoader(classifier_training_dir,32,2)
        model,loss_tracker,criterion,optimizer = initialize_and_train_model(dataloader, epochs=10) #Initialize and train classifier for 10 epochs.  #TODO:rewrite dataloader to use all positive segmentations from manual classifier (NOT switching)
        patient_scores = get_patient_scores(model,dataloader)
        all_patient_scores = []

        #Querying oracle - currently queries {query_cycles} times.
        for query_iter in range(query_cycles):
            query_number = 10
            oracle_results, oracle_results_thresholds = query_oracle_automatic(oracle_results,oracle_results_thresholds,patient_scores,ground_truth_dir,segmentation_dir,query_method="uniform",query_number=query_number)
            total_images_shown += query_number

        #Updating classifier 1 epoch at a time for 5 epochs. 
        for i in range(5):
            model = model_update(model,dataloader,oracle_results,criterion,optimizer,num_epochs=1)

            patient_scores = get_patient_scores(model,dataloader)
            all_patient_scores.append(patient_scores)

        #Space for plotting metrics if you want.
        plot_active_learning_training_metrics(all_patient_scores,oracle_results)
    
        #Space for saving oracle results and pickling data structures
        save_active_learning_results(run_id,iter_num,oracle_results,oracle_results_thresholds,classifier_training_dir)

        print("Done with one iteration of active learning querying/updating")
        
    #SEGMENTATION STAGE
    unet_model = None #load unet model for segmentation



#Outline of experiment:
#loop over:
#   Active Learning Stage
#   Segmentation Stage
#   Evaluation Stage + Metrics