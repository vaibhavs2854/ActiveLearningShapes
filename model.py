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


def initialize_and_train_model(dataloader,batch_size=32,epochs=15):
    #Initial training on mismatched image-label pairs (half matched w/ label 1, half mismatched w/ label 0)
    num_classes=2

    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=False)
    model.classifier[6] = nn.Linear(4096,num_classes)
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss();
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    loss_tracker = [] #plot loss

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for image,mask,_ in tqdm(dataloader):
            feed_in_data = torch.empty((image.shape[0],3,256,256))
            labels = [0]*image.shape[0] 
            for i in range(image.shape[0]):
                #print(batch_size/2)
                if(i<image.shape[0]/2):
                    #print(mask.shape)
                    #print(i)
                    feed_in_data[i] = torch.stack([image[i],image[i],mask[i+1]]).squeeze()
                    labels[i] = 0;
                else:
                    feed_in_data[i] = torch.stack([image[i],image[i],mask[i]]).squeeze()
                    labels[i] = 1; #1 is correct label, 0 is incorrect label
            #print(feed_in_data.shape)
            images = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (feed_in_data)
            images = images.cuda()
            labels = torch.from_numpy(np.array(labels)).cuda()
            optimizer.zero_grad()

            y = model(images)

            loss = criterion(y,labels)
            loss_tracker.append(loss.detach().cpu().item()/batch_size)
            loss.backward()
            optimizer.step()
            
    return model,loss_tracker,criterion,optimizer


#Difference from above is that we don't randomly show 0,1. Model sees that all segmentations are correct because it is.
def initialize_and_train_model_experiment(dataloader,batch_size=32,epochs=15):
    #Initial training on mismatched image-label pairs (half matched w/ label 1, half mismatched w/ label 0)
    num_classes=2

    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=False)
    model.classifier[6] = nn.Linear(4096,num_classes)
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss();
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    loss_tracker = [] #plot loss

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for image,mask,_ in tqdm(dataloader):
            feed_in_data = torch.empty((image.shape[0],3,256,256))
            labels = [1]*image.shape[0] 
            #print(feed_in_data.shape)
            images = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (feed_in_data)
            images = images.cuda()
            labels = torch.from_numpy(np.array(labels)).cuda()
            optimizer.zero_grad()

            y = model(images)

            loss = criterion(y,labels)
            loss_tracker.append(loss.detach().cpu().item()/batch_size)
            loss.backward()
            optimizer.step()
            
    return model,loss_tracker,criterion,optimizer

#evaluate, keep track of dict w/ (patient_id -> output of model) Sort by (|output-0.5|), take min and these are "unsure" classifications
def get_patient_scores(model,dataloader):
    patient_scores = {}
    model.eval()
    for image,mask,patient_id in tqdm(dataloader):
        feed_in_data = torch.empty((image.shape[0],3,256,256))
        for i in range(image.shape[0]):
            feed_in_data[i] = torch.stack([image[i],image[i],mask[i]]).squeeze()
        images = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (feed_in_data)
        images = images.cuda()

        output = nn.Softmax(dim=1)(model(images))
        #print(type(output))
        #print(output.shape)
        for i in range(images.shape[0]):
            #print(output[i].shape)
            patient_scores[patient_id[i]] = output[i].cpu().detach()[1].item()

    patient_scores = {k: patient_scores[k] for k in sorted(patient_scores, key=patient_scores.get)}
    return patient_scores


    #Retrains the model given a dict of oracle results
def model_update(model,dataloader,oracle_results,criterion,optimizer,batch_size=32,num_epochs=1):
    model.train()
    #Retrain one epoch
    for epoch in range(num_epochs):
        #make another dataloader w/ oracle results.
        for image,mask,patient_id in tqdm(dataloader):
                feed_in_data = torch.empty((image.shape[0],3,256,256))
                labels = [0]*image.shape[0] 
                for i in range(image.shape[0]):
                    #print(batch_size/2)
                    if patient_id[i] in list(oracle_results.keys()) and oracle_results[patient_id[i]]!=2: #if currently in dict
                        feed_in_data[i] = torch.stack([image[i],image[i],mask[i]]).squeeze()
                        labels[i] = oracle_results[patient_id[i]]
                    else:
                        feed_in_data[i] = torch.stack([image[i],image[i],mask[i]]).squeeze()
                        labels[i] = 2       
                #Normalize the image
                images = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (feed_in_data)
                #Remove all indices that aren't an oracle
                removed_indices = [i for i in range(image.shape[0]) if labels[i]==2]
                dummy = torch.empty((image.shape[0] - len(removed_indices),3,256,256))
                new_labels = [0]*(images.shape[0] - len(removed_indices))
                cur_index = 0
                for i in range(image.shape[0]):
                    if i in removed_indices:
                        continue
                    try:
                        dummy[cur_index] = images[i]
                        new_labels[cur_index] = labels[i]
                    except:
                        print("Error with removing indices")
                    cur_index+=1
                #images = copy.deepcopy(dummy)
                #labels = copy.deepcopy(new_labels)
                dummy = dummy.cuda()
                new_labels = torch.from_numpy(np.array(new_labels)).cuda()
                optimizer.zero_grad()
                if(dummy.shape[0]==0):
                    continue
                y = model(dummy)

                loss = criterion(y,new_labels)
                loss.backward()
                optimizer.step()
    model.eval()
    return model