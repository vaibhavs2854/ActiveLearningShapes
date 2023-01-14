import numpy as np

import torch
torch.use_deterministic_algorithms(True)

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

from floodfill import *
from dataloader import *
from model import *
from oracle import *
from unet import *
import ternausnet.models

from vivek.model import *

#set random seed
#Vivek GAN Playground
#Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from models import G, D, weights_init (Already imported)
#from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn

import argparse

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def vivek_iou(output_mask,ground_mask):
    print("inside iou")
    #shapes are batch_size x 2 x 256 x 256 for output mask, batch_size x 1 x 256 x 256 for ground mask
    if not output_mask.shape[1]==1:
        #output_mask = F.softmax(output_mask,dim=(-1,-2))
        print("Currently doesn't support non-2 channel IOU")
        return -1
    print(ground_mask)
    ground_mask = get_ints(ground_mask).squeeze(1)
    print("max: " + str(torch.count_nonzero(ground_mask)))
    output_mask = get_binary_mask(output_mask) #threshold calculated here
    print("max: " + str(torch.count_nonzero(output_mask)))
    summed = ground_mask + output_mask
    print("max: " + str(torch.count_nonzero(summed)))
    twos = summed - 2
    print("max: " + str(torch.count_nonzero(twos)))
    print("summed shape: " + str(summed.shape))
    num = 256*256 - torch.count_nonzero(twos,dim=(1,2))
    denom = torch.count_nonzero(summed,dim=(1,2))
    print(denom)
    #print(denom)
    num = num
    denom = num + (epsilon)
    print(num.shape)
    print(denom.shape)
    outputs = torch.div(num,denom)
    print(outputs)
    return torch.mean(outputs)

def recalc_iou(output_mask,ground_mask):
    #print('inside new iou')
    output_mask = get_binary_mask(output_mask) #threshold calculated here
    #calculate intersection
    output_mask = output_mask.view(-1)
    ground_mask = ground_mask.view(-1)
    
    summed = output_mask + ground_mask
    twos = summed - 2
#     print("number of 1s in both: " + str(torch.count_nonzero(twos)))
#     print("number of 1s in output: " + str(torch.count_nonzero(output_mask)))
#     print("number of 1s in ground: " + str(torch.count_nonzero(ground_mask)))
    intersection = 256*256*8 - torch.count_nonzero(twos)
    
    #calculate union
    union = torch.count_nonzero(output_mask) + torch.count_nonzero(ground_mask) - intersection
    #print('outside new iou')
    return intersection/union


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 0.0001
    outputs = get_binary_mask(outputs) #threshold calculated here
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).int()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).int()
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    return thresholded.mean().detach().item()  # Or thresholded.mean() if you are interested in average across the batch

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-random_seed")
    parser.add_argument("-save_folder")
    args = parser.parse_args()
    random_seed = int(args.random_seed)
    save_folder = args.save_folder
    save_path = save_path = save_folder[:-1] + "rand=" + str(random_seed) + "/"

    np.random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    #DEFINE BATCH_SIZE
    batch_size = 8

    cbis_trainloader,cbis_testloader = CBIS_DDSM_get_DataLoader(batch_size,2)
    cudnn.benchmark = True
    training_data_loader = cbis_trainloader #replace with data loader from above (in UNET)

    #Initialize model and initialization values
    input_nc = 1
    output_nc = 1
    ngf = ndf = 32
    netG = G(input_nc, output_nc, ngf)
    netG.apply(weights_init)
    netD = D(input_nc, output_nc, ndf)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    real_A = torch.FloatTensor(batch_size, input_nc, 256, 256)
    real_B = torch.FloatTensor(batch_size, output_nc, 256, 256)
    label = torch.FloatTensor(batch_size)
    real_label = 1
    fake_label = 0

    #Push everything onto CUDA
    netD = netD.cuda()
    netG = netG.cuda()
    criterion = criterion.cuda()
    criterion_l1 = criterion_l1.cuda()
    criterion_mse = criterion_mse.cuda()
    real_A = real_A.cuda()
    real_B = real_B.cuda()
    label = label.cuda()

    real_A = Variable(real_A)
    real_B = Variable(real_B)
    label = Variable(label)

    #Setup ADAM optimizer - REPLACE
    lr = 0.0002
    beta1 = 0.5
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    lamb = 150

    vivek_loss_tracker = []
    vivek_iou_tracker = []

    check_flag = False
    epsilon = 0.0001
    num_epochs = 300
    iou_threshold = 0.6

    for i in range(num_epochs):
        epoch = i+1
        count = 0
        metric = 0.0
        #Training Code
        for iteration, batch in enumerate(training_data_loader, 1):
            ############################
            # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
            ###########################
            # train with real
            
    #         new_batch = torch.cat((batch[0],batch[0],batch[0]),dim=1)
    #         print(new_batch.shape)
    #         print("done")
    #         break
            netD.volatile = True
            netD.zero_grad()
            #print(batch[0].shape)
            with torch.no_grad():
                real_a_cpu, real_b_cpu = batch[0], batch[1]
                real_A.resize_(real_a_cpu.size()).copy_(real_a_cpu)
                real_B.resize_(real_b_cpu.size()).copy_(real_b_cpu)

            output = netD(torch.cat((real_A, real_B), 1))
            with torch.no_grad():
                label.resize_(output.size()).fill_(real_label)
            err_d_real = criterion(output, label)
            # print (err_d_real)
            err_d_real.backward()
            d_x_y = output.data.mean()

            # train with fake
            fake_b = netG(real_A)
            output = netD(torch.cat((real_A, fake_b.detach()), 1))
            with torch.no_grad():
                label.resize_(output.size()).fill_(fake_label)
            err_d_fake = criterion(output, label)
            # print (err_d_fake)
            err_d_fake.backward()
            d_x_gx = output.data.mean()

            err_d = (err_d_real + err_d_fake) / 2.0
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
            ###########################
            netG.zero_grad()
            netD.volatile = True
            output = netD(torch.cat((real_A, fake_b), 1))
            label.data.resize_(output.size()).fill_(real_label)
            err_g = criterion(output, label) + lamb * dice_loss(fake_b, real_B)
            err_g.backward()
            d_x_gx_2 = output.data.mean()
            optimizerG.step()
            
            metric += iou_pytorch(fake_b,real_B) * batch[0].shape[0]
            count += batch[0].shape[0]
            
            if check_flag:
                print("start debug")
                print(recalc_iou(fake_b,real_B))
                print("end debug")

            #Print epoch info
            #print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}".format(
            #        epoch, iteration, len(training_data_loader), err_d.item(), err_g.item(), d_x_y, d_x_gx, d_x_gx_2))
            vivek_loss_tracker.append(err_d.item())
        vivek_iou_tracker.append(metric/count)
        if(epoch%10==0):
            print("IOU at epoch " + str(i) + ": " + str(metric/count))
        if((metric/count)> iou_threshold):
            #save model and break out of loop. Maybe print which epoch it finished at
            #save_path = "/usr/xtmp/vs196/mammoproj/Code/ActiveLearning/vivek_run/12_28/"
            #save_path = save_folder
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_name = f"model_iou={(metric/count):.5f}.pth"
            g_file = save_path + "g_" + file_name
            d_file = save_path + "d_" + file_name
            torch.save(netG,g_file)
            torch.save(netD,d_file)
            print(f"Found a good model. Has IOU: {(metric/count):.5f}. Saving models to {save_path}")
            iou_threshold = (metric/count)
        print(f"done with epoch: {epoch}")

        #Eval Code

        #Run Code:
        #modelD, modelG, criterionD, criterionG, metric_trackerGD = vivek.model_update(modelD, modelG, criterionD, criterionG, dataloader,num_epochs=5)
    #save the graph plots
    plt.plot(vivek_loss_tracker)
    plt.savefig(f"{save_path}loss.png")
    plt.clf()
    plt.plot(vivek_iou_tracker)
    plt.savefig(f"{save_path}iou.png")
    plt.clf()
    print(f"Saved figures to {save_path}.")


    #Run info
    # 1597986 compsci-g vivekrun    vs196  R       0:34      1 gpu-compute5 (10)
    # 1597987 compsci-g vivekrun    vs196  R       0:25      1 gpu-compute6 (20)
    # 1597988 compsci-g vivekrun    vs196  R       0:21      1 linux46 (30)
    # 1597989 compsci-g vivekrun    vs196  R       0:17      1 linux48 (40)
    # 1597990 compsci-g vivekrun    vs196  R       0:03      1 gpu-compute4 (50)


    # 1599306 compsci-g vivekrun    vs196  R       0:33      1 gpu-compute5 (60)
    # 1599307 compsci-g vivekrun    vs196  R       0:27      1 gpu-compute6 (70)
    # 1599308 compsci-g vivekrun    vs196  R       0:22      1 linux46 (80)
    # 1599309 compsci-g vivekrun    vs196  R       0:19      1 linux48 (90)
    # 1599310 compsci-g vivekrun    vs196  R       0:02      1 linux47 (100)

