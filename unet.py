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
#import torchvision.transforms.InterpolationMode

import ternausnet.models
import albumentations as A
import albumentations.augmentations.functional as AF
from albumentations.pytorch import ToTensorV2
from skimage import exposure
from skimage.transform import resize
from scipy import signal
import pickle

def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask

class Pet_get_data(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform,mask_transform):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        #Get image, convert to color
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #get mask, process to get segmentation
        mask = cv2.imread(
            os.path.join(self.masks_directory, image_filename.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED,
        )
        #combines the border/subject into one
        mask = preprocess_mask(mask)
        #performs transform
        return self.transform(image),self.mask_transform(mask)

class CBISDDSM_get_data(Dataset):
    def __init__(self,image_filepaths,image_transform,data_aug=True,has_weights=True):
        super().__init__()
        self.image_filepaths = image_filepaths
        self.image_transform = image_transform
        self.data_aug = data_aug
        self.has_weights = has_weights

    def __len__(self):
        return len(self.image_filepaths)

    def __getitem__(self,idx):
        filepath = self.image_filepaths[idx]
        #print("Filepath: " + filepath)
        arr_and_mask = np.load(filepath)
        copy_arr_mask = arr_and_mask.copy()
        if self.data_aug:
            copy_arr_mask = random_flip(copy_arr_mask, 0, True)
            copy_arr_mask = random_flip(copy_arr_mask, 1, True)
            copy_arr_mask = random_rotate_90(copy_arr_mask, True)
            copy_arr_mask = random_rotate_90(copy_arr_mask, True)
            copy_arr_mask = random_rotate_90(copy_arr_mask, True)
        arr = copy_arr_mask[0,:,:].copy()
        mask = copy_arr_mask[1,:,:].copy()
        if self.has_weights:
            weights = copy_arr_mask[2,:,:].copy()
        arr = exposure.equalize_hist(arr) #histogram equalization, remove if u want
        #arr = np.stack([arr,arr,arr])
        #mask = np.stack([mask,mask,mask])
        #no need to preprocess
        #print("INSIDE")
        #print(arr.shape)
        
        image = self.image_transform(arr)
        #image = our_transform(image)
        #print(image.shape)
        #print("OUTSIDE")
        mask_label = self.image_transform(mask)
        #mask_label = our_transform(mask_label)
        if self.has_weights:
            weights_label = self.image_transform(weights)
            #weights_label = our_transform(weights_label)
        #a transform
        # print(arr.shape)
        # print(mask.shape)
        
        # transformed = self.a_transform(image=arr, mask=mask)
        # a_image = transformed["image"]
        # a_mask = transformed["mask"]
        if self.has_weights:
            return image,mask_label,weights_label
        return image,mask_label

#Buffer is 32 default
def our_transform(arr,buffer=32):
    out = torch.zeros(256,256)
    center = transforms.Resize((256 - buffer*2,256 - buffer*2))(arr)
    out[buffer:256-buffer,buffer:256-buffer] = center
    return torch.unsqueeze(out,0)

def unet_dataloader(train_images_filepaths,batch_size,num_workers):
    transforms_arr = [transforms.ToTensor(),transforms.Resize((256,256))]
    transform = transforms.Compose(transforms_arr)
    trainset = CBISDDSM_get_data(train_images_filepaths,transform,data_aug=True,has_weights=True)
    trainloader = DataLoader(trainset,batch_size=batch_size,num_workers=num_workers)
    return trainloader


def CBIS_DDSM_get_DataLoader(train_images_filenames,test_images_filenames,train_images_directory,test_images_directory,batch_size,num_workers,has_weights=False):
    print("bench1")
    transforms_arr = [transforms.ToTensor(),transforms.Resize((256,256))]
    transform = transforms.Compose(transforms_arr)
    print("bench2")
    trainset = CBISDDSM_get_data(train_images_filenames,train_images_directory,transform,data_aug=True,has_weights=has_weights)
    trainloader = DataLoader(trainset,batch_size=batch_size,num_workers=num_workers)
    print("bench3")
    testset = CBISDDSM_get_data(test_images_filenames,test_images_directory,transform,data_aug=False,has_weights=has_weights) #boolean is data aug
    testloader = DataLoader(testset,batch_size=batch_size,num_workers=num_workers)
    print("bench4")
    return trainloader,testloader

#Returns a dataloader object for training
def Pet_get_DataLoader(train_images_filenames, test_images_filenames, images_directory, masks_directory,batch_size,num_workers):
    transform1 = [(transforms.ToTensor()),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),transforms.Resize((256,256))]
    transform2 = [(transforms.ToTensor()),transforms.Resize((256,256))]

    image_transform = transforms.Compose(transform1)
    mask_transform = transforms.Compose(transform2)
    trainset = Pet_get_data(train_images_filenames, images_directory, masks_directory, image_transform,mask_transform)
    testset = Pet_get_data(test_images_filenames, images_directory, masks_directory, image_transform,mask_transform)
    trainloader = DataLoader(trainset,batch_size=batch_size,num_workers=num_workers)
    testloader = DataLoader(testset,batch_size=batch_size,num_workers=num_workers)
    return trainloader,testloader

class modified_UNet(nn.Module):
    def __init__(self,pretrained_model):
        super().__init__()
        self.premodel = pretrained_model
        self.flayer = nn.ConvTranspose2d(1, 3, kernel_size=1, stride=1)
    def forward(self,x):
        upsampled = self.flayer(x)
        return self.premodel.forward(upsampled)

def create_model(model_flag,num_classes):
    model = getattr(ternausnet.models, "UNet16")(num_classes=num_classes,pretrained=True)
    #modified_model = modified_UNet(model)
    #return model
    #return UNet2(3,1)
    return model if model_flag else UNet()

def get_ints(mask):
    return torch.where(mask>0.2,1,0)

#Modify to change the relative weighting of border/other
def get_weight(mask,weight_ratio=0.5):
    return torch.where(mask>0.2,1.0,weight_ratio)


#Ground truth mask and output mask are both floats.
#Ground truth is 0/1 float, while output is a legit float
def intersection_over_union(output_mask,ground_mask):
    #shapes are batch_size x 2 x 256 x 256 for output mask, batch_size x 1 x 256 x 256 for ground mask
    if output_mask.shape[1]==2:
        output_mask = F.softmax(output_mask,dim=1)[:,1,:,:]
    else:
        #output_mask = F.softmax(output_mask,dim=(-1,-2))
        print("Currently doesn't support non-2 channel IOU")
        return -1
    ground_mask = get_ints(ground_mask).squeeze(1)
    output_mask = get_binary_mask(output_mask) #threshold calculated here
    summed = ground_mask + output_mask
    twos = summed - 2
    num = 256*256 - torch.count_nonzero(twos,dim=(1,2))
    denom = torch.count_nonzero(summed,dim=(1,2))
    outputs = torch.div(num,denom)
    return torch.mean(outputs)

def weightedpixelcros(output,target,weights):
    batch_size = output.shape[0]
    H = output.shape[2]
    W = output.shape[3]
    out = F.log_softmax(output,dim=1)
    target = get_ints(target)
    out = out.gather(1, target.view(batch_size, 1, H,W))
    wout = (out * weights).view(batch_size,-1)
    wgloss = wout.sum(1) / weights.view(batch_size, -1).sum(1)
    wgloss = -1.0 * wgloss.mean()
    return wgloss

def get_two_channels(output):
    ch1 = torch.where(output>0,1,0)
    ch2 = torch.where(output>0,0,1)
    out = torch.stack([ch1,ch2],dim=1)
    return out

def convert_to_3channel(x):
    return torch.tile(x,(1,3,1,1))

def unet_update_model(model,dataloader,num_epochs=10,has_weights=True,weight_ratio=0.5):    
    criterion = weightedpixelcros
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_tracker = [] #plot loss
    metric_tracker = []

    model.train()
    #training loop
    for epoch in range(num_epochs):
        tr_loss = 0.0
        count = 0
        metric = 0.0
        for grouped in tqdm(train_loader):
            #push to cuda
            images = grouped[0]
            images = images.float()
            labels = grouped[1]
            if has_weights:
                weights = get_weight(grouped[2],weight_ratio=weight_ratio)
                weights = weights.cuda()
            #print(images.shape)
            #images = transforms.Normalize(mean=(0.445,), std=(0.269,))(images) #normalize
            images = images.cuda() if (pet_flag or og_unet_flag) else convert_to_3channel(images).cuda()
            #Include bottom line (normalize after convert to 3-channels)
            images = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (images)
            #images = convert_to_3channel(images).cuda() if not pet_flag else images.cuda()
            #print(images.shape)
            #labels = labels.long()
            #labels = labels.squeeze(1)
            labels = labels.cuda()
            optimizer.zero_grad()
            #print("IMAGE SHAPE: ")
            #print(images.shape)
            y = model(images)
            #y = y.squeeze(1)
            #y = y.cuda()
            if has_weights:
                loss = criterion(y,labels,weights)
            else:
                loss = criterion(y,labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.detach().cpu().item()
            count += images.shape[0]
            metric += intersection_over_union(y,labels)*images.shape[0]
        loss_tracker.append(tr_loss/(count))
        metric_tracker.append(metric/count)
    return model,loss_tracker,metric_tracker

#Threshold value changed here
def get_binary_mask(mask):
    return torch.where(mask > 0.5, 1, 0)

def get_visualizations(model,loader,pet_flag=False,binary_flag=False,og_unet_flag=False,has_weights=False):
    images_arr = []
    with torch.no_grad():
        i = 0
        for grouped in tqdm(loader):
            images = grouped[0]
            labels = grouped[1]
            images = images.float()
            images = images.cuda() if (pet_flag or og_unet_flag) else convert_to_3channel(images).cuda()
            #images = convert_to_3channel(images).cuda() if not pet_flag else images.cuda()
            y = model(images)
            for i in range(images.shape[0]):
                one_image = images[i][0,:,:] if not pet_flag else images[i]
                one_label = labels[i].squeeze(0)
                if has_weights:
                    one_predicted_mask = F.softmax(y[i],dim=0)
                    images_arr.append(get_visualized_image(one_image,one_label,one_predicted_mask[1,:,:],binary_flag=binary_flag))
                else:
                    one_predicted_mask = y[i].squeeze(0)
                    images_arr.append(get_visualized_image(one_image,one_label,one_predicted_mask,binary_flag=binary_flag))
            break
    return images_arr
        
def get_visualized_image(torch_image,torch_mask,torch_pred_mask,binary_flag=False):
    image = torch_image.cpu().numpy()
    mask = torch_mask.cpu().numpy()
    binary_pred_mask = get_binary_mask(torch_pred_mask.cpu()).numpy()
    pred_mask = torch_pred_mask.cpu().numpy()
    # print("MAX")
    # print(np.max(image))
    # print(np.max(mask))
    # print(np.max(pred_mask))
    # print("MIN")
    # print(np.min(image))
    # print(np.min(mask))
    # print(np.min(pred_mask))

    # print(image.shape)
    # print(mask.shape)
    # print(pred_mask.shape)
    if len(image.shape)==2:
        stacked_image = np.stack([image,image,image])
        #print("ENTERED ENTERED ENTERED")
    else:
        stacked_image = image
        #print("NOT ENTERED NOT ENTERED")
    #print(stacked_image.shape)
    original_img = np.transpose(stacked_image, [1,2,0])

    rescaled_mask = mask - np.amin(mask)
    rescaled_mask = rescaled_mask / np.amax(rescaled_mask)
    # print(rescaled_mask.shape)
    # print(np.min(rescaled_mask))
    # print(np.max(rescaled_mask))
    testing = np.uint8(255*rescaled_mask)
    # print(np.min(rescaled_mask))
    # print(np.max(rescaled_mask))
    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[...,::-1]
    overlayed_img_1 = 0.1 * original_img + 0.5 * heatmap

    rescaled_pred_mask= pred_mask - np.amin(pred_mask)
    rescaled_pred_mask = rescaled_pred_mask / np.amax(rescaled_pred_mask)
    heatmap1 = cv2.applyColorMap(np.uint8(255*rescaled_pred_mask), cv2.COLORMAP_JET)
    heatmap1 = np.float32(heatmap1) / 255
    heatmap1 = heatmap1[...,::-1]
    overlayed_img_2 = 0.1 * original_img + 0.5 * heatmap1

    rescaled_binary_pred_mask= binary_pred_mask - np.amin(binary_pred_mask)
    rescaled_binary_pred_mask = rescaled_binary_pred_mask / np.amax(rescaled_binary_pred_mask)
    heatmap2 = cv2.applyColorMap(np.uint8(255*rescaled_binary_pred_mask), cv2.COLORMAP_JET)
    heatmap2 = np.float32(heatmap2) / 255
    heatmap2 = heatmap2[...,::-1]
    overlayed_img_3 = 0.1 * original_img + 0.5 * heatmap2

    #original: 0.5,0.3
    if binary_flag:
        return original_img,overlayed_img_1,overlayed_img_2,overlayed_img_3
    else:
        return original_img,overlayed_img_1,overlayed_img_2

def get_collage(model,loader,pet_flag=False,binary_flag=False,og_unet_flag=False,has_weights=False):
    images = get_visualizations(model,loader,pet_flag=pet_flag,binary_flag=binary_flag,og_unet_flag=og_unet_flag,has_weights=has_weights)
    num_cols = 4 if binary_flag else 3
    f, axarr = plt.subplots(10,num_cols)
    for i,image in enumerate(images):
        axarr[i,0].imshow(image[0])
        axarr[i,1].imshow(image[1])
        axarr[i,2].imshow(image[2])
        if binary_flag:
            axarr[i,3].imshow(image[3])
            # if i==8:
            #     print("DEBUGGING MAX VALUE FOR THRESHOLD")
            #     print(np.max(image[3]))
            #     print(np.min(image[3]))
            #     imsave("/usr/xtmp/vs196/mammoproj/Data/unet_viz/model_preds_viz/ternausnet_1/thresholdtesting.png",image[3])
        if(i==9):
            break
    plt.savefig("/usr/xtmp/vs196/mammoproj/Code/UNet/Pretrained_Loss/pretrained_30epoch_811_1_viz.png")
    plt.clf()

    #big arr dimensions
    c_row = images[0][0].shape[0]
    c_col = images[0][0].shape[1]
    collage = np.zeros((c_row*10,c_col*num_cols,3))
    #print('hi')
    for i,image in enumerate(images):
        # print(image[0].shape)
        # print(image[1].shape)
        # print(image[2].shape)
        # print(collage[i*c_row:(i+1)*c_row,0*c_col:1*c_col].shape)
        collage[i*c_row:(i+1)*c_row,0*c_col:1*c_col,:] = image[0]
        collage[i*c_row:(i+1)*c_row,1*c_col:2*c_col,:] = image[1]
        collage[i*c_row:(i+1)*c_row,2*c_col:3*c_col,:] = image[2]
        if binary_flag:
            collage[i*c_row:(i+1)*c_row,3*c_col:4*c_col,:] = image[3]
        if i==9:
            break
    #print(np.max(collage))
    #print(np.min(collage))
    plt.figure(figsize = (num_cols,10))
    plt.imshow(collage)
    plt.savefig("/usr/xtmp/vs196/mammoproj/Code/UNet/Pretrained_Loss/pretrained_30epoch_811_1_binary_viz_old_testing.png")
    plt.clf()

def save_info(model,loss_tracker,test_loss_tracker,epoch,pet_flag=False,og_unet_flag=False,has_weights=False,metric=None,save_path=None):
    #Pets is pet datset, Old/Ternaus are CBIS-DDSM. Old is my UNET, Ternaus is the TernausNet. Ternaus takes in 3-channel, Old takes in 1
    pet = "Pets/"
    old = "Old/"
    normal = "Ternaus/"
    weight = "Weight/"
    day = "0811"
    run_num = 1
    #CBIS run_num is 1
    #Pet run_num is 1
    run_type = pet if pet_flag else (old if og_unet_flag else (weight if has_weights else normal))
    #model_benchmarks_dir = "/usr/xtmp/vs196/mammoproj/SavedModels/UNet/" + run_type + "finalmodel_" + day + "_" + str(run_num) + "/"
    model_benchmarks_dir = save_path + "Model/"
    if not os.path.exists(model_benchmarks_dir):
        os.makedirs(model_benchmarks_dir)
    if epoch == -1:
        torch.save(model,model_benchmarks_dir + "unetmodel_" + "FINAL.pth")
    else:
        torch.save(model,model_benchmarks_dir + "unetmodel_" + str(epoch) + ".pth")
    #torch.save(model,"/usr/xtmp/vs196/mammoproj/SavedModels/UNet/" + run_type + "finalmodel_" + day + "_" + str(run_num) + ".pth")

    fig = plt.plot(loss_tracker)
    fig2 = plt.plot(test_loss_tracker)
    #loss_benchmarks_dir = "/usr/xtmp/vs196/mammoproj/Code/UNet/Pretrained_Loss/" + run_type + "pretrained_tnt_loss_" + day + "_" + str(run_num) + "/"
    loss_benchmarks_dir = save_path + "Losses/"
    if not os.path.exists(loss_benchmarks_dir):
        os.makedirs(loss_benchmarks_dir)
    if epoch==-1:
        plt.savefig(loss_benchmarks_dir + "loss_FINAL.png")
    else:
        plt.savefig(loss_benchmarks_dir + "loss_" + str(epoch) + ".png")
    plt.clf()

    if has_weights:
        fig = plt.plot(metric)
        #metric_benchmarks_dir = "/usr/xtmp/vs196/mammoproj/Code/UNet/Pretrained_Metric/" + run_type + "pretrained_metric_" + day + "_" + str(run_num) + "/"
        metric_benchmarks_dir = save_path + "IOU/"
        if not os.path.exists(metric_benchmarks_dir):
            os.makedirs(metric_benchmarks_dir)
        if epoch==-1:
            plt.savefig(metric_benchmarks_dir + "IOU_FINAL.png")
        else:
            plt.savefig(metric_benchmarks_dir + "IOU_" + str(epoch) + ".png")
        plt.clf()


# Ref: https://stackoverflow.com/a/42579291/7521428
def convolution2d(image, kernel, bias=0):
  m, n = kernel.shape
  if (m == n):
    y, x = image.shape
    y = y - m + 1
    x = x - m + 1
    new_image = np.zeros((y,x))
    for i in range(y):
      for j in range(x):
        new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel) + bias
  return new_image.astype(int)

#Image is original, non-convoluted mask. Returns a bordered mask
def find_border(mask):
    assert len(mask.shape)==2, "Mask is not 2d"
    our_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
    #conv_result = convolution2d(mask,our_kernel)
    conv_result = signal.convolve2d(mask,our_kernel,mode='same',fillvalue=0)
    borders = np.zeros_like(conv_result)
    for i in range(conv_result.shape[0]):
        for j in range(conv_result.shape[1]):
            borders[i][j] = 1 if (conv_result[i][j]==9 or conv_result[i][j]==0) else 0
    return borders
 
#Takes in bordered mask, and returns expanded mask
def expand_border(border):
  expanded_borders = np.zeros_like(border)
  border_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
  conv_border = signal.convolve2d(border, border_kernel, mode='same', fillvalue=1)
  for i in range(conv_border.shape[0]):
    for j in range(conv_border.shape[1]):
      expanded_borders[i][j] = 1 if conv_border[i][j]==9 else 0
  return expanded_borders

def save_expanded_borders(filenames_list, load_dir, save_dir_dict, border_sizes):
    for filename in tqdm(filenames_list):
        filepath = os.path.join(load_dir,filename)
        arr_and_mask = np.load(filepath)
        arr = arr_and_mask[0,:,:].copy()
        mask = arr_and_mask[1,:,:].copy()
        arr = cv2.resize(arr, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        mask =  cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

        expanded_mask = find_border(mask.copy())
        times_expanded = 0
        for bs in border_sizes:
            num_expansions = bs - times_expanded
            for j in range(times_expanded):
                expanded_mask = expand_border(expanded_mask)
            times_expanded += num_expansions
            to_save = np.stack([arr,mask,expanded_mask])
            save_path = os.path.join(save_dir_dict[bs],filename)
            np.save(save_path,to_save)

def populate_expanded_borders(train_filenames,test_filenames,train_dir,test_dir,border_sizes=[10]):
    border_sizes.sort()
    train_save_dirs = dict()
    test_save_dirs = dict()
    for border_size in border_sizes:
        train_save_dirs[border_size] = f"/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm_border_{border_size}/Train"
        test_save_dirs[border_size] = f"/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm_border_{border_size}/Test"
        if not os.path.exists(train_save_dirs[border_size]):
            os.makedirs(train_save_dirs[border_size])
        if not os.path.exists(test_save_dirs[border_size]):
            os.makedirs(test_save_dirs[border_size])

    save_expanded_borders(train_filenames, train_dir, train_save_dirs, border_sizes=border_sizes)
    save_expanded_borders(test_filenames, test_dir, test_save_dirs, border_sizes=border_sizes)

    return train_save_dirs, test_save_dirs
    #Done with test

def hyperparameter_run(cbis_train_filenames,cbis_test_filenames,has_weights=True):
    with open("/usr/xtmp/vs196/mammoproj/Data/train_save_dirs.pkl","rb") as f:
        train_dict = pickle.load(f)
    with open("/usr/xtmp/vs196/mammoproj/Data/test_save_dirs.pkl","rb") as f:
        test_dict = pickle.load(f)
    for border_size in [0,1,5,10]:
        train_dir = train_dict[border_size]
        test_dir = test_dict[border_size]
        cbis_trainloader,cbis_testloader = CBIS_DDSM_get_DataLoader(cbis_train_filenames,cbis_test_filenames,train_dir,test_dir,16,2,has_weights=has_weights)
        for weight_ratio in [0.05, 0.1, 0.2, 0.5]:
            save_path =  f"/usr/xtmp/vs196/mammoproj/SavedModels/HyperparameterUNet_nobuffer/unet_{border_size}_{weight_ratio}/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model,loss_tracker,test_loss_tracker,test_metric_tracker = run_model(cbis_trainloader,cbis_testloader,num_epochs=20,has_weights=has_weights,weight_ratio=weight_ratio,save_path=save_path)
            save_info(model,loss_tracker,test_loss_tracker,-1,has_weights=has_weights,metric=test_metric_tracker,save_path = save_path)
            print("Finished saved to: " + save_path)

if __name__ == "__main__":
    # dataset_directory = 'pet_dataset'
    # root_directory = os.path.join(dataset_directory)
    # images_directory = os.path.join(root_directory, "images")
    # masks_directory = os.path.join(root_directory, "annotations", "trimaps")

    # images_filenames = list(sorted(os.listdir(images_directory)))

    # correct_images_filenames = []
    # for i in tqdm(images_filenames):
    #   if cv2.imread(os.path.join(images_directory,i)) is not None:
    #     correct_images_filenames.append(i)

    #correct_images_filenames = [i for i in images_filenames if cv2.imread(os.path.join(images_directory, i)) is not None]


    #Comment out for now - not working with CBIS-DDSM atm
    image_directory = "/usr/xtmp/vs196/mammoproj/Data/"

    cbis_train_filenames = []
    with open('unet_trainfiles.txt','r') as train_file:
        for line in train_file.readlines():
            cbis_train_filenames.append(line[2:-1])

    cbis_test_filenames = []
    with open('unet_testfiles.txt','r') as test_file:
        for line in test_file.readlines():
            cbis_test_filenames.append(line[2:-1])

    print(cbis_train_filenames[0])
    print(cbis_test_filenames[0])

    has_weights=True

    hyperparameter_run(cbis_train_filenames,cbis_test_filenames,has_weights=True)

    train_dir = "/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm_border/Train" if has_weights else "/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm/Train"
    test_dir = "/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm_border/Test" if has_weights else "/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm/Test"


    #cbis_trainloader,cbis_testloader = CBIS_DDSM_get_DataLoader(cbis_train_filenames,cbis_test_filenames,train_dir,test_dir,16,2,has_weights=has_weights)

    #print("gotten")

    #RUN POPULATE
    # train_save_dirs, test_save_dirs = populate_expanded_borders(cbis_train_filenames,cbis_test_filenames,train_dir,test_dir,border_sizes=[0,1,5,10])

    # with open("/usr/xtmp/vs196/mammoproj/Data/train_save_dirs.pkl", 'wb') as f:
    #     pickle.dump(train_save_dirs, f)
    # with open("/usr/xtmp/vs196/mammoproj/Data/test_save_dirs.pkl", 'wb') as f:
    #     pickle.dump(test_save_dirs, f)

    # print("files are saved in: ", train_save_dirs, test_save_dirs)

    #print("done")

    # model,loss_tracker,test_loss_tracker,test_metric_tracker = run_model(cbis_trainloader,cbis_testloader,num_epochs=20,has_weights=has_weights)

    # metric = test_metric_tracker if has_weights else None
    #save relevant info
    # save_info(model,loss_tracker,test_loss_tracker,-1,has_weights=has_weights,metric=metric)

    # # sys.exit(0)
    #model_path = "/usr/xtmp/vs196/mammoproj/SavedModels/UNet/Ternaus/finalmodel_0809_2/FINAL.pth"
    #model_path = "/usr/xtmp/vs196/mammoproj/SavedModels/UNet/Weight/finalmodel_0809_3/FINAL.pth"
    #model_path = "/usr/xtmp/vs196/mammoproj/SavedModels/UNet/Weight/finalmodel_0810_2/FINAL.pth" #viz showed to general on Tuesday 8/10
    #model_path = "/usr/xtmp/vs196/mammoproj/SavedModels/UNet/Weight/finalmodel_0811_1/FINAL.pth"


    #trained_model = torch.load(model_path)
    #get_collage(trained_model,cbis_testloader,binary_flag=True,og_unet_flag=False,has_weights=has_weights)

    # #START HERE
    # print("start")
    # trained_model = torch.load("/usr/xtmp/vs196/mammoproj/SavedModels/UNet/finalmodel_08_03_v2.pth")
    # im1,im2,im3 = get_visualizations(trained_model,cbis_testloader)
    # print("finish1")

    # print(np.max(im1))
    # print(np.max(im2))
    # print(np.max(im3))

    # imsave("/usr/xtmp/vs196/mammoproj/Data/unet_viz/model_preds_viz/ternausnet_1/im1.png",im1)
    # imsave("/usr/xtmp/vs196/mammoproj/Data/unet_viz/model_preds_viz/ternausnet_1/im2.png",im2)
    # imsave("/usr/xtmp/vs196/mammoproj/Data/unet_viz/model_preds_viz/ternausnet_1/im3.png",im3)
    # #END HERE


    #plt.clf()
    #figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 24))
    #ax[0] = im1
    #ax[1] = im2
    #x[2] = im3
    #print("finish2")
    #plt.tight_layout()
    #print("finish3")
    #plt.savefig("/usr/xtmp/vs196/mammoproj/Code/UNet/viz1.png")

    #Code for the Pet Dataset on U-Net Model
    # root_directory = "/usr/xtmp/vs196/mammoproj/Code/UNet/pet_dataset/pet_dataset"
    # images_directory = os.path.join(root_directory, "images")
    # masks_directory = os.path.join(root_directory, "annotations", "trimaps")

    # print(images_directory)
    # print(masks_directory)

    # images_filenames = list(sorted(os.listdir(images_directory)))
    # correct_images_filenames = [i for i in images_filenames if cv2.imread(os.path.join(images_directory, i)) is not None]

    # random.seed(42)
    # random.shuffle(correct_images_filenames)

    # train_images_filenames = correct_images_filenames[:6000]
    # val_images_filenames = correct_images_filenames[6000:-10]
    # test_images_filenames = images_filenames[-10:]

    # print(len(train_images_filenames), len(val_images_filenames), len(test_images_filenames))


    # batch_size = 32
    # num_workers = 2
    # pet_trainloader,pet_testloader = Pet_get_DataLoader(train_images_filenames, val_images_filenames, images_directory, masks_directory,batch_size,num_workers)
    # # # model,train_loss,test_loss = run_model(pet_trainloader,pet_testloader,num_epochs=30,pet_flag=True)
    # # # save_info(model,train_loss,test_loss,pet_flag=True)

    # model_path = "/usr/xtmp/vs196/mammoproj/SavedModels/UNet/Pets/finalmodel_0805_3.pth"
    # trained_model = torch.load(model_path)
    # get_collage(trained_model,pet_testloader,pet_flag=True,binary_flag=True)



    #Debugging
    # a_transform = A.Compose([\
    #     A.Resize(256, 256),\
    #     A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),\
    #     A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),\
    #     A.Normalize(mean=(0.445,), std=(0.269,)),\
    #     ToTensorV2()])
    # image_transforms = [transforms.ToTensor(),transforms.Resize((256,256))]
    # image_transform = transforms.Compose(image_transforms)
    # mask_transforms = [transforms.ToTensor(),transforms.Resize((256,256))]
    # mask_transform = transforms.Compose(mask_transforms)
    # for i,(images,labels) in enumerate(CBISDDSM_get_data(cbis_train_filenames,train_dir,image_transform,mask_transform,a_transform)):
    #     print(i)
    #     try:
    #         print(images.shape)
    #         print(labels.shape)
    #     except:
    #         print("ERROR" + str(i))
    #         sys.exit(1)

    # sys.exit(0)

    #A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),\

    #Border Code
    #get some mask

    #add_borders(mask)

    #open some border file and show it
    #arr_mask_border = np.load("/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm_border/Train/DP_00061_RIGHT_MLO_1.npy")
    #arr_mask_border = np.load("/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm_border/Test/DP_01590_LEFT_MLO_1.npy")
    # arr_mask_border = np.load("/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm_border_10/Test/DP_01590_LEFT_MLO_1.npy")
    # #arr_mask_border = np.load("/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm_border_10/Train/DP_00001_LEFT_CC_1.npy")
    #arr_mask_border = np.load("/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm_border_0/Train/DP_01860_RIGHT_CC_1.npy")
    # arr_mask_border = np.load("/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm_border_5/Train/DP_01946_RIGHT_CC_1.npy")
    # mask = arr_mask_border[1,:,:]
    # print(np.min(mask))
    # print(np.max(mask))
    # print(arr_mask_border.shape)
    # plt.imsave("/usr/xtmp/vs196/mammoproj/Data/border_viz.png",np.transpose(arr_mask_border,[1,2,0]))
    #plt.imsave("/usr/xtmp/vs196/mammoproj/Data/border_viz.png",arr)

