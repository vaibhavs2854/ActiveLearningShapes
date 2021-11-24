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

class InhouseGetData(Dataset):
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

class CBISDDSMGetData(Dataset):
    def __init__(self,image_filenames,home_dir,image_transform,data_aug=False,has_weights=False):
        super().__init__()
        self.image_filenames = image_filenames
        self.home_dir = home_dir
        self.image_transform = image_transform
        self.data_aug = data_aug
        self.has_weights = has_weights

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self,idx):
        filename = self.image_filenames[idx]
        filepath = os.path.join(self.home_dir,filename)
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
    trainset = InhouseGetData(train_images_filepaths,transform,data_aug=True,has_weights=True)
    trainloader = DataLoader(trainset,batch_size=batch_size,num_workers=num_workers)
    return trainloader


def CBIS_DDSM_get_DataLoader(batch_size,num_workers,has_weights=True,train_images_filename='unet_trainfiles.txt',test_images_filename='unet_testfiles.txt',train_images_directory="/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm_border/Train/",test_images_directory="/usr/xtmp/vs196/mammoproj/Data/unet_cbisddsm_border/Test/"):
    train_images_filenames = []
    with open(train_images_filename,'r') as train_file:
        for line in train_file.readlines():
            train_images_filenames.append(line[2:-1])

    test_images_filenames = []
    with open(test_images_filename,'r') as test_file:
        for line in test_file.readlines():
            test_images_filenames.append(line[2:-1])
    
    transforms_arr = [transforms.ToTensor(),transforms.Resize((256,256))]
    transform = transforms.Compose(transforms_arr)

    trainset = CBISDDSMGetData(train_images_filenames,train_images_directory,transform,data_aug=True,has_weights=has_weights)
    trainloader = DataLoader(trainset,batch_size=batch_size,num_workers=num_workers)
    
    testset = CBISDDSMGetData(test_images_filenames,test_images_directory,transform,data_aug=False,has_weights=has_weights) #boolean is data aug
    testloader = DataLoader(testset,batch_size=batch_size,num_workers=num_workers)
    return trainloader,testloader

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

def unet_update_model(model,inhouse_dataloader,num_epochs=10,has_weights=True,weight_ratio=0.5):    
    criterion = weightedpixelcros
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    loss_tracker = [] #plot loss
    metric_tracker = []

    model.train()
    #training loop
    for epoch in range(num_epochs):
        tr_loss = 0.0
        count = 0
        metric = 0.0
        for grouped in tqdm(inhouse_dataloader):
            #push to cuda
            images = grouped[0]
            images = images.float()
            labels = grouped[1]
            if has_weights:
                weights = get_weight(grouped[2],weight_ratio=weight_ratio)
                weights = weights.cuda()
            #print(images.shape)
            #images = transforms.Normalize(mean=(0.445,), std=(0.269,))(images) #normalize
            images = convert_to_3channel(images).cuda()
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

#TODO: Add in test dataloader
def unet_update_model_multi_dataloader(model,inhouse_dataloader,cbis_dataloader,num_epochs=10,has_weights=True,weight_ratio=0.5):    
    criterion = weightedpixelcros
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    loss_tracker = [] #plot loss
    metric_tracker = []

    model.train()

    bigger_dataloader = inhouse_dataloader
    smaller_dataloader = cbis_dataloader

    if(len(inhouse_dataloader)<len(cbis_dataloader)):
        (bigger_dataloader,smaller_dataloader) = (smaller_dataloader,bigger_dataloader)
    
    smaller_iter = iter(smaller_dataloader)
    #training loop
    for epoch in range(num_epochs):
        tr_loss = 0.0
        count = 0
        metric = 0.0
        for grouped in tqdm(bigger_dataloader):
            try:
                cbis_grouped = next(smaller_iter)
            except:
                smaller_iter = iter(smaller_dataloader)
                cbis_grouped = next(smaller_iter)
            # TODO: Print batch size of cbis dataloaded and regular dataloader, but only on firstepoch
            #push to cuda
            images = torch.cat((grouped[0],cbis_grouped[0]))
            images = images.float()
            labels = torch.cat((grouped[1],cbis_grouped[1]))
            if has_weights:
                weights = get_weight(torch.cat((grouped[2],cbis_grouped[2])),weight_ratio=weight_ratio)
                weights = weights.cuda()
            #print(images.shape)
            #images = transforms.Normalize(mean=(0.445,), std=(0.269,))(images) #normalize
            images = convert_to_3channel(images).cuda()
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


#todo: How to save output from model (torch tensor) as numpy? DONE: .numpy()
def evaluate_model_on_new_segmentations_and_save(model,segmentation_folder,saved_oracle_filepaths,correct_save_dir,save_dir,iter_num):
    #Define transform for input into model
    transforms_arr = [transforms.ToTensor(),transforms.Resize((256,256))]
    image_transform = transforms.Compose(transforms_arr)
    
    #Find all files in segmentation_folder
    segmentation_filepaths = []
    for root, dirs, files in os.walk(segmentation_folder):
        for file in files:
            if file.endswith(".npy"):
                segmentation_filepaths.append(os.path.join(root,file))
    patient_ids = []
    for i in saved_oracle_filepaths:
        patient_ids.append(i.split("/")[-1])
    for filepath in tqdm(segmentation_filepaths):
        #redo stuff in dataloader - load in stack and apply tensor transform for model input
        arr_and_mask = np.load(filepath)
        arr = arr_and_mask[0,:,:].copy()
        mask = arr_and_mask[1,:,:].copy()

        image = image_transform(arr)
        
        
        image = image.float()
        image = convert_to_3channel(image).cuda()
        image = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (image)


        unet_seg = model(image)
        unet_seg = F.softmax(unet_seg[0],dim=0)
        unet_seg = get_binary_mask(unet_seg[1,:,:]).detach().cpu().numpy()

        #grab filename and make sure save directories are defined
        #filename = "/".join(filepath.split("/")[-2:])
        class_subfolder = save_dir + filepath.split("/")[-2] + "/"
        if not os.path.exists(class_subfolder):
            os.makedirs(class_subfolder)
        save_path = class_subfolder + filepath.split("/")[-1]

        #do the same for the labelled correctly by oracle folder
        class_subfolder = correct_save_dir + filepath.split("/")[-2] + "/"
        if not os.path.exists(class_subfolder):
            os.makedirs(class_subfolder)
        correct_oracle_save_path = class_subfolder + filepath.split("/")[-1]

        #check if in saved_oracle_filepaths
        #If file labelled correct by oracle, save og segmentation and add new to separate dir
        if filepath.split("/")[-1] in patient_ids:
            np.save(save_path,np.stack([arr,mask]))
            np.save(correct_oracle_save_path,np.stack([arr,unet_seg]))
        #if normal, save it to save_dir
        else:
            # print("DEBUGGING")
            # print(unet_seg.shape)
            # print(arr.shape)
            np.save(save_path,np.stack([arr,unet_seg]))

#Removes all 0's from oracle_results (images that oracle said are incorrect)
def remove_bad_oracle_results(oracle_results):
    output = {}
    for patient in oracle_results.keys():
        if oracle_results[patient]==1:
            output[patient] = oracle_results[patient]
    return output


#Threshold value changed here
def get_binary_mask(mask):
    return torch.where(mask > 0.2, 1, 0)

        


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


def random_flip(input, axis, with_fa=False):
    ran = random.random()
    if ran > 0.5:
        if with_fa:
            axis += 1
        return np.flip(input, axis=axis)
    else:
        return input


def random_crop(input, with_fa=False):
    
    ran = random.random()
    if ran > 0.2:
        # find a random place to be the left upper corner of the crop
        if with_fa:
            rx = int(random.random() * input.shape[1] // 10)
            ry = int(random.random() * input.shape[2] // 10)
            return input[:, rx: rx + int(input.shape[1] * 9 // 10), ry: ry + int(input.shape[2] * 9 // 10)]
        else:
            rx = int(random.random() * input.shape[0] // 10)
            ry = int(random.random() * input.shape[1] // 10)
            return input[rx: rx + int(input.shape[0] * 9 // 10), ry: ry + int(input.shape[1] * 9 // 10)]
    else:
        return input


def random_rotate_90(input, with_fa=False):
    ran = random.random()
    if ran > 0.5:
        if with_fa:
            return np.rot90(input, axes=(1,2))
        return np.rot90(input)
    else:
        return input


def random_rotation(x, chance, with_fa=False):
    ran = random.random()
    if with_fa:
        img = Image.fromarray(x[0])
        mask = Image.fromarray(x[1])
        if ran > 1 - chance:
            # create black edges
            angle = np.random.randint(0, 90)
            img = img.rotate(angle=angle, expand=1)
            mask = mask.rotate(angle=angle, expand=1, fillcolor=1)
            return np.stack([np.asarray(img), np.asarray(mask)])
        else:
            return np.stack([np.asarray(img), np.asarray(mask)])
    img = Image.fromarray(x)
    if ran > 1 - chance:
        # create black edges
        angle = np.random.randint(0, 90)
        img = img.rotate(angle=angle, expand=1)
        return np.asarray(img)
    else:
        return np.asarray(img)

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
    # for i,(images,labels) in enumerate(CBISDDSMGetData(cbis_train_filenames,train_dir,image_transform,mask_transform,a_transform)):
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



