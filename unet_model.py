import os

import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from skimage import exposure
from augmentations import random_flip, random_rotate_90
from auto_oracle import get_binary_mask_threshold_torch
from floodfill import largest_contiguous_region

import seg_model

def gaus2d(x=0, y=0, mx=0, my=0, sx=50, sy=50):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

def generate_gaussian_weights(size=256):
    # define normalized 2D gaussian
    x = np.linspace(-1*(size/2-1), size/2,num=size)
    y = np.linspace(-1*(size/2-1), size/2,num=size)
    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
    z = gaus2d(x, y)
    z = z/np.amax(z)
    return z

class InhouseGetData(Dataset):
    def __init__(self,image_filepaths,image_transform,data_aug=True,has_weights=True):
        super().__init__()
        self.image_filepaths = image_filepaths
        self.image_transform = image_transform
        self.data_aug = data_aug
        self.has_weights = has_weights
        self.gaussian_weight = generate_gaussian_weights()

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
        
        image = self.image_transform(arr)
        image = torch.from_numpy(np.multiply(self.gaussian_weight,image.numpy())) #Gaussian weight the image before training.

        mask_label = self.image_transform(mask)
        mask_label = torch.where(mask_label>0.5,1,0) #Make sure mask is binarized. Check with Alina if 0.5 is right.

        if self.has_weights:
            weights_label = self.image_transform(weights)
        
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
        
        image = self.image_transform(arr)
        
        mask_label = self.image_transform(mask)
        if self.has_weights:
            weights_label = self.image_transform(weights)
        if self.has_weights:
            return image,mask_label,weights_label
        return image,mask_label

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


def intersection_over_union_exp(output_mask, ground_mask):
    ground_mask = get_ints(ground_mask).squeeze(1)
    summed = ground_mask + output_mask
    twos = summed - 2
    num = 256*256 - torch.count_nonzero(twos)
    denom = torch.count_nonzero(summed)
    outputs = torch.div(num, denom)
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


def convert_to_3channel(x):
    return torch.tile(x,(1,3,1,1))


#Threshold value changed here
def get_binary_mask(mask):
    return torch.where(mask > 0.2, 1, 0)

class unet_model(seg_model.seg_model): 
    def __init__(self):
        super().__init__()
        self.dataloader = None
        self.model_source = "/usr/xtmp/vs196/mammoproj/Code/SavedModels/ControlALUNet/0726/unetmodel_size150.pth"
        self.module = None
    
    def load_model(self, train_dir ): 
        self.dataloader = unet_dataloader(train_dir, batch_size=8, num_workers=2)
        self.module = torch.load(self.model_source)
#         self.verts['Pretrained Model Loaded'] = 

    def update_model(self,num_epochs = 5,has_weights=True,weight_ratio=0.5):    
        criterion = weightedpixelcros
        optimizer = torch.optim.SGD(self.module.parameters(),lr=0.001,momentum=0.9)
        loss_tracker = [] #plot loss
        metric_tracker = []

        self.module.train()
        #training loop
        for epoch in range(num_epochs):
            tr_loss = 0.0
            count = 0
            metric = 0.0
            for grouped in tqdm(self.dataloader):
                #push to cuda
                images = grouped[0]
                images = images.float()
                labels = grouped[1]
                if has_weights:
                    weights = get_weight(grouped[2],weight_ratio=weight_ratio)
                    weights = weights.cuda()
                images = convert_to_3channel(images).cuda()
                #Include bottom line (normalize after convert to 3-channels)
                images = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (images)
                labels = labels.cuda()
                optimizer.zero_grad()
                y = self.module(images)
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
        
#         self.verts[f'Model Updated (epoch = {}'] = 
        return loss_tracker,metric_tracker
        
    def save_model(self, save_path):
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        torch.save(self.module, save_path)

    def predict(self, input_folder, output_folder = None, correct_save_dir = None, saved_oracle_filepaths = {}):
        #Define transform for input into model
        transforms_arr = [transforms.ToTensor(),transforms.Resize((256,256))]
        image_transform = transforms.Compose(transforms_arr)
        
        #Find all files in input_folder
        segmentation_filepaths = []
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.endswith(".npy"):
                    segmentation_filepaths.append(os.path.join(root,file))
        patient_ids = []
        for i in saved_oracle_filepaths:
            patient_ids.append(i.split("/")[-1])
        for filepath in tqdm(segmentation_filepaths):
            #redo stuff in dataloader - load in stack and apply tensor transform for model input
            arr_and_unbin_output = np.load(filepath)
            arr = arr_and_unbin_output[0,:,:].copy()
            unbin_output = arr_and_unbin_output[1,:,:].copy()

            unbin_output = image_transform(unbin_output)[0,:,:]
            arr = exposure.equalize_hist(arr) #histogram equalization, as we histogram equalize in dataloader. 
            image = image_transform(arr)
            
            image = image.float()
            detach_image = image.detach().cpu().numpy()[0,:,:]
            image = convert_to_3channel(image).cuda()
            image = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (image)


            unet_seg = self.module(image)
            unbinarized_unet_seg = F.softmax(unet_seg[0],dim=0)[1,:,:]
            #unet_seg here is unbinarized.
            unet_seg = get_binary_mask(unbinarized_unet_seg)
            unbinarized_unet_seg = unbinarized_unet_seg.detach().cpu().numpy()
            #grab filename and make sure save directories are defined
            class_subfolder = os.path.join(output_folder, filepath.split("/")[-2] + "/")
            if not os.path.exists(class_subfolder):
                os.makedirs(class_subfolder)
            save_path = os.path.join(class_subfolder, filepath.split("/")[-1])

            #do the same for the labelled correctly by oracle folder
            class_subfolder = os.path.join(correct_save_dir, filepath.split("/")[-2] + "/")
            if not os.path.exists(class_subfolder):
                os.makedirs(class_subfolder)
            correct_oracle_save_path = os.path.join(class_subfolder,filepath.split("/")[-1])

            #check if in saved_oracle_filepaths
            #If file labelled correct by oracle, save og segmentation and add new to separate dir
            if filepath.split("/")[-1] in patient_ids:
                np.save(save_path,np.stack([detach_image,unbin_output]))
                np.save(correct_oracle_save_path, np.stack([detach_image,unbinarized_unet_seg]))
            #if normal, save it to save_dir
            else:
                np.save(save_path,np.stack([detach_image,unbinarized_unet_seg]))


    def validate(self, input_folder, output_folder = None):
        self.module.eval()
        transforms_arr = [transforms.ToTensor(), transforms.Resize((256, 256))]
        image_transform = transforms.Compose(transforms_arr)
        ious = []
        segmentation_filepaths = []
        gaussian_weight = generate_gaussian_weights()

        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.endswith(".npy"):
                    segmentation_filepaths.append(os.path.join(root, file))
                    
        if output_folder: 
            os.makedirs(os.path.join(output_folder, 'bad'), exist_ok=True)
            os.makedirs(os.path.join(output_folder, 'good'), exist_ok=True)
            os.makedirs(os.path.join(output_folder, 'mid'), exist_ok=True)
            
        for filepath in tqdm(segmentation_filepaths):
            arr_and_bin_output = np.load(filepath)
            arr = arr_and_bin_output[0, :, :].copy()
            bin_output = arr_and_bin_output[1, :, :].copy()

            mask = image_transform(bin_output)[0, :, :]
            # Makes sure mask is binarized for iou calculation. Check if 0.5 is correct with Alina.
            mask = torch.where(mask > 0.5, 1, 0)
            arr = exposure.equalize_hist(arr)  # add hist equalization to
            image = image_transform(arr)
            image = torch.from_numpy(np.multiply(
                gaussian_weight, image.numpy()))  # Apply Gaussian filter

            image = image.float()
            image = convert_to_3channel(image).cuda()
            image = transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)

            unet_seg = self.module(image)
            unbinarized_unet_seg = F.softmax(unet_seg[0], dim=0)[1, :, :]
            max_iou = -1
            thresholded_mask = None
            thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1,
                        0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # try to find the best iou
            for threshold in thresholds:
                unet_seg = get_binary_mask_threshold_torch(
                    unbinarized_unet_seg, threshold).detach().cpu().numpy()
                unet_seg_ff = largest_contiguous_region(
                    unet_seg)  # flood fill binary segmentation
                unet_seg_ff_torch = torch.from_numpy(unet_seg_ff)
                # unet_seg = get_binary_mask(unbinarized_unet_seg).cpu()
                iou = intersection_over_union_exp(unet_seg_ff_torch, mask)
                if (iou > max_iou):
                    max_iou = iou
                    thresholded_mask = unet_seg_ff
            ious.append(max_iou)
            
            if output_folder:
                test_images_save_path = ""
                file_id = filepath.split("/")[-1]
                if (max_iou < 0.1):
                    # save bad iou
                    test_images_save_path = os.path.join(output_folder, 'bad', file_id)
                elif (max_iou > 0.9):
                    # save good iou
                    test_images_save_path = os.path.join(output_folder, 'good', file_id)
                else: 
                    test_images_save_path = os.path.join(output_folder, 'mid', file_id)
                
                np.save(test_images_save_path, np.stack(
                    [mask.numpy(), thresholded_mask]))
        # Grab histogram of ious
        # Floodfill after thresholding and before iou calculation
        ious_np = np.asarray(ious)
        # np.save(
        #     "/usr/xtmp/vs196/mammoproj/Code/ActiveLearning/iouhist_randomrun2.npy", ious_np)
        return np.average(np.asarray(ious))

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

