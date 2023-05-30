import os

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
        arr_and_mask = np.load(filepath)
        copy_arr_mask = arr_and_mask.copy()
        arr = copy_arr_mask[0,:,:].copy()
        mask = copy_arr_mask[1,:,:].copy()
        
        #mask = largest_contiguous_region(mask)
        
        image = (self.image_transform(arr))
        mask_label = (self.image_transform(mask))
        
        patient_id = '/'.join(filepath.split("/")[-2:])[:-4]
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