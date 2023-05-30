import os

import numpy as np
import cv2
from scipy import signal
from tqdm import tqdm

"""Largest Contiguous Region Helper Methods"""
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

def bfs_flood_fill(mask,i,j,val):
    queue = []
    queue.append((i,j))
    while len(queue)>0:
        i,j = queue.pop(0)
        if i<0 or j<0 or i>=mask.shape[0] or j>=mask.shape[1]:
            continue
        if not mask[i,j]==1:
            continue
        mask[i,j] = val
        queue.append((i+1,j))
        queue.append((i-1,j))
        queue.append((i,j+1))
        queue.append((i,j-1))
    return mask

def get_all_regions(mask):
    val = 2
    count_dict = {}
    modifying_mask = mask.copy()
    while 1 in modifying_mask:
        tuples = np.nonzero(modifying_mask==1)
        i,j = tuples[0][0],tuples[1][0]
        modifying_mask = bfs_flood_fill(modifying_mask,i,j,val)
        count_dict[val] = len(np.where(modifying_mask==val)[0])
        val += 1
    return count_dict, modifying_mask

def largest_contiguous_region(mask):
    if 1 not in mask:
        return mask
    count_dict,modifying_mask = get_all_regions(mask)
    max_val = 2
    max_ = count_dict[max_val]
    for key in count_dict.keys():
        if count_dict[key] > max_:
            max_val = key
            max_ = count_dict[key]
    return np.where(modifying_mask==max_val,1,0)

def get_int(mask):
    return np.where(mask>0.2,1,0)

"""END"""

#Evaluate retrained UNet with OracleImages/Iterx data, floodfill the resultant masks, and save [image,mask] stack.
#Iter 0 flag deals with the first case vs subsequent cases
def convert_directory_to_floodfill(in_dir,iter0=False):
    #assumes in_dir ends with /. Generates the out directory (with ff) and if doesn't exist makes it
    """This takes in the output from the UNet Model (WITHOUT floodfill and WITHOUT borders)
        Uses floodfill to generate largest contiguous region and Generates borders"""
    out_dir = in_dir[:-1] + "_ff/" if not iter0 else "/usr/xtmp/vs196/mammoproj/Code/ActiveLearning/UNetSegmentations/Iter0_ff/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_images_filepaths = []
    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith(".npy"):
                train_images_filepaths.append(os.path.join(root,file))
    for file in tqdm(train_images_filepaths):
        array_and_mask = np.load(file)
        arr = array_and_mask[0,:,:].copy()
        mask = largest_contiguous_region(array_and_mask[1,:,:].copy()) #does the floodfill on mask
        #Makes expanded border as 3rd channel
        resized_mask =  cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

        border_mask = find_border(resized_mask.copy())
        times_expanded = 0
        for j in range(5):
            border_mask = expand_border(border_mask)
        #Saves
        #border_mask = cv2.resize(border_mask,dsize=mask.shape,interpolation=cv2.INTER_NEAREST)
        arr = cv2.resize(arr,dsize=(256,256),interpolation=cv2.INTER_CUBIC)
        ff_array_and_mask_and_border = np.stack([arr,resized_mask,border_mask])
        save_dir_dir = out_dir + file.split("/")[-2] + "/"
        if not os.path.exists(save_dir_dir):
            os.makedirs(save_dir_dir)
        save_path = out_dir + "/".join(file.split("/")[-2:])
        np.save(save_path,ff_array_and_mask_and_border)
    return out_dir
