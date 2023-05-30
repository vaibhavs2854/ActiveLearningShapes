import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm

from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet_model import convert_2d_image_to_nifti, plan_and_preprocess
from floodfill import convert_directory_to_floodfill
from manual_oracle import  save_oracle_results


def get_binary_mask_threshold(mask,threshold):
    return np.where(mask > threshold, 1, 0)


#Removes all 0's from oracle_results (images that oracle said are incorrect)
def remove_bad_oracle_results(oracle_results):
    output = {}
    for patient in oracle_results.keys():
        if oracle_results[patient]==1:
            output[patient] = oracle_results[patient]
    return output


# Saves any correct segmentations found by the oracle, and additionally pickle dumps any structures
def save_active_learning_results(save_dir, oracle_results, oracle_results_thresholds, im_dir):
    correct_segs_save_dir = os.path.join(save_dir, "CorrectSegmentations")

    saved_oracle_filepaths = save_oracle_results(
        oracle_results, oracle_results_thresholds, im_dir, correct_segs_save_dir)
    fpath = os.path.join(save_dir, "saved_data_struct")
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    saved_oracle_filepaths_filepath = os.path.join(fpath, "Oracle_Filepaths.pickle")
    pickle.dump(saved_oracle_filepaths, open(saved_oracle_filepaths_filepath, "wb"))
    pickle.dump(oracle_results, open(os.path.join(fpath, "Oracle_Results.pickle"), "wb"))
    pickle.dump(oracle_results_thresholds, open(os.path.join(fpath,"Oracle_Results_Thresholds.pickle"), "wb"))

    return saved_oracle_filepaths


#Applies the threshold to each mask saved from the oracle. Resaves [arr,thresholded_mask] stack into save_dir using same conventions (.../Shape/name.npy)
def threshold_and_save_images(saved_oracle_filepaths, oracle_results_thresholds, save_dir):
    for filepath in tqdm(saved_oracle_filepaths):
        if ("/".join(filepath.split("/")[-2:]))[:-4] not in oracle_results_thresholds:
            threshold = 0.2
        else:
            threshold = oracle_results_thresholds[("/".join(filepath.split("/")[-2:]))[:-4]]
        arr_and_mask = np.load(filepath)
        copy_arr_mask = arr_and_mask.copy()

        arr = copy_arr_mask[0,:,:].copy()
        mask = copy_arr_mask[1,:,:].copy()
        #apply threshold to mask
        mask = get_binary_mask_threshold(mask, threshold)
        to_save = np.stack([arr, mask])
        
        save_save_dir = os.path.join(save_dir, "/".join(filepath.split("/")[-2:]))
        if not os.path.exists(os.path.join(save_dir, filepath.split("/")[-2])):
            os.makedirs(os.path.join(save_dir, filepath.split("/")[-2]))
        np.save(save_save_dir, to_save)


def update_dir_with_oracle_info(save_dir, oracle_results_thresholds, im_dir):
    save_dir = os.path.join(save_dir, "OracleThresholdedImages")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # find all filepaths in im_dir
    all_filepaths = []
    for root, dirs, files in os.walk(im_dir):
        for file in files:
            if file.endswith(".npy"):
                all_filepaths.append(os.path.join(root, file))

    threshold_and_save_images(
        all_filepaths, oracle_results_thresholds, save_dir)
    save_dir = convert_directory_to_floodfill(save_dir, iter0=False)
    return save_dir


def redirect_saved_oracle_filepaths_to_thresheld_directory(saved_oracle_filepaths, im_dir):
    new_filepaths = [(im_dir + "/".join(filepath.split("/")[-2:]))
                     for filepath in saved_oracle_filepaths]
    return new_filepaths


def save_files_for_nnunet(task_id, run_id, filepaths):
    new_gz_dir = os.path.join(os.environ['nnUNet_raw_data_base'],'nnUNet_raw_data', f'Task{task_id}_{run_id}')
    os.makedirs(new_gz_dir, exist_ok=True)

    target_imagesTr = os.path.join(new_gz_dir, 'imagesTr')
    target_labelsTr = os.path.join(new_gz_dir, 'labelsTr')
    os.makedirs(target_imagesTr, exist_ok=True)
    os.makedirs(target_labelsTr, exist_ok=True)

    for t in filepaths:
        unique_name = os.path.splitext(os.path.split(t)[-1])[0]  # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        input_file = t

        img = np.load(input_file)
        img = img.copy()

        img_r = cv2.resize(img[0], (640,640))
        mask = cv2.resize(img[1], (640,640))

        output_image_file = os.path.join(target_imagesTr, unique_name)  # do not specify a file ending! This will be done for you
        output_seg_file = os.path.join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you
        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please take a look at the code for this function and adapt it to your needs
        train_img = convert_2d_image_to_nifti(img_r.copy(), output_image_file, is_seg=False)

        # nnU-Net expects the labels to be consecutive integers. This can be achieved with setting a transform
        train_seg = convert_2d_image_to_nifti(mask.copy(), output_seg_file, is_seg=True,
                                    transform=lambda x: (x >= 1).astype(int))

        # finally we can call the utility for generating a dataset.json
    generate_dataset_json(os.path.join(new_gz_dir, 'dataset.json'), target_imagesTr, None, ("RGB",),
                    labels={0: 'background', 1: 'lesion'}, 
                    dataset_name=f'Task{task_id}_{run_id}', 
                    license='hands off!')
    
    # subprocess.run(["nnUNet_plan_and_preprocess", "-t", f"{task_id}", "--verify_dataset_integrity"])
    plan_and_preprocess([task_id, ], verify_integrity = True)
