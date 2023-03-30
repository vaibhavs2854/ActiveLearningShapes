# Python Library imports
import os
import sys
import random

import numpy as np
import shutil
import torch
import pandas as pd
import glob
import cv2

import SimpleITK as sitk
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

import pickle
import argparse
import subprocess

from nnunet.dataset_conversion.utils import generate_dataset_json

# Backend py file imports
from floodfill import convert_directory_to_floodfill
from dataloader import get_DataLoader
from model import model_update, get_patient_scores, initialize_and_train_model_experiment
from oracle import largest_contiguous_region, save_oracle_results
from unet import unet_dataloader, get_ints, unet_update_model
from unet import threshold_and_save_images, remove_bad_oracle_results, evaluate_model_on_new_segmentations_and_save
from unet import exposure, convert_to_3channel
from nnunet_model import predict_simplest_AL, batch_iou, convert_2d_image_to_nifti, nnunet_update_model, plan_and_preprocess
from manual_oracle import get_binary_mask_threshold_torch, query_oracle_automatic


# Standard deviation of 50
def gaus2d(x=0, y=0, mx=0, my=0, sx=50, sy=50):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


def generate_gaussian_weights(size=256):
    # define normalized 2D gaussian
    x = np.linspace(-1*(size/2-1), size/2, num=size)
    y = np.linspace(-1*(size/2-1), size/2, num=size)
    x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D
    z = gaus2d(x, y)
    z = z/np.amax(z)
    return z


def intersection_over_union_exp(output_mask, ground_mask):
    ground_mask = get_ints(ground_mask).squeeze(1)
    summed = ground_mask + output_mask
    twos = summed - 2
    num = 256*256 - torch.count_nonzero(twos)
    denom = torch.count_nonzero(summed)
    outputs = torch.div(num, denom)
    return torch.mean(outputs)


def evaluate_metric_on_validation(model, validation_dir, viz_save=False):
    model.eval()
    transforms_arr = [transforms.ToTensor(), transforms.Resize((256, 256))]
    image_transform = transforms.Compose(transforms_arr)
    ious = []
    segmentation_filepaths = []
    gaussian_weight = generate_gaussian_weights()

    for root, dirs, files in os.walk(validation_dir):
        for file in files:
            if file.endswith(".npy"):
                segmentation_filepaths.append(os.path.join(root, file))

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

        unet_seg = model(image)
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
        test_images_save_path = ""
        id = filepath.split("/")[-1]
        if (max_iou < 0.1):
            # save bad iou
            # save unet_seg_threshold
            test_images_save_path = f"/usr/xtmp/jly16/mammoproj/nnunet_integration_tmp/iou_test_images_1/bad/{id}"
        if (max_iou > 0.9):
            # save good iou
            test_images_save_path = f"/usr/xtmp/jly16/mammoproj/nnunet_integration_tmp/iou_test_images_1/good/{id}"
        if viz_save:
            np.save(test_images_save_path, np.stack(
                [mask.numpy(), thresholded_mask]))
    # Grab histogram of ious
    # Floodfill after thresholding and before iou calculation
    ious_np = np.asarray(ious)
    # np.save(
    #     "/usr/xtmp/vs196/mammoproj/Code/ActiveLearning/iouhist_randomrun2.npy", ious_np)
    return np.average(np.asarray(ious))



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

def active_learning_experiment(active_learning_train_cycles, imgs_seen, segmtr_model, run_id, output_dir, iter_num, oracle_query_method, unet = False):
    # ACTIVE LEARNING STAGE

    # INITIALIZE CLASSIFIER
    # File definitions and static setup
    segmentation_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset/train/" # files should be .npy, (2, , ) channels: image, binarized segmentation
    classifier_training_dir = segmentation_dir
    oracle_results = dict()
    oracle_results_thresholds = dict()
    total_images_shown = 0
    saved_oracle_filepaths = []
    
    print("===TRAINING DISCRIMINATOR===")
    dataloader = get_DataLoader(classifier_training_dir, 32, 2)
    discriminator, loss_tracker, criterion, optimizer = initialize_and_train_model_experiment(
        dataloader, epochs=10)
    patient_scores = get_patient_scores(discriminator, dataloader)
    all_patient_scores = []
    all_patient_scores.append(patient_scores)

    # QUERYING THE ORACLE
    print("===QUERYING THE ORACLE===")
    ground_truth_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset/train/"
    assert imgs_seen % 10 == 0
    active_learning_train_cycles = imgs_seen // 10
    # Begin loop over number of active learning/
    for _ in tqdm(range(active_learning_train_cycles)):
        # Querying oracle - currently queries {query_cycles} times.
        try:
            oracle_results, oracle_results_thresholds = query_oracle_automatic(
                oracle_results, oracle_results_thresholds, patient_scores,
                ground_truth_dir, segmentation_dir,
                query_method=oracle_query_method, query_number=10)
        except Exception as e:
            print(str(e))
            print("Something went wrong with the automatic oracle query")
            sys.exit(1)
        # if oracle_results is None:
        #     continue
        total_images_shown += 10

        # Updating classifier 1 epoch at a time for 5 epochs.
        print(f"=UPDATING CLASSIFIER (total images: {total_images_shown})")
        for i in range(1):
            discriminator = model_update(
                discriminator, dataloader, oracle_results, criterion, optimizer, num_epochs=1)

            patient_scores = get_patient_scores(discriminator, dataloader)
            all_patient_scores.append(patient_scores)

    # IN-BETWEEN STAGE
    print("===SAVING CLASSIFIED ORACLE RESULTS===")
    run_dir = os.path.join(output_dir, "Run_" + run_id, "Iter" + str(iter_num))
    # Space for saving oracle results and pickling data structures
    saved_oracle_filepaths = save_active_learning_results(
        run_dir, oracle_results, oracle_results_thresholds, classifier_training_dir)
    # not necessary as oracle_results is never even used again in this method.
    oracle_results = remove_bad_oracle_results(oracle_results)

    # if no images are classified as correct by oracle, print and return
    if len(saved_oracle_filepaths) == 0:
        print("No oracle results classified as correct.")
        return 0
    else:
        print(
            f"Oracle classifies {len(saved_oracle_filepaths)} images as correct.")

    # SEGMENTATION STAGE
    # Preprocess data with information learned from active learning.
    print("===PREPARING FOR SEGMENTATION STAGE===")
    unet_train_dir = update_dir_with_oracle_info(run_dir, oracle_results_thresholds, segmentation_dir)
    new_saved_oracle_filepaths = redirect_saved_oracle_filepaths_to_thresheld_directory(
        saved_oracle_filepaths, unet_train_dir)
    if not unet:
        last_task = sorted(glob.glob(os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data','Task*')))[-1]
        last_task = last_task.split('nnUNet_raw_data/Task')[-1][:3]
        new_task = int(last_task) + 1
        save_files_for_nnunet(new_task, run_id, new_saved_oracle_filepaths)

    print("===TRAINING SEGMENTER FOR 5 EPOCHS===")
    # Train model using learned oracle data for 5 epochs
    # learned oracle data = images that are in the "new saved oracle filepaths" (the images that the oracle said looked good)   
    if unet:
        segmtr_dataloader = unet_dataloader(
            new_saved_oracle_filepaths, batch_size=8, num_workers=2)
        segmtr_model, _, _ = unet_update_model(
            segmtr_model, segmtr_dataloader, num_epochs=5)
    else:
        trainer = nnunet_update_model(
            os.path.join(os.environ['nnUNet_preprocessed'], f'Task{new_task}_{run_id}'))
    
    # potentially save model this iteration if we want. # to be used later 
    if unet:
        model_save_path = os.path.join(run_dir, "unetmodel.pth")
        torch.save(segmtr_model, model_save_path)
    else:
        model_save_dir = os.path.join(run_dir, 'all')
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, "Iter" + str(iter_num)+".model")
        trainer.save_checkpoint(model_save_path)

        plans_pkl = '/usr/xtmp/jly16/mammoproj/data/nnUNet_trained_models/nnUNet/2d/Task501_cbis-ddsm/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl'
        if not os.path.isfile(os.path.join(run_dir,'plans.pkl')): 
            shutil.copy2(plans_pkl, run_dir)

    # evaluation 1: generate new segmentations of training images and save them. (This is for the next stage of active learning)
    # Evaluate ON IMAGES IN SEGMENTATION_FOLDER AND GENERATE SEGMENTATIONS OF THEM
    print("=== CREATING SEGMENTATIONS FOR TRAIN SET ===")
    # dir for marked correct by the oracle, do not overwrite the old segmentation, so save them here as an archive
    correct_save_dir = os.path.join(run_dir, "Segmentations_C" )

    # completely new set of segmentations created by the updated unet
    save_dir = os.path.join(run_dir,"Segmentations")
    
    if unet: 
        segmentation_folder = segmentation_dir
        evaluate_model_on_new_segmentations_and_save(
            segmtr_model, segmentation_folder, saved_oracle_filepaths, correct_save_dir, save_dir, iter_num)
    else:
        segmentation_folder = '/usr/xtmp/jly16/mammoproj/data/nnUNet_raw_data_base/nnUNet_raw_data/Task504_duke-mammo/imagesTr'
        # TODO: ADD SAVING CORRECT SAVE_DIR? - done? in "save_files_for_nnunet"... they dont get written over... low pri
        predict_simplest_AL(segmentation_folder, save_dir)   
    # Push save_dir as the oracle image dir for the next iteration. That's where we populate with unbinarized segmentations from recently trained UNet

        
    # evaluation 2: generate segmentations of validation and see how accurate our new segmenter is
    print("=== CREATING SEGMENTATIONS FOR TEST SET ===")
    viz = False
    if unet:
        manual_fa_valid_dir = f"/usr/xtmp/vs196/mammoproj/Data/manualfa/manual_validation/"
        validation_metric = evaluate_metric_on_validation(
            segmtr_model, manual_fa_valid_dir, viz_save=viz)
    else:
        valid_dir = os.path.join(
            os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', f"Task504_duke-mammo")
        val_pred_dir = os.path.join(run_dir, "ValSegmentations")
        # TODO: separate good iou from bad iou...
        # TODO: thresholding? 
        predict_simplest_AL(os.path.join(valid_dir, 'imagesTs'), val_pred_dir)
        validation_metric = batch_iou(os.path.join(valid_dir, 'labelsTs'), val_pred_dir, viz_save=viz)
    print(f"Metric of new segmenter after active learning is: {validation_metric}.")
   
    return validation_metric, model_save_path


# Outline of experiment:
# loop over:
#   Active Learning Stage
#   Segmentation Stage
#   Evaluation Stage + Metrics


def run_active_learning_experiment(run_id, output_dir, random_seed, unet = False):
    print("Starting run")
    # base_ops = ['nnunet', 'unet']
    # assert base in base_ops, f"Parameter `base` should be in {base_ops}"

    # pandas dataframe where columns are query_type query_number IOU location of saved model
    experiment_output = pd.DataFrame(
        columns=['random_seed', 'query_type', 'imgs_seen', 'IOU', 'saved_model_location'])

    imgs_seen_list = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 170, 200]
    oracle_query_methods = ["uniform",]
#     oracle_query_methods = ["uniform", "random",
#                             "percentile=0.8", "best", "worst"]
    for oracle_query_method in oracle_query_methods:
        for imgs_seen in imgs_seen_list:
            run_unique_id = f"{run_id}_{oracle_query_method}_{imgs_seen}_{random_seed}"
            print(run_unique_id)
            if unet:
                model_save_path = "/usr/xtmp/vs196/mammoproj/Code/SavedModels/ControlALUNet/0726/unetmodel_size150.pth"
                # model_save_path = grab a fresh unet.
                segmtr_model = torch.load(model_save_path)
            else:
                segmtr_model = None
            validation_metric, saved_model_location = active_learning_experiment(10,
                                                                    imgs_seen,
                                                                    segmtr_model,
                                                                    run_unique_id,
                                                                    output_dir,
                                                                    iter_num=0,
                                                                    oracle_query_method=oracle_query_method,
                                                                    unet = unet)

            print(
                f"Done with {imgs_seen} for query method {oracle_query_method}")
            experiment_output = experiment_output.append({'random_seed': random_seed,
                                                          'query_type': oracle_query_method,
                                                          'imgs_seen': imgs_seen,
                                                          'IOU': validation_metric,
                                                          'saved_model_location': saved_model_location},
                                                          ignore_index=True)

    print("Finished run")
    return experiment_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', "--run_id", required = True)
    parser.add_argument('-o', "--output_dir", required = True)
    parser.add_argument("--random_seed", nargs=1, type=int, required = True)
    parser.add_argument('--unet', action='store_true')
    parser.add_argument('--nnunet', dest='unet', action='store_false')
    parser.set_defaults(unet=False)
    args = parser.parse_args()

    random_seed_number = args.random_seed[0]
    torch.manual_seed(random_seed_number)
    torch.cuda.manual_seed(random_seed_number)
    np.random.seed(random_seed_number)
    random.seed(random_seed_number)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    run_id = args.run_id
    output_dir = args.output_dir
    unet = args.unet

    experiment_output = run_active_learning_experiment(run_id, output_dir, random_seed_number, unet = unet)
    save_dir = os.path.join(output_dir, f"Run_{run_id}/")
    os.makedirs(save_dir, exist_ok=True)
    experiment_output.to_csv(os.path.join(save_dir, 'experiment_output.csv'), sep=',')

    # save the experiment output pandas dataframe

    # for i in range(len(metrics)):
    #     print(f"{query_numbers[i]} {metrics[i]}")
    # print("done")
