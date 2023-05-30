# Python Library imports
import os
import sys
import random

import numpy as np
import torch
import pandas as pd
import glob

from tqdm import tqdm
import argparse

# Backend py file imports
from file_organization import save_active_learning_results, update_dir_with_oracle_info
from file_organization import redirect_saved_oracle_filepaths_to_thresheld_directory, remove_bad_oracle_results
from file_organization import save_files_for_nnunet
from dataloader import get_DataLoader
from disc_model import disc_model
from auto_oracle import query_oracle_automatic
import seg_model
import unet_model
import nnunet_model


def active_learning_experiment(active_learning_train_cycles, imgs_seen, run_id, output_dir, iter_num, oracle_query_method, unet = False):
    # ACTIVE LEARNING STAGE

    # INITIALIZE CLASSIFIER
    # File definitions and static setup
    # files should be .npy, (2, , ) channels: image, binarized segmentation
    discriminator_training_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset/train/" 
    oracle_query_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset/train/" 
    ground_truth_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset/train/" 
    
    total_images_shown = 0
    saved_oracle_filepaths = []
    
    print("===TRAINING DISCRIMINATOR===")
    batch_size = 32
    dataloader = get_DataLoader(discriminator_training_dir, batch_size, 2)

    discriminator = disc_model()
    discriminator.load_model(dataloader)
    discriminator.initialize_model(batch_size = batch_size, epochs=10) # initial training
    
    print("===GENERATING INITIAL PATIENT SCORES===")
    segmentation_dataloader = get_DataLoader(oracle_query_dir, batch_size, 2)
    patient_scores = discriminator.get_scores(segmentation_dataloader)
    
    all_patient_scores = []
    all_patient_scores.append(patient_scores)

    # QUERYING THE ORACLE
    print("===QUERYING THE ORACLE===")
    assert imgs_seen % 10 == 0
    active_learning_train_cycles = imgs_seen // 10
    # Begin loop over number of active learning/
    oracle_results = dict()
    oracle_results_thresholds = dict()
    for _ in tqdm(range(active_learning_train_cycles)):
        # Querying oracle - currently queries {query_cycles} times.
        try:
            oracle_results, oracle_results_thresholds = query_oracle_automatic(
                oracle_results, oracle_results_thresholds, patient_scores,
                ground_truth_dir, oracle_query_dir,
                query_method=oracle_query_method, query_number=10)
        except Exception as e:
            print(str(e))
            print("Something went wrong with the automatic oracle query")
            sys.exit(1)
        total_images_shown += 10

        # Updating classifier 1 epoch at a time for 5 epochs.
        print(f"=UPDATING DISCRIMINATOR (total images: {total_images_shown})")
        for i in range(5):
            discriminator.update_model(oracle_results,batch_size = batch_size, num_epochs=1)

            patient_scores = discriminator.get_scores(segmentation_dataloader)
            all_patient_scores.append(patient_scores)

    # IN-BETWEEN STAGE
    print("===SAVING CLASSIFIED ORACLE RESULTS===")
    run_dir = os.path.join(output_dir, "Run_" + run_id, "Iter" + str(iter_num))
    # Space for saving oracle results and pickling data structures
    saved_oracle_filepaths = save_active_learning_results(
        run_dir, oracle_results, oracle_results_thresholds, oracle_query_dir)

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
    segmenter_train_dir = update_dir_with_oracle_info(run_dir, oracle_results_thresholds, oracle_query_dir)
    new_saved_oracle_filepaths = redirect_saved_oracle_filepaths_to_thresheld_directory(
        saved_oracle_filepaths, segmenter_train_dir)
    if not unet:
        last_task = sorted(glob.glob(os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data','Task*')))[-1]
        last_task = last_task.split('nnUNet_raw_data/Task')[-1][:3]
        task_id = int(last_task) + 1
        save_files_for_nnunet(task_id, run_id, new_saved_oracle_filepaths)

    print("===TRAINING SEGMENTER FOR 5 EPOCHS===")
    # Train model using learned oracle data for 5 epochs
    # learned oracle data = images that are in the "new saved oracle filepaths" (the images that the oracle said looked good)   
    if unet:
        segmenter = unet_model.unet_model()
        segmenter_train_dir  = new_saved_oracle_filepaths
    else:
        segmenter = nnunet_model.nnunet_model()
        segmenter_train_dir = os.path.join(os.environ['nnUNet_preprocessed'], f'Task{task_id}_{run_id}')
    segmenter.load_model(segmenter_train_dir)
    segmenter.update_model(num_epochs = 5)
    
    # potentially save model this iteration if we want. # to be used later 
    if unet:
        model_save_path = os.path.join(run_dir, "unetmodel.pth")
    else:
        model_save_path = os.path.join(run_dir, 'all', "Iter" + str(iter_num)+".model")
    segmenter.save_model(model_save_path)

    # evaluation 1: generate new segmentations of training images and save them. (This is for the next stage of active learning)
    # Evaluate ON IMAGES IN SEGMENTATION_FOLDER AND GENERATE SEGMENTATIONS OF THEM
    print("=== CREATING SEGMENTATIONS FOR TRAIN SET ===")
    # dir for marked correct by the oracle, do not overwrite the old segmentation, so save them here as an archive
    correct_save_dir = os.path.join(run_dir, "Segmentations_C" )

    # completely new set of segmentations created by the updated segmenter
    save_dir = os.path.join(run_dir,"Segmentations")
    
    if unet: 
        segmentation_folder = discriminator_training_dir
    else:
        segmentation_folder = '/usr/xtmp/jly16/mammoproj/data/nnUNet_raw_data_base/nnUNet_raw_data/Task504_duke-mammo/imagesTr'
    segmenter.predict(segmentation_folder, save_dir, correct_save_dir = correct_save_dir, saved_oracle_filepaths = saved_oracle_filepaths)   
    # Push save_dir as the oracle query dir for the next iteration. That's where we populate with unbinarized segmentations from recently trained segmenter

        
    # evaluation 2: generate segmentations of validation and see how accurate our new segmenter is
    print("=== CREATING SEGMENTATIONS FOR TEST SET ===")
    if unet:
        valid_input_dir =  f"/usr/xtmp/vs196/mammoproj/Data/manualfa/manual_validation/"
    else:
        valid_input_dir = os.path.join(
            os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', f"Task504_duke-mammo", 'imagesTs')
    
    valid_output_dir = os.path.join(run_dir, "ValSegmentations")
    validation_metric = segmenter.validate(valid_input_dir, valid_output_dir)
    print(f"Metric of new segmenter after active learning is: {validation_metric}.")
   
    return validation_metric, model_save_path


def run_active_learning_experiment(run_id, output_dir, random_seed, unet = False):
    print("Starting run")

    # pandas dataframe where columns are query_type query_number IOU location of saved model
    experiment_output = pd.DataFrame(
        columns=['random_seed', 'query_type', 'imgs_seen', 'IOU', 'saved_model_location'])

    imgs_seen_list = [20, 80]
    # imgs_seen_list = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 170, 200]
    # oracle_query_methods = ["uniform",]
    oracle_query_methods = ["uniform", "random",
                            "percentile=0.8", "best", "worst"]
    for oracle_query_method in oracle_query_methods:
        for imgs_seen in imgs_seen_list:
            run_unique_id = f"{run_id}_{oracle_query_method}_{imgs_seen}_{random_seed}"
            print(run_unique_id)
            validation_metric, saved_model_location = active_learning_experiment(10,
                                                                    imgs_seen,
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
