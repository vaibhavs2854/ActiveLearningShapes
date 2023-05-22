# Python Library imports
import os
import re

import numpy as np
import shutil
import glob
import tempfile

from viz import display_image_annotation


class seg_model: 
    def __init__(self):
        self.verts = {}
        self.model_save_path = None
        return
    
    def load_model(self, train_dir): 
        #self.verts['base model'] = model.epochs
        return
    
    def update_model(self, num_epochs = 5):
        #self.verts['AL N images'] = model.epochs
        return 
    
    def predict(self, input_folder, output_folder = None, correct_save_dir = None, saved_oracle_filepaths = {}):
        return
    
    def validate(self, input_folder, output_folder = None):
        return 

    def save_model(self, run_dir): 
        return

    def show_segmentations(self, filepaths): 
        # create temp folders (predict simpel AL can only take folders of inputs and saves to output folder)
        with tempfile.TemporaryDirectory() as temp_input_folder: 
            with tempfile.TemporaryDirectory() as temp_output_folder: 
                # move all files we want to predict on to the temp folder created to store inputs
                for file in filepaths: 
                    shutil.copy(file, os.path.join(temp_input_folder, os.path.split(file)[1]))

                self.validate(temp_input_folder, output_folder = temp_output_folder)
                
                # get list of all output files
                out_filepaths = []
                for root, dirs, files in os.walk(temp_output_folder, topdown=False):
                    for name in files:
                        out_filepaths.append(os.path.join(root, name))
                
                # show all from output folder 
                file_pairs = []
                annotations = []
                for in_file in filepaths:
                    f = os.path.split(in_file)[1]
                    f = '_'.join(re.split('\_|\.', f)[:4]) +'.'+'.'.join(f.split('.')[1:])
                    out_file = [full_path for full_path in out_filepaths if f in full_path][0]
                    file_pairs.append([in_file, out_file])
                    annotations.append(f)
                display_image_annotation(file_pairs,annotations)