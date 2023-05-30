import numpy as np 
import os
import contextlib
import glob
import SimpleITK as sitk
import shutil
from time import time
import tempfile

from nnunet.run.default_configuration import get_default_configuration
from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import save_json, join, isdir, load_json, maybe_mkdir_p


import nnunet
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.utils import crop
from nnunet.paths import nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity
from nnunet.training.model_restore import recursive_find_python_class

import seg_model


def convert_2d_image_to_nifti(img: np.array, output_filename_truncated: str, spacing=(999, 1, 1),
                              transform=None, is_seg: bool = False) -> None:
    """
    MODIFIED FROM nnunet.conversions.convert_2d_image_to_nifti ... input here is a 2d np.array, theirs is a filename
    Take a 2d image in np.array format and converts it into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!
    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net
    If Transform is not None it will be applied to the image after loading.
    Segmentations will be converted to np.uint32!
    :param is_seg:
    :param transform:
    :param input_filename:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """

    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

    for j, i in enumerate(img):

        if is_seg:
            i = i.astype(np.uint32)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        if not is_seg:
            sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")
        return itk_img
    

def plan_and_preprocess(task_ids:list,
    planner_name3d ="ExperimentPlanner3D_v21",
    planner_name2d ="ExperimentPlanner2D_v21", 
    tl = 8, tf = 8, 
    verify_integrity = True, 
    overwrite_plans = None, 
    overwrite_plans_identifier = None):

    if planner_name3d == "None":
        planner_name3d = None
    if planner_name2d == "None":
        planner_name2d = None

    if overwrite_plans is not None:
        if planner_name2d is not None:
            print("Overwriting plans only works for the 3d planner. I am setting '--planner2d' to None. This will "
                  "skip 2d planning and preprocessing.")
        assert planner_name3d == 'ExperimentPlanner3D_v21_Pretrained', "When using --overwrite_plans you need to use " \
                                                                       "'-pl3d ExperimentPlanner3D_v21_Pretrained'"

    # we need raw data
    tasks = []
    for i in task_ids:
        i = int(i)

        task_name = convert_id_to_task_name(i)

        if verify_integrity:
            verify_dataset_integrity(join(nnUNet_raw_data, task_name))

        crop(task_name, False, tf)

        tasks.append(task_name)

    search_in = join(nnunet.__path__[0], "experiment_planning")

    if planner_name3d is not None:
        planner_3d = recursive_find_python_class([search_in], planner_name3d, current_module="nnunet.experiment_planning")
        if planner_3d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name3d)
    else:
        planner_3d = None

    if planner_name2d is not None:
        planner_2d = recursive_find_python_class([search_in], planner_name2d, current_module="nnunet.experiment_planning")
        if planner_2d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name2d)
    else:
        planner_2d = None

    for t in tasks:
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
        #splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, t)
        #lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

        # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
        dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
        modalities = list(dataset_json["modality"].values())
        collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
        dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=tf)  # this class creates the fingerprint
        _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner


        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
        shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

        threads = (tl, tf)

        print("number of threads: ", threads, "\n")

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                if planner_3d is not None:
                    exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
                    exp_planner.plan_experiment()
                    exp_planner.run_preprocessing(threads)
                if planner_2d is not None:
                    exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir_this_task)
                    exp_planner.plan_experiment()
                    exp_planner.run_preprocessing(threads)


## Adapted from nnunet.inference.predict_simple
def predict_simple_AL(input_folder, output_folder, model_folder_name, 
    model = "3d_fullres",
    folds = ["all",], 
    save_npz = False, 
    lowres_segmentations = None,
    part_id = 0, num_parts = 1,
    num_threads_preprocessing = 6,
    num_threads_nifti_save = 2,
    disable_tta = False,
    overwrite_existing = False,
    mode = 'normal',
    all_in_gpu = None,
    step_size = 0.5,
    chk = 'model_final_checkpoint',
    # chk = 'model_final',
    disable_mixed_precision = False, 
    ):
    assert model in ["2d", "3d_lowres", "3d_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or "  \
                                                                             "3d_cascade_fullres"
                                                                            
    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == None:
        pass
    else:
        raise ValueError("Unexpected value for argument folds")

    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    st = time()
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                                num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                                overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                                mixed_precision=not disable_mixed_precision,
                                step_size=step_size, checkpoint_name=chk)
    end = time()
    save_json(end - st, join(output_folder, 'prediction_time.txt'))
    return

# ## front facing  above's Predict_simple_AL that uses the cbis-ddsm 3d_fullres model
# def predict_simplest_AL(input_folder, output_folder):
#     task_name = 'Task501_cbis-ddsm'
#     model = '2d'

#     model_folder_name = os.path.join('/usr/xtmp/jly16/mammoproj/data/nnUNet_trained_models/nnUNet/',\
#         model,task_name,'nnUNetTrainerV2__nnUNetPlansv2.1')

#     predict_simple_AL(input_folder, output_folder, model_folder_name, model = model)

#     return 

def batch_iou(label_dir, pred_dir, viz_save = True):
    label_filepaths = sorted(glob.glob(f"{label_dir}/*.nii.gz"))
    pred_filepaths = sorted(glob.glob(f"{pred_dir}/*.nii.gz"))

    assert np.all([os.path.basename(label_filepaths[i])==os.path.basename(pred_filepaths[i]) for i in range(len(label_filepaths))]), "files don't match!"
    files = [os.path.basename(label_filepaths[i]) for i in range(len(label_filepaths))]

    ious = []
    for filepath in files:
        # thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1,
        #                 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # try to find the best iou
        # for threshold in thresholds:
        #     unet_seg = get_binary_mask_threshold_torch(
        #         unbinarized_unet_seg, threshold).detach().cpu().numpy()
        #     unet_seg_ff = largest_contiguous_region(
        #         unet_seg)  # flood fill binary segmentation
        #     unet_seg_ff_torch = torch.from_numpy(unet_seg_ff)
        #     # unet_seg = get_binary_mask(unbinarized_unet_seg).cpu()
        #     iou = intersection_over_union_exp(unet_seg_ff_torch, mask)
        #     if (iou > max_iou):
        #         max_iou = iou
        #         thresholded_mask = unet_seg_ff
        pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_dir, filepath)))
        label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_dir, filepath)))
        iou = (label&pred).sum()/(label|pred).sum()

        ious.append(iou)
        test_images_save_path = None
        if (iou < 0.1):
            # save bad iou
            # save unet_seg_threshold
            test_images_save_path = f"/usr/xtmp/jly16/mammoproj/nnunet_integration_tmp/iou_test_images_1/bad/{filepath}"
        if (iou > 0.9):
            # save good iou
            test_images_save_path = f"/usr/xtmp/jly16/mammoproj/nnunet_integration_tmp/iou_test_images_1/good/{filepath}"

        if (not test_images_save_path == None) and viz_save :
            sitk.WriteImage(np.stack([label.numpy(), pred]), test_images_save_path)
            # np.save(test_images_save_path, np.stack(
            #     [label.numpy(), pred]))
    
    return np.average(np.asarray(ious))


class nnunet_model(seg_model.seg_model): 
    def __init__(self, 
                 base_model_task_id = 'Task501_cbis-ddsm', 
                 network = '2d',
                 network_trainer = 'nnUNetTrainerV2',
                 plans_identifier = default_plans_identifier):
        super().__init__()
        if not base_model_task_id.startswith("Task"):
            task_id = int(base_model_task_id)
            base_model_task_id = convert_id_to_task_name(task_id)
        
        self.task_id = base_model_task_id
        self.network = network
        self.network_trainer = network_trainer
        self.plans_identifier = plans_identifier
        self.save_path = None
    
    def load_model(self, train_dir, 
                   fp32 = False, 
                   validation_only = False, 
                   deterministic = False, 
                   use_compressed_data = False, ): 
        
        run_mixed_precision = not fp32
        decompress_data = not use_compressed_data

        plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
            trainer_class = get_default_configuration(self.network, self.task_id, self.network_trainer, self.plans_identifier)
        
        if trainer_class is None:
            raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

        self.trainer = trainer_class(plans_file, 'all', output_folder=output_folder_name, dataset_directory=dataset_directory,
                                batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                                deterministic=deterministic,
                                fp16=run_mixed_precision)
        self.trainer.dataset_directory = train_dir
        self.trainer.initialize(not validation_only)
        self.trainer.load_best_checkpoint()
        self.verts['base model'] = self.trainer.epoch
    
    def update_model(self,num_epochs = 5, ):
        self.trainer.max_num_epochs = self.trainer.epoch + num_epochs
        self.trainer.log_file = None
        self.trainer.use_progress_bar = True
        self.trainer.num_batches_per_epoch = int(np.rint(len(glob.glob( os.path.join(self.trainer.dataset_directory,'gt_segmentations','*') ))/ self.trainer.batch_size))+1
        self.trainer.num_val_batches_per_epoch = self.trainer.num_batches_per_epoch 
        print(f'train batches per epoch: {self.trainer.num_batches_per_epoch} \t val: {self.trainer.num_val_batches_per_epoch} \t batch_size = {self.trainer.batch_size}')
        self.trainer.run_training()
        self.verts[f'Model Updated (epoch = {self.trainer.epoch}'] = self.trainer.epoch
        
    def save_model(self, save_path):
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        self.trainer.save_checkpoint(save_path)
        self.save_path = save_path

        run_dir = os.path.join(os.path.split(save_path)[0], '..')

        if not os.path.isfile(os.path.join(run_dir,'plans.pkl')): 
            plans_pkl = os.path.join(os.environ['RESULTS_FOLDER'],'nnUNet', self.network, \
                                 self.task_id,  self.network_trainer+'__'+self.plans_identifier,'plans.pkl')
        
            shutil.copy2(plans_pkl, run_dir)
    
    def predict(self, input_folder, output_folder = None, correct_save_dir = None, saved_oracle_filepaths = {}):
        ## TODO: update so it uses self.trainer.... see nnunet.inference.predict.predict_from_folder
        model_folder_name = os.path.join(os.path.split(self.save_path)[0],'..')
        # TODO: ADD SAVING CORRECT SAVE_DIR? - done? in "save_files_for_nnunet"... they dont get written over... low pri
        predict_simple_AL(input_folder, output_folder, model_folder_name, model = self.network, chk =os.path.splitext(os.path.split(self.save_path)[1])[0] )
    
    def validate(self, input_folder, output_folder = None):
        ## TODO: update so it uses self.trainer.... see nnunet.inference.predict.predict_from_folder
        model_folder_name = os.path.join(os.path.split(self.save_path)[0],'..')
        if output_folder: 
            predict_simple_AL(input_folder, output_folder, model_folder_name, model = self.network, chk =os.path.splitext(os.path.split(self.save_path)[1])[0] )
            return batch_iou(os.path.join(os.path.split(input_folder)[0], 'labelsTs'), output_folder, viz_save=False)
        else: 
            with tempfile.TemporaryDirectory() as output_folder: 
                predict_simple_AL(input_folder, output_folder, model_folder_name, model = self.network, chk =os.path.splitext(os.path.split(self.save_path)[1])[0] )

                return batch_iou(os.path.join(os.path.split(input_folder)[0], 'labelsTs'), output_folder, viz_save=False)