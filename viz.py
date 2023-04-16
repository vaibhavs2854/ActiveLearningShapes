import matplotlib.pyplot as plt
import numpy as np 
import os
import SimpleITK as sitk
import pandas as pd


def save_img_matrix(filename, run_query_ids, imgs_seen_ops, input_folder, pred_folder, label_folder, save_dir = None, seed = 44, iter_num = 0):
    ## run_query_ids is a list of ids in the format '{run_id}_{query_method}'
    row = len(run_query_ids)
    col = len(imgs_seen_ops)+2
    fig, ax = plt.subplots(row,col, figsize = (col*3, row*3.5), squeeze = True, tight_layout = True)
    fig.suptitle(filename, fontsize = 15)

    for r in range(len(run_query_ids)):
        for i in range(len(imgs_seen_ops)):
            run_id =  f'{run_query_ids[r]}_{imgs_seen_ops[i]}_{seed}'
            pred_folder_run = os.path.join(pred_folder,'Run_'+run_id,f'Iter{iter_num}','ValSegmentations')
            pred_path = os.path.join(pred_folder_run, filename)
            img = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
            if r == 0:
                ax[r, i+1].set_title(f'Images Seen: {imgs_seen_ops[i]}')
            ax[r, i+1].imshow(img[0], cmap = 'gray')
            ax[r, i+1].axis('off')

        label_path = os.path.join(label_folder,filename)
        ax[r, -1].imshow(sitk.GetArrayFromImage(sitk.ReadImage(label_path))[0], cmap = 'autumn_r')
        if r == 0:
            ax[r, -1].set_title(f'Truth Label')

        ax[r, -1].axis('off')

        img_path = os.path.join(input_folder, filename[:-7]+'_0000.nii.gz')
        ax[r, 0].imshow(sitk.GetArrayFromImage(sitk.ReadImage(img_path))[0], cmap = 'gray')
        if r == 0:
            ax[r, 0].set_title(f'Source Image')
        ax[r, 0].set_ylabel(run_query_ids[r],rotation=90, labelpad=20, fontsize = 10)
        ax[r,0].set_xticks([])
        ax[r,0].set_yticks([])
        # ax[r, 0].axis('off')
        # ax[r,0].annotate(run_query_ids[r], xy=(0, 0.5), xytext=(-ax[r,0].yaxis.labelpad - 5, 0),
        #         xycoords=ax[r,0].yaxis.label, textcoords='offset points',
        #         size='large', ha='right', va='center')
    
    if save_dir != None:
        plt.savefig(os.path.join(save_dir, filename+'model_comp.png'), dpi = 500)
    return fig, ax


def plot_save_val_metrics(run_ids, csv_folder, save_dir = None):
    for run_id in run_ids: 
        df = pd.read_csv(os.path.join(csv_folder, "Run_"+run_id,'experiment_output.csv'))
        for query_type in df.query_type.unique():
            df_q = df.loc[df['query_type'] == query_type]
            plt.plot(df_q.imgs_seen, df_q.IOU, label = run_id + '_' + query_type )

    plt.legend()
    if save_dir != None:
        plt.savefig(os.path.join(save_dir, '__'.join(run_ids) + '_valIOU.png'), dpi = 500)