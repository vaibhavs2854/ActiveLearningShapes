import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np 
import os
import SimpleITK as sitk
import pandas as pd
import cv2


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


def display_image_annotation(filepath,annotations):
    ncols, nrows = 3, len(filepath)
    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(9, 3*len(filepath)+1)
    fig.tight_layout()
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig,hspace=0,wspace=0)
    
    anno_opts = dict(xy=(0.05, 0.05), xycoords='axes fraction', va='bottom', ha='left',color='cyan',fontweight='extra bold',fontsize='8')

    f_axes = []
    for row in range(nrows):
        f_axes.append([])
        for col in range(ncols):
            f_axes[-1].append(fig.add_subplot(spec[row, col]))

    for ax_num, ax in enumerate(f_axes[0]):
            if ax_num == 0:
                ax.set_title("Image", fontdict=None, loc='left', color = "k")
            elif ax_num == 1:
                ax.set_title("Segmentation", fontdict=None, loc='left', color = "k")
            elif ax_num == 2:
                ax.set_title("Overlay", fontdict=None, loc='left', color = "k")

    for row in range(nrows):
        if type(filepath[row][0]) == str: 
            if os.path.splitext(filepath[row][0])[1] == '.png':
                image = (cv2.imread(filepath[row][0]))/255
                mask = (255 - cv2.imread(filepath[row][1]))/255
            elif os.path.splitext(filepath[row][0])[1] == '.npy':
                image = np.load(filepath[row][0])[0]
                mask = np.load(filepath[row][1])[1]
            elif os.path.splitext(filepath[row][0])[1] == '.gz':
                image = sitk.GetArrayFromImage(sitk.ReadImage(filepath[row][0]))[0]
                mask = sitk.GetArrayFromImage(sitk.ReadImage(filepath[row][1]))[0]
            else: 
                print("FILE MUST BE .npy or .gz")
                return fig
        else:
            image = filepath[row][0]
            mask = filepath[row][1]
        f_axes[row][0].imshow(image,cmap='gray')
        f_axes[row][0].set_axis_off()
        
        f_axes[row][1].imshow(mask,cmap='gray')
        f_axes[row][1].set_axis_off()

        heatmap = cv2.applyColorMap(np.uint8(255*(1-mask)), cv2.COLORMAP_AUTUMN)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
                      
        if heatmap.shape[:2] != image.shape[:2]: 
            heatmap = cv2.resize(heatmap, dsize=image.shape, interpolation=cv2.INTER_CUBIC)
        if len(image.shape)==3 and image.shape[2] == 3:
            img = 0.6 * image + 0.3*heatmap
        else:
            img = 0.6 * np.reshape(np.stack([image.copy(),image.copy(),image.copy()],axis=2), (image.shape[0], image.shape[1], 3)) + 0.3*heatmap
        f_axes[row][2].imshow(img)
        f_axes[row][2].set_axis_off()
        
        f_axes[row][0].annotate(annotations[row],**anno_opts)
    return fig
