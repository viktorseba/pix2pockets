#%%
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:58:07 2023

@author: jonas
"""
import torch
import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from yolo_code_that_we_use import *
from tqdm import tqdm
from confusion_matrix_class import ConfusionMatrix

#%matplotlib inline
#%%

def get_detection(im_path, model, post_process=True, conf_thresh=0.4, iou_thresh=0.5, output_format = 'corner'):
    im = Image.open(im_path)
    results = model(im)
    
    detection = results.pandas().xyxy[0].to_numpy()
    
    xmin = detection[:,0]
    ymin = detection[:,1]
    xmax = detection[:,2]
    ymax = detection[:,3]
    w = xmax-xmin
    h = ymax-ymin
    
    if output_format == 'corner':
        transformed_boxes = np.array([xmin,ymin,w,h]).T
    elif output_format == 'center':
        x_center = xmin + w/2
        y_center = ymin + h/2
        transformed_boxes = np.array([x_center, y_center, w, h]).T
    else:
        print(f"output_format must either be 'corner' or 'center'. Your input was {output_format}.")
    detection[:, :4] = transformed_boxes
    
    detection = detection[:,:-1].astype(np.float64)
    
    if post_process:
        detection = detection[detection[:, 4].argsort()[::-1]]  # Sort according to confidence

        # Run non-max-suppresion
        keep_list = np.arange(0, detection.shape[0])
        remove_list = []
        iou = bb_IoU(detection[:, :4], detection[:, :4]) - np.eye(detection.shape[0])
        
        overlaps = np.where((iou >= iou_thresh))
        overlaps = np.stack(overlaps, 1)
        for idx1, idx2 in overlaps:
            conf1 = detection[idx1,4]
            conf2 = detection[idx2,4]
            if conf1 <= conf2: remove_list.append(idx1)
            else: remove_list.append(idx2)
        
        remove_list = np.unique(remove_list)
        keep_list = [k for k in keep_list if k not in remove_list]
        
        detection = detection[keep_list]
        
        # print(detection.shape[0])
        # Keep 7 highest striped and solids 1 cue and black and 18 dots
        for c, n in zip([0, 1, 2, 3, 4], [7, 7, 1, 1, 18]):
            # print(detection[:, 5])
            idxs = [i for i, x in enumerate(detection[:, 5]) if x == c]
            keep = idxs[:n]
            remove = idxs[n:]
            
            for i in keep:
                detection[i, 4] = max(detection[i, 4], conf_thresh)
            
            # print(c, len(idxs), len(keep), len(remove))
            for j in remove:
                detection[j, 4] = 0.0
            
        # Remove detections with conf_score less than threshhold
        detection = detection[detection[:, 4] >= conf_thresh]
    
    return detection

def get_label(label_path, im_size):
    ann = []
    f = open(label_path)
    lines = f.readlines()
    
    for line in lines:
        ann.append([float(x) for x in line.split(' ')])
    
    ground_truth = np.array(ann)
    img_h, img_w = im_size
    
    #print(bbox)
    xc = ground_truth[:, 1]
    yc = ground_truth[:, 2] 
    wc = ground_truth[:, 3]
    hc = ground_truth[:, 4]
    
    w = wc*img_w
    h = hc*img_h
    
    x = xc*img_w - w/2
    y = yc*img_h - h/2
    transformed_boxes = np.array([x,y,w,h]).T
    
    #labels = np.array([int(x) for x in ground_truth[:,0]])
    ground_truth[:,1:] = transformed_boxes
    #ground_truth_new = np.vstack((labels,transformed_boxes))
    ground_truth[:,0] = ground_truth[:,0].astype(int)
    return ground_truth


def helper_plot_bbox(ax, boxes, valid_classes, colors, kind="detection"):
    alpha = (0.8,)
    class_count = [0,0,0,0,0]
    classes = []
    for bb in boxes:
        if kind == "detection":
            cl = int(bb[5]) # class
            if valid_classes[cl]:
                # print(cl)
                x,y,w,h = bb[:4]
                c = colors[cl] + alpha
                class_count[cl] += 1
                classes.append(cl)
                ax.add_patch(patches.Rectangle((x,y),w,h,linewidth=1,edgecolor=c,facecolor='none'))
            
        elif kind == "label":
            cl = int(bb[0]) # class
            if valid_classes[cl]:
                x,y,w,h = bb[1:]
                c = colors[cl]
                class_count[cl] += 1
                classes.append(cl)
                ax.add_patch(patches.Rectangle((x,y),w,h,linewidth=1,edgecolor=c,facecolor='none'))
        
        
    return classes, class_count

def plot_bboxes(im, label, detection, 
                im_idx=None, data_name=None, 
                classes_to_plot=['Striped', 'Solid', 'Cue', 'Black', 'Dot'], 
                title_names=['Ground Truth', 'Detection'], 
                legend=True, save_path=None, show_axes=False):
    
    boxes = []
    to_plot = []
    artist = []
    if label is not None:
        boxes.append(label)
        to_plot.append("label")
        artist.append(mlines.Line2D([], [], color='k', linestyle='-', label=f'{title_names[0]}: {label.shape[0]}'))
    
    if detection is not None:
        boxes.append(detection)
        to_plot.append("detection")
        artist.append(mlines.Line2D([], [], color='k', linestyle='--', label=f'{title_names[1]}: {detection.shape[0]}'))
    
    N_plots = len(to_plot)
    if N_plots == 0:
        return None

    classes = ['Striped', 'Solid', 'Cue', 'Black', 'Dot']
    valid_classes = [True if x in classes_to_plot else False for x in classes]
    colors = [(0,0,1),(0,1,0),(1,0,0),(0,1,1),(1,0,1),(1,1,0)]
    
    found_classes = []
    found_classes_count = []
    
    fig, axs = plt.subplots(1, N_plots, figsize=(10,5*N_plots), sharex=True, sharey=True)
    
    for i, (ax, kind) in enumerate(zip(np.array(axs).reshape(-1), to_plot)):
        ax.imshow(im)
        if not show_axes: ax.axis('off')
        cl, class_count = helper_plot_bbox(ax, boxes[i], valid_classes, colors, kind=kind)
        found_classes.append(cl)
        found_classes_count.append(class_count)
        
    found_classes_count = np.array(found_classes_count).T
    unique_classes = np.unique([x for fc in found_classes for x in fc])
    if len(unique_classes) != 0:
        for c in unique_classes:
            c = int(c)
            artist.append(patches.Patch(color=colors[c], label = classes[c] + f': {found_classes_count[c]}'))
        # fig.legend(loc='upper center',handles=artist,bbox_to_anchor=(0.5,-0.05),
                  # fancybox=True, ncol=len(artist))
        if legend: fig.legend(loc='center', handles=artist, bbox_to_anchor=(1, 0.5), prop={"size": 10})
    
    if (data_name is not None) or (im_idx is not None):
        fig.suptitle(f'Bounding boxes for image {im_idx} in dataset {data_name}')
        
    fig.tight_layout
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=500)
    
    plt.show()
    return None

def compare_post_procsess(im, label, detection, detection_post, im_idx=None, data_name=None, title_names=['Ground Truth', 'No post processing', 'With post proccessing'], save_path=None):

    # Get image and plot
    fig, axs = plt.subplots(2,2, figsize=(8,5), sharex=True, sharey=True)
    # axs[0] is labels, axs[1] is detections
    axs[0][0].imshow(im) #Label
    axs[1][0].imshow(im) #detection
    axs[1][1].imshow(im) #detection post
    for ax in axs:
        ax[0].axis('off')
        ax[1].axis('off')

    # ax = plt.gca()
    #colors = ['b','g','r','c','m','y']
    colors = [(0,0,1),(0,1,0),(1,0,0),(0,1,1),(1,0,1),(1,1,0)]
    alpha = (0.8,) #The comma is necessary (maybe alpha doesnt work on edgecolors?)
    classes = ['Striped', 'Solid', 'Cue', 'Black', 'Dot']
    artist = []
    class_count_label = [0,0,0,0,0]
    class_count_det = [0,0,0,0,0]
    class_count_post = [0,0,0,0,0]
    # artist.append(patches.Patch(color=(1,1,1), label = ''))
    # Plot labels
    if label is not None:
        n_lab = label.shape[0]
        label_classes = label[:,0]
        for bb in label:
            #print(bb)
            x,y,w,h = bb[1:]
            c = colors[int(bb[0])]
            class_count_label[int(bb[0])] += 1
            axs[0][0].add_patch(patches.Rectangle((x,y),w,h,linewidth=1,edgecolor=c,facecolor='none'))
        artist.append(mlines.Line2D([], [], color='k', linestyle='-', label=f'{title_names[0]}: {n_lab}'))
    axs[0][0].set_title(title_names[0], fontsize=20)
    
    # Get detection bounding boxes
    if detection is not None:
        n_det = detection.shape[0]
        detection_classes = detection[:,5]
        for bb in detection:
            x,y,w,h = bb[:4]
            c = colors[int(bb[5])] + alpha
            class_count_det[int(bb[5])] += 1
            axs[1][0].add_patch(patches.Rectangle((x,y),w,h,linestyle='-',linewidth=1,edgecolor=c,facecolor='none'))
        artist.append(mlines.Line2D([], [], color='k', linestyle='-', label=f'{title_names[1]}: {n_det}'))
    axs[1][0].set_title(title_names[1], fontsize=20)  
    
    if detection_post is not None:
        n_det_post = detection_post.shape[0]
        detection_classes_post = detection_post[:,5]
        for bb in detection_post:
            x,y,w,h = bb[:4]
            c = colors[int(bb[5])] + alpha
            class_count_post[int(bb[5])] += 1
            axs[1][1].add_patch(patches.Rectangle((x,y),w,h,linestyle='-',linewidth=1,edgecolor=c,facecolor='none'))
        artist.append(mlines.Line2D([], [], color='k', linestyle='-', label=f'{title_names[2]}: {n_det_post}'))
    axs[1][1].set_title(title_names[2], fontsize=20)  
    
    
    unique_classes = np.unique(np.concatenate((label_classes,detection_classes)))
    
    #print(label_classes, detection_classes)
    #print(unique_classes)
    
    if len(unique_classes) != 0:
        for c in unique_classes:
            c = int(c)
            artist.append(patches.Patch(color=colors[c], label = classes[c] + f': {class_count_label[c]}, {class_count_det[c]}, {class_count_post[c]}'))
        # fig.legend(loc='upper center',handles=artist,bbox_to_anchor=(0.5,-0.05),
                  # fancybox=True, ncol=len(artist))
        fig.legend(loc='center',handles=artist,bbox_to_anchor=(0.69,0.7), frameon=False, prop={"size":12})
    
    # fig.suptitle('Bounding boxes')
    # if (data_name is not None) or (im_idx is not None):
    #     fig.suptitle(f'Bounding boxes for image {im_idx} in dataset {data_name}')
    fig.tight_layout
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=500)
    
    plt.show()
    return None

def evaluate_batch(im_paths, label_paths, post_processing=False, plot=False, names=(), data_name = None):
    """

    Parameters
    ----------
    im_paths : list 
        list containing paths to individual images
        
    label_paths : list 
        list containing paths to individual labels
    plot : Bool, optional
        Set to True if plots should be shown. The default is False.
    names : Dict, optional
        Dictionary of the classes in the dataset. The default is ().

    Returns
    -------
    list
        mp, mr, map50, map, maps
        mean precision, mean recall, mean average precision @ 0.5, mean average precision @ 0.5:0.95, mean average precision pr class

    """
    if isinstance(im_paths, str) and isinstance(label_paths, str):
        im_paths = [im_paths]
        label_paths = [label_paths]
    
    if len(im_paths) != len(label_paths):
        raise Exception("Number of images and labels do not match")
    nc = 6
    iouv = np.linspace(0.5, 0.95, 10) # iou values
    stats = []
    
    # Iterate through images
    for im_path, label_path in tqdm(zip(im_paths, label_paths)):
        
        #print(im_path)
        #print(label_path)
        
        # Get image
        im = cv2.imread(im_path)
        im_size = im.shape[:2]
        
        # Get detection and labels
        if post_processing: detections = get_detection(im_path, model)
        else: detections = get_detection(im_path, model, post_process=False, conf_thresh=0, iou_thresh=0)
        labels = get_label(label_path, im_size)
        
        # for each detection-label pair, compute 'correct'
        correct = process_batch(detections, labels, iouv)
        # append (correct, conf, pcls, tcls) to a 'stat' variable
        stats.append([correct, detections[:, 4], detections[:, 5], labels[:, 0]])
    
    # Now make sure stat is in the right format (list(4) (correct, conf, pcls, tcls))
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plot, names=names, data_name=data_name)
    
    # return tp, fp, p, r, f1, ap, ap_class, stats

    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    
    
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return [mp, mr, map50, map, maps], [tp, fp, p, r, f1, ap, ap_class]
 
    

def table_analysis(im_paths, label_paths, model, data_name=None):
    # Every image becomes a point (table_area/image_area, performance)
    
    # perf_names = ['F1', 'map50', 'map'] #F1-score, mAP@0.5, mAP@0.4:0.95
    perf_names = ['F1', 'map50']
    table_ratio = []
    perf = {x: [] for x in perf_names}
    
    # Iterate through images
    for im_path, label_path in tqdm(zip(im_paths, label_paths)):
        # Get image
        im = cv2.imread(im_path)
        im_size = im.shape[:2]
        
        # Get detection and labels
        detection = get_detection(im_path, model)
        label = get_label(label_path, im_size)
        
        
        # Calculate table area from dots
        # idx_det = np.where(detection[:, 5] == 4)[0]
        idx_det = detection[:, 5] == 4
        
        # print(sum(idx_det))
        if len(detection[idx_det][:, 0]) == 0:
            continue
        
        x_min = np.min(detection[idx_det][:, 0])
        y_min = np.min(detection[idx_det][:, 1])
        
        x_max = np.max(detection[idx_det][:,0] + detection[idx_det][:,2])
        y_max = np.max(detection[idx_det][:,1] + detection[idx_det][:,3])
        # idx_lab = np.where(label[:,0] == 4)[0]
        
        # iou = bb_IoU(detection[idx_det, :4], label[idx_lab, 1:])
        # table_bb = detection[idx_det[np.argmax(iou)]]
        
        #print(iou, idx_det, idx_det[np.argmax(iou)])
        
        # table_area = table_bb[2]*table_bb[3]
        table_area = (x_max - x_min) * (y_max - y_min)

        table_ratio.append(table_area/(im_size[0]*im_size[1])) 
        
        # Get performances
        # op, stat = evaluate_batch(im_path, label_path)
        # mp, mr, map50, map, maps = op
        # f1 = 2*((mp*mr)/(mp+mr))
        cm = ConfusionMatrix(nc=5, conf=0.0, iou_thres=0.5)
        cm.process_batch(detection, label)
        metrics = cm.calc_metrics_pr_class()
        
        # mAP = metrics["mAP"]
        perf['F1'].append(np.mean(metrics["f1"]))
        perf['map50'].append(metrics["mAP"])
        # perf['map'].append()
    
    # plot curves!
    
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    
    ax.grid()
    for i, name in enumerate(perf):
        ax.scatter(table_ratio, perf[name], linewidth=1, label=name)  


    ax.set_xlabel('table ratio')
    ax.set_ylabel('Performance')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Table Analysis')
    if data_name is not None:
        ax.set_title(f'Table Analysis, Dataset: {data_name}')
    #fig.savefig(save_dir, dpi=250)
    plt.show()
    #plt.close(fig)
    
    return (table_ratio, perf)

def load_dataset(dataset_name, folder_type='test', yolo_path='yolov5', weight_name=None):
    """

    Parameters
    ----------
    dataset_name : str
        Path to the dataset. The dataset should include train, test 
        and val folders, as well as the model weight that was created from
        this dataset.
    folder_type : str, optional
        'train', 'val' or 'test'. The default is 'test'.
    yolo_path : str, optional
        Path to the yolo folder. The default is 'yolov5'.

    Returns
    -------
    model : yolo model
    
    im_paths : list
        list containing paths to individual images in the 
        dataset//folder_type//images folder.
    label_paths : list
        List containing paths to individual labels in the
        dataset//folder_type//labels folder.

    """
    weight_path = None
    for filename in os.listdir(dataset_name):
        
        # find weight
        if filename[-3:] == '.pt':
            if weight_name is None: 
                weight_path = os.path.join(dataset_name, filename)
            elif filename[:-3] == weight_name:
                weight_path = os.path.join(dataset_name, filename)
            
                
        
        # print(weight_path)
        # find type folder
        if filename == folder_type:
            im_folder = os.path.join(dataset_name, folder_type, 'images')
            label_folder = os.path.join(dataset_name, folder_type, 'labels')

        #print(filename)
    if weight_path is None:
        return print(f"No weight was found with filename {weight_name}")
    
    im_paths = []
    for filename in sorted(os.listdir(im_folder), key=len):
        #print(os.path.join(im_folder, filename))
        im_paths.append(os.path.join(im_folder, filename))


    label_paths = []
    for filename in sorted(os.listdir(label_folder), key=len):
        #print(os.path.join(label_folder, filename))
        label_paths.append(os.path.join(label_folder, filename))
        
    print(f"Found weight path: {weight_path}")
    model = torch.hub.load(yolo_path, 'custom', path=weight_path, source='local')
    model.eval()
    
    name = dataset_name.split('\\')[-1]
    return model, im_paths, label_paths, name
    

def set_im_and_label_paths_to_all(folder_to_all_ims):
    im_folder = os.path.join(folder_to_all_ims, 'images')
    label_folder = os.path.join(folder_to_all_ims, 'labels')
    
    im_paths = []
    for filename in sorted(os.listdir(im_folder), key=len):
        #print(os.path.join(im_folder, filename))
        im_paths.append(os.path.join(im_folder, filename))


    label_paths = []
    for filename in sorted(os.listdir(label_folder), key=len):
        #print(os.path.join(label_folder, filename))
        label_paths.append(os.path.join(label_folder, filename))
    
    return im_paths, label_paths
#%%
dataset_name = 'datasets\\dataset155' 
# dataset_name = 'datasets\\pool100_dataset'
folder_type = 'test'
#folder_type = 'train'
yolo_path = 'yolov5'

model, im_paths, label_paths, name1 = load_dataset(dataset_name, folder_type, yolo_path)

folder_to_all_ims = 'Dataset_195\\yolo_format'
im_paths, label_paths = set_im_and_label_paths_to_all(folder_to_all_ims)
#12
#%% Check if plot is working.....
im_idx = 0
im = cv2.imread(im_paths[im_idx])[:,:,::-1]
im_size = im.shape[:2]
plt.imshow(im)
plt.show()
#labels = get_label(label_paths[im_idx],im_size)

#%% Plot bounding boxes for each image in the folder
im_idxes = [168]
for im_idx in im_idxes:
    im = cv2.imread(im_paths[im_idx])[:,:,::-1]
    im_size = im.shape[:2]
    
    detections = get_detection(im_paths[im_idx], model)
    labels = get_label(label_paths[im_idx], im_size)
    
    im_number = int(r"{path}".format(path = im_paths[im_idx]).split('\\')[-1].split('.')[0])
    
    classes_to_plot=['Striped', 'Solid', 'Cue', 'Black', 'Dot']
    # classes_to_plot=['Dot']
    # save_path = f"Billeder til paper\\{im_idx}_only_dots"
    plot_bboxes(im, labels, detections, im_idx= im_number, data_name=name1, legend=True)
    # plot_bboxes(im, labels, None, classes_to_plot=classes_to_plot, im_idx=None, data_name=None, legend=False, save_path=save_path)
    #plt.imshow(im)
    #plt.show()


#%% Compute combined performance for every image in the folder
names = model.names
#op = evaluate_batch(detections, labels, plot=True, names=names)
op, stat = evaluate_batch(im_paths[1], label_paths[1], plot=False, names=names)

#%% Compute performance individually for each image in the folder
names = model.names
for img_id in range(10):
    im = cv2.imread(im_paths[img_id])
    im_size = im.shape[:2]

    #detections = get_detection(im_paths[img_id], model)
    #labels = get_label(label_paths[img_id], im_size)
    op = evaluate_batch(im_paths[img_id], label_paths[img_id], plot=True, names=names)

#%% Run table analysis
table_ratio, perf = table_analysis(im_paths, label_paths, model, data_name=name1)


#%%
#%% Load different datasets and run...
folder_type = 'test'
yolo_path = 'yolov5'

datasets = []
for p in sorted(os.listdir('datasets'), key=len):
    #if p not in ['dataset100_602020','dataset100_403030']: #These datasets doesnt have a weight yet
        print(p)
        datasets.append(os.path.join('datasets', p))
#%% table analysis
data = datasets[-1]
model, im_paths, label_paths, name = load_dataset(data, folder_type, yolo_path)
print(f'\nCurrent dataset: {name}\n')
# table_ratio, perf = table_analysis(im_paths, label_paths, model, data_name= name)
print('\n')

#%% Compute performance for each dataset and plot
xaxis = []
yaxis = []
for data in datasets:
    data_size = data.split('dataset')[-1]
    xaxis.append(data_size) #Get number of images in train
    
    # weight_name = str(data_size)+'best'
    weight_name=None
    
    model, im_paths, label_paths, d_name = load_dataset(data, folder_type, yolo_path)
    print(f'\nCurrent dataset: {d_name}\n')
    mod_names = model.names
    # op = evaluate_batch(im_paths, label_paths, plot=True, names=mod_names, data_name=d_name)
    cm = ConfusionMatrix(nc=5, conf=0.0, iou_thres=0.5)
    for img_id in tqdm(range(len(im_paths))):
        im = cv2.imread(im_paths[img_id])[:,:,::-1]
        im_size = im.shape[:2]
        
        detections = get_detection(im_paths[img_id], model, post_process=True, conf_thresh=0.4, iou_thresh=0.5)
        labels = get_label(label_paths[img_id], im_size)
        
        cm.process_batch(detections, labels)
    
    print('\n')
    metrics = cm.calc_metrics_pr_class()
    
    yaxis.append(metrics["mAP"])
#%%
fig, ax = plt.subplots(1, 1, figsize=(5, 3), tight_layout=True)
# ax.scatter(xaxis,yaxis)
ax.plot(xaxis, yaxis, '-o')
ax.set_ylim(0,1)
ax.set_xlabel('train size')
ax.set_ylabel('Performance mAP@50')
ax.set_title('Performance of model as the train-size increases')
plt.show()
# fig.savefig("Thesis images/train_size_performance.png", bbox_to_anchor='tight', dpi=500)
    
    

#%% Pick a dataset and plot bounding boxes
# data_idx = 4

for data_idx in range(len(datasets)):
    folder_type = 'test'
    yolo_path = 'yolov5'
    
    
    current_dataset = datasets[data_idx]
    model, im_paths, label_paths, dname1 = load_dataset(current_dataset, folder_type, yolo_path)
    
    
    for img_id in range(len(im_paths)):
        im = cv2.imread(im_paths[img_id])[:,:,::-1]
        im_size = im.shape[:2]
        
        detections = get_detection(im_paths[img_id], model)
        labels = get_label(label_paths[img_id], im_size)
        
        d_set = current_dataset.split('\\')[1]
        savepath = f"Thesis images/test_detection_results/{d_set}/det_err_{img_id}.png"
        # plot_bboxes(im, labels, None, im_idx = img_id, data_name = dname1)
        # plot_bboxes(im, None, detections, im_idx = img_id, data_name = dname1)
        plot_bboxes(im, labels, detections, title_names=['Ground Truth', 'Detection'], save_path=savepath)
        #plt.imshow(im)
        #plt.show()

#%% Show difference of post processing in get_detection
folder_type = 'test'
yolo_path = 'yolov5'
# weight_name = str(data_size)+'best'

current_dataset = datasets[-1]
data_size = current_dataset.split('dataset')[-1]
weight_name = str(data_size)+'best_2000'
model, im_paths, label_paths, dname1 = load_dataset(current_dataset, folder_type, yolo_path, weight_name=weight_name)

#%%
tp, fp, p, r, f1, ap, ap_class, stats = evaluate_batch(im_paths, label_paths, post_processing=False)
tp1, fp1, p1, r1, f11, ap1, ap_class1, stats1 = evaluate_batch(im_paths, label_paths, post_processing=True)

#%%

img_id = 12
im = cv2.imread(im_paths[img_id])[:,:,::-1]
im_size = im.shape[:2]
labels = get_label(label_paths[img_id], im_size)
detections = get_detection(im_paths[img_id], model, post_process=False, conf_thresh=0, iou_thresh=0)
# detections = np.concatenate((detections[:,5:], detections[:,:4]), axis=1)
detections_post = get_detection(im_paths[img_id], model)
compare_post_procsess(im, labels, detections, detections_post)
plt.show()

#%%
iouv = np.linspace(0.5, 0.95, 10)
stats = []

# for each detection-label pair, compute 'correct'
correct = process_batch(detections, labels, iouv)
# append (correct, conf, pcls, tcls) to a 'stat' variable
stats.append([correct, detections[:, 4], detections[:, 5], labels[:, 0]])

# Now make sure stat is in the right format (list(4) (correct, conf, pcls, tcls))
stats = [np.concatenate(x, 0) for x in zip(*stats)]
tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, names='', data_name='')


stats1 = []
correct1 = process_batch(detections_post, labels, iouv)
# append (correct, conf, pcls, tcls) to a 'stat' variable
stats1.append([correct1, detections_post[:, 4], detections_post[:, 5], labels[:, 0]])

# Now make sure stat is in the right format (list(4) (correct, conf, pcls, tcls))
stats1 = [np.concatenate(x, 0) for x in zip(*stats1)]
tp1, fp1, p1, r1, f11, ap1, ap_class1 = ap_per_class(*stats1, plot=False, names='', data_name='')


#%%
for img_id in range(len(im_paths)):
# for img_id in range(1):
    im = cv2.imread(im_paths[img_id])[:,:,::-1]
    im_size = im.shape[:2]
    
    labels = get_label(label_paths[img_id], im_size)
    detections = get_detection(im_paths[img_id], model, post_process=False, conf_thresh=0, iou_thresh=0)
    # detections = np.concatenate((detections[:,5:], detections[:,:4]), axis=1)
    detections_post = get_detection(im_paths[img_id], model)
    
    # tp, fp, p, r, f1, ap, ap_class, stats = evaluate_batch(im_paths[img_id], label_paths[img_id])
    # d_set = current_dataset.split('\\')[1]
    # savepath = f"Thesis images/test_detection_results/{d_set}/post_proc_{img_id}.png"
    # plot_bboxes(im, labels, None, im_idx = img_id, data_name = dname1)
    # plot_bboxes(im, None, detections, im_idx = img_id, data_name = dname1)
    compare_post_procsess(im, labels, detections, detections_post)
    # plt.imshow(im)
    plt.show()


#%% Single image
img_id = 0
im = cv2.imread(im_paths[img_id])[:,:,::-1]
im_size = im.shape[:2]

detections = get_detection(im_paths[img_id], model)
labels = get_label(label_paths[img_id], im_size)


plot_bboxes(im, labels, None, im_idx = img_id, data_name = dname1)
plot_bboxes(im, None, detections, im_idx = img_id, data_name = dname1)
plot_bboxes(im, labels, detections, im_idx = img_id, data_name = dname1)


#%% Confusion matrix

cm = np.zeros((7,7))
for img_id in range(len(im_paths)):
    im = cv2.imread(im_paths[img_id])[:,:,::-1]
    im_size = im.shape[:2]
    detections = get_detection(im_paths[img_id], model)
    labels = get_label(label_paths[img_id], im_size)
    cm += Confusion_Matrix(labels, detections)
    
plt.imshow(cm)
plt.show()

#%% Confusion matrix class
from confusion_matrix_class import ConfusionMatrix
cm_class_nopost = ConfusionMatrix(nc=5, conf=0.0, iou_thres=0.5)
# cm_class_withpost = ConfusionMatrix(nc=5, conf=0.4, iou_thres=0.4)
cm_class_withpost2 = ConfusionMatrix(nc=5, conf=0.0, iou_thres=0.5)
c_thresh = 0.4
for img_id in range(len(im_paths)):
# for img_id in [4]:
    print(img_id)
    im = cv2.imread(im_paths[img_id])[:,:,::-1]
    im_size = im.shape[:2]
    
    detections_withpost = get_detection(im_paths[img_id], model, post_process=True, conf_thresh=0.4, iou_thresh=0.5)
    detections_nopost = get_detection(im_paths[img_id], model, post_process=False)
    labels = get_label(label_paths[img_id], im_size)
    # plot_bboxes(im, labels, detections_withpost)
    # compare_post_procsess(im, labels, detections_nopost, detections_withpost, save_path="testtest_im4")
    # compare_post_procsess(im, labels, detections_nopost, detections_withpost)
    cm_class_nopost.process_batch(detections_nopost, labels)
    # cm_class_withpost.process_batch(detections_withpost, labels)
    cm_class_withpost2.process_batch(detections_withpost, labels)
    

cm_class_nopost.plot(normalize=False, names=list(model.names.values()), conf_thresh=c_thresh)
# cm_class_withpost.plot(normalize=False, names=list(model.names.values()), conf_thresh=c_thresh)
cm_class_withpost2.plot(normalize=False, names=list(model.names.values()), conf_thresh=c_thresh)
metrics = cm_class_withpost2.calc_metrics_pr_class()
print(cm_class_nopost.calc_metrics_pr_class())
print(cm_class_withpost2.calc_metrics_pr_class())

# print(cm_class_nopost.tps)
# Image 18, recall lowers with post processing
#%%

iou = bb_IoU(labels[:, 1:], detections_nopost[:, :4])

x = np.where(iou >= 0.5)
if x[0].shape[0]:
    # matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
    matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None], detections_nopost[x[1], 4][:, None]), 1)
    # return matches
    if x[0].shape[0] > 1:
        matches1 = matches[matches[:, 3].argsort()[::-1]]
        # matches15 = matches1[matches1[:, 2].argsort()[::-1]]
        matches2 = matches1[np.unique(matches1[:, 1], return_index=True)[1]]
        # matches22 = matches15[np.unique(matches15[:, 1], return_index=True)[1]]
        matches25 = matches2[matches2[:, 3].argsort()[::-1]]
        # matches3 = matches2[np.unique(matches2[:, 0], return_index=True)[1]]
        matches33 = matches25[np.unique(matches25[:, 0], return_index=True)[1]]

m0, m1, _, _ = matches33.transpose().astype(int)

#%%
iou = bb_IoU(labels[:, 1:], detections_nopost[:, :4])

x = np.where(iou >= 0.5)
if x[0].shape[0]:
    # matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
    matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None], detections_nopost[x[1], 4][:, None]), 1)
    # return matches
    if x[0].shape[0] > 1:
        matches1 = matches[matches[:, 3].argsort()[::-1]]
        # matches15 = matches1[matches1[:, 2].argsort()[::-1]]
        matches2 = matches1[np.unique(matches1[:, 1], return_index=True)[1]]
        # matches22 = matches15[np.unique(matches15[:, 1], return_index=True)[1]]
        matches25 = matches2[matches2[:, 3].argsort()[::-1]]
        # matches3 = matches2[np.unique(matches2[:, 0], return_index=True)[1]]
        matches33 = matches25[np.unique(matches25[:, 0], return_index=True)[1]]

m0, m1, _, _ = matches33.transpose().astype(int)

# for i, gc in enumerate(gt_classes):
#     j = m0 == i
#     if n and sum(j) == 1:
#         self.matrix[detection_classes[m1[j]][0], gc] += 1  # correct
#     else:
#         self.matrix[self.nc, gc] += 1  # true background

# if n:
#     for i, dc in enumerate(detection_classes):
#         if not any(m1 == i):
#             self.matrix[dc, self.nc] += 1  # predicted background
#%% Boundingboxes of images not from a dataset
im_break = 'break_im.png'

im = cv2.imread(im_break)[:,:,::-1]
im_size = im.shape[:2]

detections = get_detection(im_break, model)
#labels = get_label(label_paths[im_idx], im_size)

plot_bboxes(im, None, detections)


#%% Find all errors in test folder
dataset_idx = 4
dataset_name = datasets[dataset_idx]
folder_type = 'test'
yolo_path = 'yolov5'

model, im_paths, label_paths, name1 = load_dataset(dataset_name, folder_type, yolo_path)

#%% Go thorugh all datasets and test images and plot the confusion matrix

for data_idx in range(len(datasets)):
# for data_idx in [4]:
    folder_type = 'test'
    yolo_path = 'yolov5'
    
    
    current_dataset = datasets[data_idx]
    # data_size = current_dataset.split('dataset')[-1]
    # weight_name = str(data_size)+'best'
    weight_name=None
    d_set = current_dataset.split('\\')[1]
    model, im_paths, label_paths, dname1 = load_dataset(current_dataset, folder_type, yolo_path, weight_name=weight_name)
    
    conf_thresh=0.4
    iou_thresh=0.4
    # iouv = np.array([0.4])
    cm_list = [ConfusionMatrix(nc = 5, conf=conf_thresh, iou_thres=iou_thresh) for _ in range(21)]
    
    # for i in range(len(im_paths)):
    for i in [4]:
        im = cv2.imread(im_paths[i])[:,:,::-1]
        im_size = im.shape[:2]
        save_path = f"Thesis images/test_detection_results/{d_set}_2000_epochs/CM_{i}_noPost.png"
        
        # detections = get_detection(im_paths[i], model, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        detections = get_detection(im_paths[i], model, post_process=False, conf_thresh=0, iou_thresh=0)
        labels = get_label(label_paths[i], im_size)
        cm_list[i].process_batch(detections, labels)
        cm_list[i].plot(normalize=True, names=list(model.names.values()), conf_thresh=conf_thresh, im_id = i, save_path=save_path) 
        # fig.savefig(save_path)
        cm_list[-1].process_batch(detections, labels)
    
    # save_path = f"Thesis images/test_detection_results/{d_set}_2000_epochs/CM_all.png"
    # cm_list[-1].plot(normalize=True, names=list(model.names.values()), conf_thresh=conf_thresh, im_id = 'All', save_path=save_path) 

#%%
plot_bboxes(im, None, detections[17:19,:])



#%% process batch understanding..
iou_matrix = bb_IoU(labels[:, 1:], detections[:, :4])
iouv = np.linspace(0.5,0.95, 10)
correct_c = labels[:, 0:1] == detections[:, 5]

x = np.where((iou_matrix >= iouv[0]) & correct_c)
matches = np.concatenate((np.stack(x, 1), iou_matrix[x[0], x[1]][:,None]), 1)  # [label, detect, iou]

matches1 = matches[matches[:, 2].argsort()[::-1]] # sort iou descending

matches2 = matches[np.unique(matches[:, 1], return_index=True)[1]] # remove duplicate detections
# matches = matches[matches[:, 2].argsort()[::-1]]
matches3 = matches[np.unique(matches[:, 0], return_index=True)[1]] # remove duplicate ground truths

#%%
correct = process_batch(detections, labels, iouv)

#%%
stats = []
stats.append((correct, detections[:, 4], detections[:, 5], labels[:, 0])) #(correct, conf, pcls, tcls)

stats1 = [np.concatenate(x, 0) for x in zip(*stats)] 

#%% ap_pr_class test
tp, conf, pred_cls, target_cls = correct, detections[:, 4], detections[:, 5], labels[:, 0]
plot = True
names = model.names
eps=1e-16

i = np.argsort(-conf)
tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

# Find unique classes
unique_classes, nt = np.unique(target_cls, return_counts=True)
nc = unique_classes.shape[0]  # number of classes, number of detections

# Create Precision-Recall curve and compute AP for each class
px, py = np.linspace(0, 1, 1000), []  # for plotting
ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
for ci, c in enumerate(unique_classes):
    i = pred_cls == c
    n_l = nt[ci]  # number of labels
    n_p = i.sum()  # number of predictions
    if n_p == 0 or n_l == 0:
        continue

    # Accumulate FPs and TPs
    fpc = (1 - tp[i]).cumsum(0)
    tpc = tp[i].cumsum(0)

    # Recall
    recall = tpc / (n_l + eps)  # recall curve
    #print(recall[:,0])
    r[ci] = np.interp(-px, -conf[i].astype(np.float64), recall[:, 0], left=0)  # negative x, xp because xp decreases

    # Precision
    precision = tpc / (tpc + fpc)  # precision curve
    p[ci] = np.interp(-px, -conf[i].astype(np.float64), precision[:, 0], left=1)  # p at pr_score

    # AP from recall-precision curve
    for j in range(tp.shape[1]):
        ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
        if plot and j == 0:
            py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
   
# Compute F1 (harmonic mean of precision and recall)
f1 = 2 * p * r / (p + r + eps)

if plot:
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    plot_pr_curve1(px, py, ap, names)
    plot_mc_curve1(px, f1, names, ylabel='F1')
    plot_mc_curve1(px, p, names, ylabel='Precision')
    plot_mc_curve1(px, r, names, ylabel='Recall')



#%%

iou = bb_IoU(detections[[0,1,2,3,4],:4], labels[2,1:])

#%%
perf_names = ['F1', 'map50', 'map'] #F1-score, mAP@0.5, mAP@0.4:0.95
table_ratio = []
perf = {x: [] for x in perf_names}

for i in range(10):
    perf["F1"].append(i)
    
#%%
dataset_name = 'datasets\\dataset155' 
folder_type = 'test'
yolo_path = 'yolov5'

model, im_paths, label_paths, name1 = load_dataset(dataset_name, folder_type, yolo_path)
#%%

path1 = os.path.join('ting til seb','sebim3.png')
# path2 = os.path.join('ting til seb','sebim2.png')
det1 = get_detection(path1, model)
# det2 = get_detection(path2, model)

im1 = cv2.imread(path1)[:,:,::-1]
# im2 = cv2.imread(path2)[:,:,::-1]

plot_bboxes(im1, None, det1, im_idx=None, data_name=None)
# plot_bboxes(im2, None, det2, im_idx=None, data_name=None)
np.save('ting til seb/detection3', det1)
# np.save('ting til seb/detection2', det2)



