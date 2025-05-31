


import torch
import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from auxillary.yolo_code_that_we_use import *
from tqdm.notebook import tqdm
from auxillary.confusion_matrix_class import ConfusionMatrix

# from AUC
# duplicate later
# def get_detection(im_path, model, post_process=True, conf_thresh=0.4, iou_thresh=0.5, output_format='corner'):
#     im = Image.open(im_path)
#     results = model(im)
    
#     detection = results.pandas().xyxy[0].to_numpy()
    
#     xmin = detection[:,0]
#     ymin = detection[:,1]
#     xmax = detection[:,2]
#     ymax = detection[:,3]
#     w = xmax-xmin
#     h = ymax-ymin
    
#     if output_format == 'corner':
#         transformed_boxes = np.array([xmin,ymin,w,h]).T
#     elif output_format == 'center':
#         x_center = xmin + w/2
#         y_center = ymin + h/2
#         transformed_boxes = np.array([x_center, y_center, w, h]).T
#     else:
#         print(f"output_format must either be 'corner' or 'center'. Your input was {output_format}.")
#     detection[:, :4] = transformed_boxes
    
#     detection = detection[:,:-1].astype(np.float64)
    
#     if post_process:
#         detection = detection[detection[:, 4].argsort()[::-1]]  # Sort according to confidence

#         # Run non-max-suppresion
#         keep_list = np.arange(0, detection.shape[0])
#         remove_list = []
#         iou = bb_IoU(detection[:, :4], detection[:, :4]) - np.eye(detection.shape[0])
        
#         overlaps = np.where((iou >= iou_thresh))
#         overlaps = np.stack(overlaps, 1)
#         for idx1, idx2 in overlaps:
#             conf1 = detection[idx1,4]
#             conf2 = detection[idx2,4]
#             if conf1 <= conf2: remove_list.append(idx1)
#             else: remove_list.append(idx2)
        
#         remove_list = np.unique(remove_list)
#         keep_list = [k for k in keep_list if k not in remove_list]
        
#         detection = detection[keep_list]
        
#         # print(detection.shape[0])
#         # Keep 7 highest striped and solids 1 cue and black and 18 dots
#         for c, n in zip([0, 1, 2, 3, 4], [7, 7, 1, 1, 18]):
#             # print(detection[:, 5])
#             idxs = [i for i, x in enumerate(detection[:, 5]) if x == c]
#             keep = idxs[:n]
#             remove = idxs[n:]
            
#             for i in keep:
#                 detection[i, 4] = max(detection[i, 4], conf_thresh)
            
#             # print(c, len(idxs), len(keep), len(remove))
#             for j in remove:
#                 detection[j, 4] = 0.0
            
#         # Remove detections with conf_score less than threshhold
#         detection = detection[detection[:, 4] >= conf_thresh]
    
#     return detection

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

# duplicate later
# def helper_plot_bbox(ax, boxes, valid_classes, colors, kind="detection"):
#     alpha = (0.8,)
#     class_count = [0,0,0,0,0]
#     classes = []
#     for bb in boxes:
#         if kind == "detection":
#             cl = int(bb[5]) # class
#             if valid_classes[cl]:
#                 # print(cl)
#                 x,y,w,h = bb[:4]
#                 c = colors[cl] + alpha
#                 class_count[cl] += 1
#                 classes.append(cl)
#                 ax.add_patch(patches.Rectangle((x,y),w,h,linewidth=1,edgecolor=c,facecolor='none'))
            
#         elif kind == "label":
#             cl = int(bb[0]) # class
#             if valid_classes[cl]:
#                 x,y,w,h = bb[1:]
#                 c = colors[cl]
#                 class_count[cl] += 1
#                 classes.append(cl)
#                 ax.add_patch(patches.Rectangle((x,y),w,h,linewidth=1,edgecolor=c,facecolor='none'))
        
        
#     return classes, class_count

# duplicate later
# def plot_bboxes(im, label, detection, 
#                 im_idx=None, data_name=None, 
#                 classes_to_plot=['Striped', 'Solid', 'Cue', 'Black', 'Dot'], 
#                 title_names=['Ground Truth', 'Detection'], 
#                 legend=True, save_path=None, show_axes=False):
    
#     boxes = []
#     to_plot = []
#     artist = []
#     if label is not None:
#         boxes.append(label)
#         to_plot.append("label")
#         artist.append(mlines.Line2D([], [], color='k', linestyle='-', label=f'{title_names[0]}: {label.shape[0]}'))
    
#     if detection is not None:
#         boxes.append(detection)
#         to_plot.append("detection")
#         artist.append(mlines.Line2D([], [], color='k', linestyle='--', label=f'{title_names[1]}: {detection.shape[0]}'))
    
#     N_plots = len(to_plot)
#     if N_plots == 0:
#         return None

#     classes = ['Striped', 'Solid', 'Cue', 'Black', 'Dot']
#     valid_classes = [True if x in classes_to_plot else False for x in classes]
#     colors = [(0,0,1),(0,1,0),(1,0,0),(0,1,1),(1,0,1),(1,1,0)]
    
#     found_classes = []
#     found_classes_count = []
    
#     fig, axs = plt.subplots(1, N_plots, figsize=(10,5*N_plots), sharex=True, sharey=True)
    
#     for i, (ax, kind) in enumerate(zip(np.array(axs).reshape(-1), to_plot)):
#         ax.imshow(im)
#         if not show_axes: ax.axis('off')
#         cl, class_count = helper_plot_bbox(ax, boxes[i], valid_classes, colors, kind=kind)
#         found_classes.append(cl)
#         found_classes_count.append(class_count)
        
#     found_classes_count = np.array(found_classes_count).T
#     unique_classes = np.unique([x for fc in found_classes for x in fc])
#     if len(unique_classes) != 0:
#         for c in unique_classes:
#             c = int(c)
#             artist.append(patches.Patch(color=colors[c], label = classes[c] + f': {found_classes_count[c]}'))
#         # fig.legend(loc='upper center',handles=artist,bbox_to_anchor=(0.5,-0.05),
#                   # fancybox=True, ncol=len(artist))
#         if legend: fig.legend(loc='center', handles=artist, bbox_to_anchor=(1, 0.5), prop={"size": 10})
    
#     if (data_name is not None) or (im_idx is not None):
#         fig.suptitle(f'Bounding boxes for image {im_idx} in dataset {data_name}')
        
#     fig.tight_layout()
#     if save_path is not None:
#         fig.savefig(save_path, bbox_inches='tight', dpi=500)
    
#     plt.show()
#     return None

def compare_post_process(im, label, detection, detection_post, im_idx=None, data_name=None, title_names=['Ground Truth', 'No post processing', 'With post proccessing'], save_path=None):

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
    fig.tight_layout()
    if save_path is not None:
        print('saved to', save_path)
        fig.savefig(save_path, bbox_inches='tight', dpi=500)
    
    plt.show()
    plt.close()

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

def load_dataset(dataset_name, weight_path, folder_type='train'):
    """

    Parameters
    ----------
    dataset_name : str
        Path to the dataset. The dataset should include train, test 
        and val folders, as well as the model weight that was created from
        this dataset.
    folder_type : str, optional
        'train', 'val' or 'test'. The default is 'test'.

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
    for filename in os.listdir(dataset_name):
        if filename == folder_type:
            im_folder = dataset_name / folder_type / 'images'
            label_folder = dataset_name / folder_type / 'labels'
            # label_folder = os.path.join(dataset_name, folder_type, 'labels')
    
    im_paths = []
    for filename in sorted(os.listdir(im_folder), key=len):
        #print(os.path.join(im_folder, filename))
        # im_paths.append(os.path.join(im_folder, filename))
        im_paths.append(im_folder / filename)

    label_paths = []
    for filename in sorted(os.listdir(label_folder), key=len):
        #print(os.path.join(label_folder, filename))
        # label_paths.append(os.path.join(label_folder, filename))
        label_paths.append(label_folder / filename)
        
    # model = torch.hub.load(yolo_path, 'custom', path=weight_path, source='local')
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weight_path)  # local model
    model.eval()
    return model, im_paths, label_paths
    

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


# from 'main'

def load_detection_model(model_path):
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)  # local model
    model.eval()
    
    # torchhub messes with matplotlib
    # %matplotlib inline
    return model

def get_detection(im_path, model, post_process=True, conf_thresh=0.4, iou_thresh=0.5, output_format = 'center'):
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

        # Keep 7 highest striped and solids 1 cue and black and 18 dots
        for c, n in zip([0, 1, 2, 3, 4], [7, 7, 1, 1, 18]):
            idxs = [i for i, x in enumerate(detection[:, 5]) if x == c]
            keep = idxs[:n]
            remove = idxs[n:]

            for i in keep:
                detection[i, 4] = max(detection[i, 4], conf_thresh)  # We are at least 'conf_thresh' confident about these predictions
            
            for j in remove:
                detection[j, 4] = 0.0
            
        # Remove detections with conf_score less than threshhold
        detection = detection[detection[:, 4] >= conf_thresh]
    
    return cv2.imread(str(im_path))[:,:,::-1], detection

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
                ax.add_patch(patches.Rectangle((x-w/2,y-h/2),w,h,linewidth=1,edgecolor=c,facecolor='none'))
            
        elif kind == "label":
            cl = int(bb[0]) # class
            if valid_classes[cl]:
                x,y,w,h = bb[1:]
                c = colors[cl]
                class_count[cl] += 1
                classes.append(cl)
                ax.add_patch(patches.Rectangle((x-w/2,y-h/2),w,h,linewidth=1,edgecolor=c,facecolor='none'))
        
    return classes, class_count

def plot_bboxes(im, label, detection, 
                im_idx=None, 
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
    
    if (im_idx is not None):
        fig.suptitle(f'Bounding boxes for image {im_idx}')
        
    fig.tight_layout()
    if save_path is not None:
        print('saved to', save_path)
        fig.savefig(save_path, bbox_inches='tight', dpi=500)
    
    plt.show()
    plt.close()
