# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 15:31:38 2023

@author: jonas
"""
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

def bb_IoU(bboxA, bboxB):
    """
    Parameters
    ----------
    bboxA : Array(N, 4) (xmin,ymin,w,h)
        DESCRIPTION.
    bboxB : Array(M, 4) (xmin,ymin,w,h)
        DESCRIPTION.

    Returns
    -------
    iou : Array(N,N) (0-1 if correct)

    """
    # If N or M is one
    if bboxA.ndim == 1: bboxA = np.array([bboxA])
    if bboxB.ndim == 1: bboxB = np.array([bboxB])
    
    
    N = bboxA.shape[0]
    M = bboxB.shape[0]
    
    xminA,yminA,wA,hA = [bboxA[:,i] for i in range(4)]
    xminB,yminB,wB,hB = [bboxB[:,i] for i in range(4)]
    
    iou = np.zeros((N,M))
    for i in range(N):
        area_A = wA[i]*hA[i]
        
        for j in range(M):
            xmin = max(xminA[i],xminB[j])
            ymin = max(yminA[i],yminB[j])
            xmax = min(xminA[i]+wA[i],xminB[j]+wB[j])
            ymax = min(yminA[i]+hA[i],yminB[j]+hB[j])
            
            interArea = max(0, xmax - xmin) * max(0, ymax - ymin)
            area_B = wB[j]*hB[j]

            iou[i,j] = interArea/(area_A + area_B - interArea)
    
    
    #unionArea = area_A + area_B - interArea
    
    #print(interArea, unionArea)
    return iou 

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x, y, w, h, conf, class
        labels (array[M, 5]), class, x, y, w, h
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    # iouv = torch.linspace(0.5, 0.95, 10, device=device)
    # iouv = np.linspace(0.5,0.95, 10)
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool) # init
    iou = bb_IoU(labels[:, 1:], detections[:, :4]) # Calculate pair-wise iou
    
    # Bool matrix indicating if det[i] and lab[j] classes are the same
    correct_class = labels[:, 0:1] == detections[:, 5] 
    
    for i in range(len(iouv)):
        x = np.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        
        if x[0].shape[0]: # Check if there is at least one match
            matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [label, detect, iou]
            
            if x[0].shape[0] > 1: # If multiple matches
                matches = matches[matches[:, 2].argsort()[::-1]] # Sort descending iou score
                
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]] # Remove duplicate detections
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]] # Remove duplicate ground truths
            # Set column i elements (detections) to true if IoU > iouv[i] and class match 
            # i.e. that detection is a tp
            correct[matches[:, 1].astype(int), i] = True
    return np.array(correct, dtype=bool)

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix='', data_name=None):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
        data_name: Name of the dataset
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
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
        # print(recall[:,0])
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
        plot_pr_curve(px, py, ap, names, data_name=data_name)
        plot_mc_curve(px, f1, names, ylabel='F1', data_name=data_name)
        plot_mc_curve(px, p, names, ylabel='Precision', data_name=data_name)
        plot_mc_curve(px, r, names, ylabel='Recall', data_name=data_name)
    
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

# Plots ----------------------------------------------------------------------------------------------------------------


def plot_pr_curve(px, py, ap, names=(), data_name=None):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Precision-Recall Curve')
    if data_name is not None:
        ax.set_title(f'Precision-Recall Curve, Dataset: {data_name}')
    #plt.close(fig)
    plt.show()

def plot_mc_curve(px, py, names=(), xlabel='Confidence', ylabel='Metric', data_name=None):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel}-Confidence Curve')
    if data_name is not None:
        ax.set_title(f'{ylabel}-Confidence Curve, Dataset: {data_name}')
    #fig.savefig(save_dir, dpi=250)
    plt.show()
    #plt.close(fig)
