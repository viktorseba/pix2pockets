# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:03:46 2023

@author: jonas
"""
import numpy as np
import matplotlib.pyplot as plt

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

class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.4, iou_thres=0.4):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres
        self.tps = [[] for _ in range(nc)]
        self.metrics = {"precision": [],
                        "recall": [],
                        "f1": [],
                        "AP": [],
                        "mAP": None}

    # def process_batch(self, detections, labels):
    #     """
    #     Return intersection-over-union (Jaccard index) of boxes.
    #     Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    #     Arguments:
    #         detections (Array[N, 6]), x1, y1, x2, y2, conf, class
    #         labels (Array[M, 5]), class, x1, y1, x2, y2
    #     Returns:
    #         None, updates confusion matrix accordingly
    #     """
    #     if detections is None:
    #         gt_classes = labels[:,0].astype(int)
    #         for gc in gt_classes:
    #             self.matrix[self.nc, gc] += 1  # background FN
    #         return

    #     detections = detections[detections[:, 4] >= self.conf]
    #     gt_classes = labels[:, 0].astype(int)
    #     detection_classes = detections[:, 5].astype(int)
    #     iou = bb_IoU(labels[:, 1:], detections[:, :4])

    #     x = np.where(iou >= self.iou_thres)
    #     if x[0].shape[0]:
    #         matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
    #         # return matches
    #         if x[0].shape[0] > 1:
    #             matches = matches[matches[:, 2].argsort()[::-1]]
    #             matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
    #             matches = matches[matches[:, 2].argsort()[::-1]]
    #             matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    #     else:
    #         matches = np.zeros((0, 3))

    #     n = matches.shape[0] > 0
    #     m0, m1, _ = matches.transpose().astype(int)
    #     # print("m0")
    #     # print(m0)
    #     # print("m1")
    #     # print(m1)
    #     for i, gc in enumerate(gt_classes):
    #         j = m0 == i
    #         if n and sum(j) == 1:
    #             self.matrix[detection_classes[m1[j]][0], gc] += 1  # correct
    #         else:
    #             self.matrix[self.nc, gc] += 1  # true background

    #     if n:
    #         for i, dc in enumerate(detection_classes):
    #             if not any(m1 == i):
    #                 self.matrix[dc, self.nc] += 1  # predicted background
    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels[:,0].astype(int)
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        # detections = detections[detections[:, 4] >= self.conf]
        gt_classes = labels[:, 0].astype(int)
        detection_classes = detections[:, 5].astype(int)
        detection_confidence = detections[:, 4]
        iou = bb_IoU(labels[:, 1:], detections[:, :4])

        x = np.where(iou >= self.iou_thres)
        if x[0].shape[0]:
            # matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None], detections[:, 4]), 1)
            matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None], detections[x[1], 4][:, None]), 1)
            # return matches
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 3].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                matches = matches[matches[:, 3].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _, _ = matches.transpose().astype(int)
        # print("m0")
        # print(m0)
        # print("m1")
        # print(m1)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                dc = detection_classes[m1[j]][0]
                d_conf = detection_confidence[m1[j]]
                # print(d_conf)
                self.matrix[dc, gc] += 1  # correct
                if dc == gc: self.tps[dc].append([True, d_conf[0]])
                else: self.tps[dc].append([False, d_conf[0]])
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background
                    self.tps[dc].append([False, detection_confidence[i]])

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        fn = self.matrix.sum(0) - tp
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1], fn[:-1]  # remove background class

    def calc_ap(self, eps=1e-8):
        n_l = self.matrix.sum(0)[:-1]  # Number of labels
        x = np.linspace(0, 1, 11)
        aps = []
        print(f"Number of labels:\n {n_l}")
        for c, tps in enumerate(self.tps):
            print(f"True positives:\n {tps}")
            if len(tps) < 1:
                aps.append(0)
                continue
            tps = np.array(tps)
            tps = tps[tps[:, 1].argsort()[::-1]]
            
            fpc = (1 - tps[:, 0]).cumsum(0)
            tpc = tps[:, 0].cumsum(0)
            
            # Recall
            recall = tpc / (n_l[c] + eps)  # recall curve

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            precision = [np.max(precision[i:]) for i in range(len(precision))]  # smoothed precision
            
            mrec = np.concatenate(([-0.01], recall, [1.01]))
            # mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([1.0], precision, [0.0]))

            aps.append(np.mean(np.interp(x, mrec, mpre)))
        
        return np.array(aps)
    
    def calc_metrics_pr_class(self, ndigits=3, eps=1e-8):
        tp, fp, fn = self.tp_fp()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        
        aps = self.calc_ap()
        # aps = np.array([x for x in aps if x is not None])
        mAP = np.mean(aps)
        self.metrics["precision"] = precision.round(ndigits)
        self.metrics["recall"] = recall.round(ndigits)
        self.metrics["f1"] = f1.round(ndigits)
        self.metrics["AP"] = aps.round(ndigits)
        self.metrics["mAP"] = round(mAP, ndigits)
        return self.metrics

    def plot(self, normalize=True, names=(), conf_thresh=None, im_id=None, save_path=None):
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00) 
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=3.2 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['bg']) if labels else 'auto'
        H = sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           'size': 30},
                       cmap='Greens',
                       # fmt='.1%',
                       fmt='.1f',
                       cbar=False,
                       square=False,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        # if normalize:
        #     for t in H.texts:
        #         t.set_text(t.get_text() + '%')
        ax.set_xlabel('True', fontsize=35)
        ax.set_ylabel('Predicted', fontsize=35)
        # ax.set_title('Confusion Matrix', fontsize=20)
        # cax = H.figure.axes[-1]
        # cax.tick_params(labelsize=20)
        # title_str = 'Confusion Matrix'
        # if conf_thresh is not None:
        #     # ax.set_title(f'Confusion Matrix, conf_thresh = {conf_thresh}')
        #     title_str += f', conf_thresh = {conf_thresh}'
        # if im_id is not None:
        #     im_id = str(im_id)
        #     title_str += f', im_id = {im_id}'
        
        # ax.set_title(title_str)
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight', dpi=500)
            
        plt.show()

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))
