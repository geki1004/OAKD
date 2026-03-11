import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Measurement:
    def __init__(self, num_classes:int, ignore_idx=None) :
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
    
    def _make_confusion_matrix(self, pred:np.ndarray, target:np.ndarray):

        assert pred.shape[0] == target.shape[0], "pred and target ndarray's batchsize must have same value"
        N = pred.shape[0]
        pred_label = pred.argmax(axis=1) # (N, H, W)
        
        pred_1d = np.reshape(pred_label, (N, -1)) # (N, HxW)
        target_1d = np.reshape(target, (N, -1)) # (N, HxW)
        cats = self.num_classes * target_1d + pred_1d # (N, HxW)
        conf_mat = np.apply_along_axis(lambda x: np.bincount(x, minlength=self.num_classes**2), axis=-1, arr=cats) # (N, 9)
        conf_mat = np.reshape(conf_mat, (N, self.num_classes, self.num_classes)) # (N, 3, 3)
        return conf_mat

    def no_bg_miou(self, conf_mat:np.ndarray):
        # 0 = bg, 1 = weed, 2 = crop

        sum_col = np.sum(conf_mat, -2)  # (N, 3), TP+FP
        sum_row = np.sum(conf_mat, -1)  # (N, 3), TP+FN

        miou_nbg = np.mean((conf_mat[:, 1, 1] + conf_mat[:, 2, 2]) / (sum_col[:, 1]+sum_row[:, 1]-conf_mat[:, 1, 1]+ sum_col[:, 2]+sum_row[:, 2]-conf_mat[:, 2, 2]+1e-8))
        miou_nbg = np.array(miou_nbg)
        miou_nbg = np.mean(miou_nbg)
        return miou_nbg

    def accuracy(self, pred, target):
        N = pred.shape[0]
        pred = pred.argmax(axis=1) # (N, H, W)
        pred = np.reshape(pred, (pred.shape[0], pred.shape[1]*pred.shape[2])) # (N, HxW)
        target = np.reshape(target, (target.shape[0], target.shape[1]*target.shape[2])) # (N, HxW)
        
        if self.ignore_idx != None:
             not_ignore_idxs = np.where(target != self.ignore_idx) # where target is not equal to ignore_idx
             pred = pred[not_ignore_idxs]
             target = target[not_ignore_idxs]
             
        return np.mean(np.sum(pred==target, axis=-1)/pred.shape[-1])
    
    def miou(self, conf_mat:np.ndarray):
        iou_list = []

        sum_col = np.sum(conf_mat, -2) # (N, 3)
        sum_row = np.sum(conf_mat, -1) # (N, 3)
        for i in range(self.num_classes):
            batch_mean_iou = np.mean(conf_mat[:, i, i] / (sum_col[:, i]+sum_row[:, i]-conf_mat[:, i, i]+1e-8))
            iou_list.append(batch_mean_iou)
        iou_ndarray = np.array(iou_list)

        miou = np.mean(iou_ndarray)
        return iou_list, miou
    
    def precision(self, conf_mat:np.ndarray):
        sum_col = np.sum(conf_mat, -2)# (N, num_classes) -> 0으로 예측, 1로 예측 2로 예측 각각 sum
        mprecision = np.mean(np.array([(conf_mat[:, 1, 1]+conf_mat[:, 2, 2])/ (sum_col[:, 1]+sum_col[:, 2]+1e-7)]), axis=-1)
        mprecision = np.mean(mprecision)
        return mprecision

    def recall(self, conf_mat:np.ndarray):
        sum_row = np.sum(conf_mat, -1)# (N, 3) -> 0으로 예측, 1로 예측 2로 예측 각각 sum
        mrecall = np.mean(np.array([(conf_mat[:, 1, 1]+conf_mat[:, 2, 2])/ (sum_row[:, 1]+sum_row[:, 2]+1e-7)]), axis=-1)
        mrecall = np.mean(mrecall)
        return mrecall
    
    def f1score(self, recall, precision):
        return 2*recall*precision/(recall + precision+1e-8)
    
    def measure(self, pred:np.ndarray, target:np.ndarray):
        conf_mat = self._make_confusion_matrix(pred, target)
        acc = self.accuracy(pred, target)
        iou_list, _ = self.miou(conf_mat)
        miou = self.no_bg_miou(conf_mat)
        precision = self.precision(conf_mat)
        recall = self.recall(conf_mat)
        f1score = self.f1score(recall, precision)
        return acc, miou, iou_list, precision, recall, f1score
        
    __call__ = measure