# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 22:34:21 2021

@author: ZhangWei
"""

import numpy as np
import pandas as pd
import os
import cv2

def mask_dice(prediction, groundtruth):
    intersection = np.logical_and(groundtruth, prediction)
    union = np.logical_or(groundtruth, prediction)
    if np.sum(union) + np.sum(intersection) == 0:
        return 1
    dice_score = 2*np.sum(intersection) / (np.sum(union) + np.sum(intersection))
    return dice_score

def norm_mask(mask):
    mask[mask<127.5] = 0
    mask[mask>=127.5] = 1
    return mask
def denorm_mask(mask):
    mask[mask<127.5] = 0
    mask[mask>=127.5] = 255
    return mask

def trans_by_flow(flow, img):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    trans_img = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return trans_img

def my_sort(strs):
    return int(strs.split('.')[0].split('-')[1])

path = 'E:\\optical_flow\\data\\LIDC-IDRI-slices\\'
output_path = 'E:\\optical_flow\\result\\plain-optical-flow\\result-image\\'
files = os.listdir(path)

Dice = []

for file in files:
    nodules = os.listdir(os.path.join(path, file))
    if not os.path.exists(os.path.join(output_path, file)):
        os.mkdir(os.path.join(output_path, file))
        
    for nodule in nodules:
        if not os.path.exists(os.path.join(output_path, file, nodule)):
            os.mkdir(os.path.join(output_path, file, nodule))
        if not os.path.exists(os.path.join(output_path, file, nodule, 'images')):
            os.mkdir(os.path.join(output_path, file, nodule, 'images'))
        if not os.path.exists(os.path.join(output_path, file, nodule, 'mask-0')):
            os.mkdir(os.path.join(output_path, file, nodule, 'mask-0'))
            
        images = os.listdir(os.path.join(path, file, nodule, 'images'))
        masks = os.listdir(os.path.join(path, file, nodule, 'mask-0'))
        images.sort(key = my_sort)
        masks.sort(key = my_sort)
        assert len(images) == len(masks), 'length of images and masks is invalid.'
        mid = len(images) // 2
        t_path = os.path.join(path, file, nodule)
        t_out_path = os.path.join(output_path, file, nodule)
        # forward and backward
        fab = [range(mid + 1, len(images)), range(mid - 1, -1, -1)]
        for direction_range in fab:
            img1 = cv2.cvtColor(cv2.imread(os.path.join(t_path, 'images', images[mid])), cv2.COLOR_BGR2GRAY)
            img1_mask = cv2.cvtColor(cv2.imread(os.path.join(t_path, 'mask-0', images[mid])), cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(t_out_path, 'images', 'slice-' + str(mid) + '-ori.png'), img1)
            cv2.imwrite(os.path.join(t_out_path, 'mask-0', 'slice-' + str(mid) + '-ori.png'), img1_mask)
            for idx in direction_range:
                img2 = cv2.cvtColor(cv2.imread(os.path.join(t_path, 'images', images[idx])), cv2.COLOR_BGR2GRAY)
                img2_mask = cv2.cvtColor(cv2.imread(os.path.join(t_path, 'mask-0', images[idx])), cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev=img1, next=img2, flow=None, pyr_scale=0.5, levels=5,
                                                    winsize=15,
                                                    iterations=3, poly_n=3, poly_sigma=1.2,
                                                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                trans_img2 = trans_by_flow(flow, img1)
                trans_img2_mask = trans_by_flow(flow, img1_mask)
                Dice.append(mask_dice(norm_mask(trans_img2_mask.copy()), norm_mask(img2_mask.copy())))
                cv2.imwrite(os.path.join(t_out_path, 'images', images[idx]), trans_img2)
                cv2.imwrite(os.path.join(t_out_path, 'mask-0', images[idx]), denorm_mask(trans_img2_mask.copy()))
                img1 = img2
                img1_mask = trans_img2_mask

print('[INFO] Dice: ', np.mean(Dice))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            