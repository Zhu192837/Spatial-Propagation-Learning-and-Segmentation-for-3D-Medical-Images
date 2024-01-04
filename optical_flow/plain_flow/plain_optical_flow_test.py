import numpy as np
import pandas as pd
import os
import cv2
from sklearn.externals import joblib

ap = os.getcwd()
test_set = joblib.load(os.path.join(ap, 'save/immediate_data/test_set.pkl'))
output_path = os.path.join(ap, 'result/plain-optical-flow/result-image/')
save_test_result_ids = list(range(10))

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

Dice = []

for i in range(len(test_set)):
    path = test_set[i]
    images = os.listdir(os.path.join(path, 'images'))
    masks = os.listdir(os.path.join(path, 'mask-0'))
    images.sort(key = my_sort)
    masks.sort(key = my_sort)
    assert len(images) == len(masks), 'length of images and masks is invalid.'
    mid = len(images) // 2
    if i in save_test_result_ids:
        patient_id, nodule_id = path.split('/')[-2:]
        if not os.path.exists(os.path.join(output_path, patient_id)):
            os.mkdir(os.path.join(output_path, patient_id))
        if not os.path.exists(os.path.join(output_path, patient_id, nodule_id)):
            os.mkdir(os.path.join(output_path, patient_id, nodule_id))
            
        t_out_path = os.path.join(output_path, patient_id, nodule_id)
        if not os.path.exists(os.path.join(t_out_path, 'images')):
            os.mkdir(os.path.join(t_out_path, 'images'))
        if not os.path.exists(os.path.join(t_out_path, 'mask-0')):
            os.mkdir(os.path.join(t_out_path, 'mask-0'))
            
    # forward and backward
    fab = [range(mid + 1, len(images)), range(mid - 1, -1, -1)]
    for direction_range in fab:
        img1 = cv2.cvtColor(cv2.imread(os.path.join(path, 'images', images[mid])), cv2.COLOR_BGR2GRAY)
        img1_mask = cv2.cvtColor(cv2.imread(os.path.join(path, 'mask-0', images[mid])), cv2.COLOR_BGR2GRAY)
        if i in save_test_result_ids:
            cv2.imwrite(os.path.join(t_out_path, 'images', 'slice-' + str(mid) + '-ori.png'), img1)
            cv2.imwrite(os.path.join(t_out_path, 'mask-0', 'slice-' + str(mid) + '-ori.png'), img1_mask)
        for idx in direction_range:
            img2 = cv2.cvtColor(cv2.imread(os.path.join(path, 'images', images[idx])), cv2.COLOR_BGR2GRAY)
            img2_mask = cv2.cvtColor(cv2.imread(os.path.join(path, 'mask-0', images[idx])), cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev=img1, next=img2, flow=None, pyr_scale=0.5, levels=5,
                                                winsize=15,
                                                iterations=3, poly_n=3, poly_sigma=1.2,
                                                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            trans_img2 = trans_by_flow(flow, img1)
            trans_img2_mask = trans_by_flow(flow, img1_mask)
            Dice.append(mask_dice(norm_mask(trans_img2_mask.copy()), norm_mask(img2_mask.copy())))
            if i in save_test_result_ids:
                cv2.imwrite(os.path.join(t_out_path, 'images', images[idx]), trans_img2)
                cv2.imwrite(os.path.join(t_out_path, 'mask-0', images[idx]), denorm_mask(trans_img2_mask.copy()))
            img1 = img2
            img1_mask = trans_img2_mask
        
print('[INFO] Dice: ', np.mean(Dice))
                
                
                
                
                
                
                
                
                
                
                
                
                