import torch
import numpy as np
import os
import cv2
import itertools
from model_seg_flow import Net
from torch.utils.data import DataLoader
from model_seg_flow import net_utils
import torch, gc
from model_seg_flow import FeaturePath
from model_seg_flow import FlowSegPath
from model_seg_flow import net_utils
from model_seg_flow.Net import OpticalSegFLow
import torch.nn as nn
try:
    from sklearn.externals import joblib
except:
    import joblib
torch.cuda.empty_cache()#清除PyTorch缓存。
torch.cuda.set_device(3)#设置使用GPU编号为0的设备。
torch.cuda.empty_cache()#清除PyTorch缓存。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net.OpticalSegFLow(device=device, num_channels=1, num_levels=6, use_cost_volume=True)#创建一个基于光流法和U-Net的肺部CT图像分割模型，其中OpticalSegFLow是一个自定义的类，实现了这个模型。
net.to(device)#将模型移动到GPU上进行训练或推理。
model_save_path = '/home/zhangwei/spatialaffinitynetwork/model.pth1'
#checkpoint = torch.load(model_save_path)
checkpoint = torch.load(model_save_path,  map_location='cuda:3')
epochs = checkpoint['epoch']
net.load_state_dict(checkpoint['seg_flow_net'])
ap = '/home/zhangwei/optical_flow/data/LIDC-IDRI-CHAOS/LIDC-IDRI-0001/nodule-0/images/'
pic = os.listdir(ap)
forward_imgs = []
backward_imgs = []
forward_warps = []
backward_warps = []
def get_data(pics):
    for i in range(len(pics)):
        if len(pics)%2 == 0 :
            mid_img = 'slice-'+ str(len(pics)//2)+'-ori.png'
            if i < len(pics)//2 :
                if i+1 == len(pics)//2:
                    forward_img = ('slice-'+ str(i)+'.png',mid_img)
                else:
                    forward_img = ('slice-'+ str(i)+'.png','slice-'+ str(i+1)+'.png')
                forward_imgs.append(forward_img)
            if i > len(pics)//2 :
                if i-1 == len(pics)//2:
                    backward_img = ('slice-'+ str(i)+'.png',mid_img)
                else:
                    backward_img = ('slice-'+ str(i)+'.png','slice-'+ str(i-1)+'.png')
                backward_imgs.append(backward_img)
        elif len(pics)%2 != 0 :
            mid_img = 'slice-'+ str(int(int(len(pics))/2-0.5))+'-ori.png'
            if i < int(int(len(pics))/2-0.5) :
                if i+1 == int(int(len(pics))/2-0.5):
                    forward_img = ('slice-'+ str(i)+'.png',mid_img)
                else:
                    forward_img = ('slice-'+ str(i)+'.png','slice-'+ str(i+1)+'.png')
                forward_imgs.append(forward_img)
            if i > int(int(len(pics))/2-0.5):
                if i-1 == int(int(len(pics))/2-0.5):
                    backward_img = ('slice-'+ str(i)+'.png',mid_img)
                else:
                    backward_img = ('slice-'+ str(i)+'.png','slice-'+ str(i-1)+'.png')
                backward_imgs.append(backward_img)
    print(backward_imgs)
    print(len(backward_imgs))
    return forward_imgs, backward_imgs
def process_mask(self, mask):
        mask[mask>=127.5] = 255
        mask[mask<127.5] = 0
        mask = mask
        return mask
def process_mask(mask):
    mask[mask>=127.5] = 255
    mask[mask<127.5] = 0
    mask = mask
    return mask
def pack_tensor(image, is_mask=False):
    path = ap + image
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = image / 255
    image = torch.unsqueeze(torch.from_numpy(image), 0).type(torch.FloatTensor)
    return image
def pack_tensor1(path, is_mask=False):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if is_mask:
        image = process_mask(image)
    image = image/ 255
    image = torch.unsqueeze(torch.from_numpy(image), 0).type(torch.FloatTensor)
    return image
def forward_1(imgs):
    for i in range(len(imgs)):
        img1,img2 = forward_imgs[i]
        image1 = pack_tensor(img1, is_mask=False) .to(device)
        image2 = pack_tensor(img2, is_mask=False) .to(device)
        image1 = image1.unsqueeze(0).to(device)
        image2 = image2.unsqueeze(0).to(device)
        #print(image1.shape)
        flows_f, segs_f, fp1, fp2, flows_b, segs_b = net(image2, image1, training=False)
        warp = net_utils.flow_to_warp(flows_f[0]) 
        forward_warps.append(warp)
    return image1,forward_warps
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    torch.cuda.set_device(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()#清除PyTorch缓存。
    with torch.no_grad():
        get_data(pic)
        forward_1(forward_imgs)
        coarse_map_1 = pack_tensor1('/home/zhangwei/optical_flow/result/plain-seg-flow/result-imageCHAOS/LIDC-IDRI-0001/nodule-0/mask-0/slice-23.png',is_mask= True)
        coarse_map_1 = coarse_map_1.unsqueeze(0).to(device)
        coarse_map_ = (coarse_map_1 + 1) * -1
        coarse_map_1 = torch.cat([coarse_map_, coarse_map_1], dim=1)
        if len(pic)%2 != 0 :
            for n in range (int(int(len(pic))/2-0.5)-int(23)):
                coarse_map_1 = net_utils.resample(coarse_map_1 , forward_warps[n+50]).to(device)
                #refined_coarse_map = net_utils.resample(refined_coarse_map , forward_warps[n]).to(device)
                print(n+50)
        elif len(pic)%2 == 0 :
            for n in range (int(int(len(pic))/2)-int(23)):
                coarse_map_1 = net_utils.resample(coarse_map_1 , forward_warps[n+23]).to(device)
                #refined_coarse_map = net_utils.resample(refined_coarse_map , forward_warps[n]).to(device)
                print(n+23)
        coarse_map_1_ = coarse_map_1.clone()
        coarse_map_1_ = coarse_map_1*255
        coarse_map_1_[coarse_map_1_>=127.5] = 255
        coarse_map_1_[coarse_map_1_< 127.5] = 0
        #print('1',refined_map_)
        #print(refined_map)
        coarse_map_1_ = coarse_map_1_.detach().cpu().numpy()
        cv2.imwrite('/home/zhangwei/optical_flow/coarse_map_-'+str(1)+'.png',coarse_map_1_[0,1] )