import torch
import numpy as np
import os
import cv2
import itertools
from model_seg_flow import Net
from torch.utils.data import DataLoader
from load_data import LIDC_IDRI_flow
from tqdm import tqdm
from model_seg_flow import net_utils
try:
    from sklearn.externals import joblib
except:
    import joblib

print('************** train plain-net-flow **************')

torch.cuda.set_device(1)

BATCH_SIZE = 1
epochs = 0
NUM_EPOCH = 200
continue_training = False
#root = './data/LIDC-IDRI-slices/'
abpath = os.getcwd()
root = os.path.join(abpath, 'data/LIDC-IDRI-slices/')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = LIDC_IDRI_flow(root=root, load=True, training=True)
test_dataset = LIDC_IDRI_flow(root=root, load=True, training=False)
net = Net.OpticalSegFLow(device=device, num_channels=1, num_levels=6, use_cost_volume=True)
net.to(device)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

optimizer = torch.optim.Adam(list(net._pyramid.parameters()) + list(net._flow_model.parameters()), lr=1e-4)

test_set = joblib.load(os.path.join(abpath, 'save/immediate_data/test_set.pkl')) 
output_path = os.path.join(abpath, 'result/plain-seg-flow/result-image/')
save_test_result_ids = list(range(10))

model_save_path = os.path.join(abpath, 'save/checkpoint/plain-seg-flow/model.pth')

train_loss_history = []
test_loss_history = []
test_dice_history = []

if continue_training:
    print('Continuing with model from ' + model_save_path)

    checkpoint = torch.load(model_save_path)
    
    epochs = checkpoint['epoch']
    net.load_state_dict(checkpoint['seg_flow_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    train_loss_history = checkpoint['train_loss_history']
    test_loss_history = checkpoint['test_loss_history']
    test_dice_history = checkpoint['test_dice_history']

def train():
    try:
        for epoch in range(epochs, NUM_EPOCH):
            net.train()
            losses = []
            bar = tqdm(train_loader, ncols=100)
            for step, (img1, img2, mask1, mask2) in enumerate(bar):
                bar.set_description('Epoch %i' % epoch)
                img1 = img1.to(device)
                img2 = img2.to(device)
                mask1 = mask1.to(device)
                mask2 = mask2.to(device)
                
                flows_f, segs_f, fp1, fp2, flows_b, segs_b = net(img1, img2, training=True)
                loss = net.compute_loss(img1, img2, fp1, fp2, flows_f, segs_f, mask1, mask2,
                                        flows_b, segs_b)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                bar.set_postfix(loss=loss.item())
            loss = np.mean(losses)
            train_loss_history.append(loss)
            
            if epoch % 2 == 0:
                photo_loss, dice = test()
                test_loss_history.append(photo_loss)
                test_dice_history.append(dice)
                print('mean loss: ', photo_loss)
                print('mean dice: ', dice)
            
            print("Epoch: " + str(epoch) + ", Saving model to " + model_save_path)
            torch.save({
                'epoch': epoch,
                'seg_flow_net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss_history': train_loss_history,
                'test_loss_history': test_loss_history,
                'test_dice_history': test_dice_history
            }, model_save_path)
            
    except KeyboardInterrupt:
        pass
    except:
        raise

def test():
    with torch.no_grad():
        net.eval()

        photo_loss = []
        dice = []
        bar = tqdm(test_loader)
        for step, (backward_images, backward_masks, forward_images, forward_masks) in enumerate(bar):
            bar.set_description('Test')
            # backward and forward
            images_warp = [[], []]
            masks_warp = [[], []]
            flows_vis = [[], []]
            images = [backward_images, forward_images]
            masks = [backward_masks, forward_masks]
            for idx in range(len(images)):
                B, slices, H, W = images[idx].shape
                if slices < 2:
                    continue
                img1 = images[idx][:, 0:1, :, :].to(device)
                mask1 = masks[idx][:, 0:1, :, :].to(device)
                images_warp[idx].append(img1)
                masks_warp[idx].append(mask1)
                for i in range(slices - 1):
                    img1 = images[idx][:, i:(i+1), :, :].to(device)
                    img2 = images[idx][:, (i+1):(i+2), :, :].to(device)
                
                    mask2 = masks[idx][:, (i+1):(i+2), :, :].to(device)
                    flows_f, segs_f, fp2, fp1, flows_b, segs_b = net(img2, img1 , training=False)
                    warp = net_utils.flow_to_warp(flows_f[0])
                    warped_img1 = net_utils.resample(img1, warp)
                    warped_mask1 = net_utils.resample(mask1, warp)
                
                    loss = net_utils.compute_l1_loss(img2, warped_img1, flows_f, use_mag_loss=False)
                    photo_loss.append(loss.item())
                
                    dice.append(net_utils.mask_dice(
                            net_utils.norm_mask(warped_mask1.clone()),
                            mask2
                        ))
                    images_warp[idx].append(warped_img1)
                    masks_warp[idx].append(warped_mask1)
                    flows_vis[idx].append(flows_f[0])
                
                    mask1 = warped_mask1
            # ori_iamges
            #images_warp = [[backward_images[:, i:i+1, :, :] for i in range(backward_images.shape[1])],
            #               [forward_images[:, i:i+1, :, :] for i in range(forward_images.shape[1])]]
            if step in save_test_result_ids:
                save_test_result(step, images_warp, masks_warp, flows_vis)

        return np.mean(photo_loss), np.mean(dice)

def save_test_result(step, images_warp, masks_warp, flows_vis):
    path = test_set[step]
    LIDC_ID, nodule_id = path.split('/')[-2:]
    path = os.path.join(abpath, 'data', 'LIDC-IDRI-slices', LIDC_ID, nodule_id)
    slices = os.listdir(os.path.join(path, 'images'))
    mid = len(slices) // 2
    
    if not os.path.exists(os.path.join(output_path, LIDC_ID)):
        os.mkdir(os.path.join(output_path, LIDC_ID))
    if not os.path.exists(os.path.join(output_path, LIDC_ID, nodule_id)):
        os.mkdir(os.path.join(output_path, LIDC_ID, nodule_id))
    path = os.path.join(output_path, LIDC_ID, nodule_id)
    if not os.path.exists(os.path.join(path, 'images')):
        os.mkdir(os.path.join(path, 'images'))
    if not os.path.exists(os.path.join(path, 'mask-0')):
        os.mkdir(os.path.join(path, 'mask-0'))
    if not os.path.exists(os.path.join(path, 'flows')):
        os.mkdir(os.path.join(path, 'flows'))
    for idx in range(len(images_warp)):
        if len(images_warp[idx]) < 2:
            continue
        if idx == 0:
            # backward
            appendix = list(range(mid, -1, -1))
        else:
            # forward
            appendix = list(range(mid, len(slices), 1))
        assert len(images_warp[idx]) == len(appendix)
        assert len(masks_warp[idx]) == len(appendix)
        for i in range(len(appendix)):
            ap = appendix[i]
            image = torch.squeeze(images_warp[idx][i]).cpu().numpy() * 255
            filename = 'slice-' + str(ap) + '.png' if ap != mid else 'slice-' + str(ap) + '-ori.png'
            cv2.imwrite(os.path.join(path, 'images', filename), image)

            mask = net_utils.denorm_mask(torch.squeeze(masks_warp[idx][i]).cpu().numpy())
            filename = 'slice-' + str(ap) + '.png' if ap != mid else 'slice-' + str(ap) + '-ori.png'
            cv2.imwrite(os.path.join(path, 'mask-0', filename), mask)
            
            if i < len(flows_vis[idx]):
                flow = torch.squeeze(flows_vis[idx][i]).cpu().numpy()
                flow_color = flow_vis_fn(np.transpose(flow, axes=(1, 2, 0)))
                filename = 'flow-' + str(appendix[i]) + '-' + str(appendix[i + 1]) + '.png'
                cv2.imwrite(os.path.join(path, 'flows', filename), flow_color)
            
def flow_vis_fn(flow):
    hsv = np.zeros((128, 128, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_color

if __name__ == '__main__':
    train()
    
    
    
    
    
    
    
    
