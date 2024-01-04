from model_seg_flow import FeaturePath
from model_seg_flow import FlowSegPath
from model_seg_flow import net_utils
import torch.nn as nn
class OpticalSegFLow(nn.Module):
    def __init__(self, device, num_channels=1, num_levels = 6, use_cost_volume=True):
        super(OpticalSegFLow, self).__init__()
        
        self.device = device

        self._pyramid = FeaturePath.FeaturePath(device=device, in_channels_img=num_channels)
        self._flow_model = FlowSegPath.FlowSegPath(device = device, num_levels = num_levels, 
                                                   seg_num_levels=4,
                                               num_channels_upsampled_context=32,
                                               use_cost_volume=use_cost_volume, use_feature_warp=True)

    def forward(self, img1, img2, training=True):
        """Compute the flow between image pairs
        :param img1: [BCHW] image 1 batch
        :param img2: [BCHW] image 2 batch
        :return: [B2HW] flow tensor for each image pair in batch, feature pyramid list for img pairs
        """
        fp1 = self._pyramid(img1)
        fp2 = self._pyramid(img2)
        flow_forward, seg_forward = self._flow_model(fp1, fp2, training)
        flow_backward, seg_backward = self._flow_model(fp2, fp1, training)
        #occlusions = net_utils.occlusion_mask(flow_forward, flow_backward, range(len(flow_forward)))
        return flow_forward, seg_forward, fp1, fp2, flow_backward, seg_backward

    def compute_loss(self, img1, img2, features1, features2, flows_f, segs_f, mask1, mask2,flows_b=None, segs_b=None):
    #这是一个Python函数的定义，其名称为compute_loss，接受多个输入参数，包括两个图像img1和img2，以及一些特征、光流、分割和掩码列表。
    # 其中flows_b和segs_b是可选参数，默认为None。
        f1 = [img1] + features1
        f2 = [img2] + features2
    #将img1和img2与它们各自的特征拼接在一起，形成两个新的列表f1和f2。
        warps = [net_utils.flow_to_warp(f) for f in flows_f]
        warped_f2 = [net_utils.resample(f, w) for (f, w) in zip(f2, warps)]
    #根据flows_f计算每个特征层级的正向光流，并使用该光流将f2中的特征重新采样到f1的空间中。
        #compute_level = range(len(flows))
        compute_level = [0]
    #设置要计算损失的特征层级，这里只计算最顶层的特征层级。
        loss_feature_consistency = net_utils.loss_feature_consistency(f1, warped_f2, flows_f,
                                                                      compute_level)
    #计算特征一致性损失，用于衡量在img1和img2之间进行特征对齐时的误差。
        loss_smoothness = net_utils.loss_smoothness(img1, flows_f, compute_level)
    #计算光流平滑性损失，用于鼓励光流场具有连续性和平滑性。
        loss_ssim_weight = net_utils.loss_ssim_weight(img1, img2, flows_f, compute_level,
                                                      self.device)
    #计算结构相似性损失，以提高图像重建的质量。
        loss_segment = net_utils.loss_segmentation(mask1, mask2, segs_f, compute_level, self.device)
    #计算分割损失，用于衡量分割结果的正确性。
        #loss = net_utils.compute_all_loss(f1, warped_f2, flows, self.device)
        loss = 10 * loss_feature_consistency + loss_smoothness + loss_ssim_weight + \
                10 * loss_segment
    #将特征一致性损失和分割损失乘以10，然后将所有的损失加权相加得到总体损失。
        if flows_b is not None:
            loss_seg_consistency = net_utils.loss_seg_consistency(mask1, mask2, flows_f, flows_b,
                                                                  compute_level, self.device)
            loss += loss_seg_consistency
    #如果提供了反向光流和分割列表，则计算分
        return loss
#这段代码是model_seg_flow模块中OpticalSegFLow类的一部分。
#OpticalSegFLow类是一个神经网络模型，用于进行基于光流的图像分割。
#__init__方法定义了类的初始化，它接受四个参数：device、num_channels、num_levels和use_cost_volume。
# 其中，device指定了计算设备，num_channels指定输入图像的通道数，num_levels指定光流金字塔的级别数，use_cost_volume指定是否使用成本体积。
#类中的两个属性_pyramid和_flow_model分别是FeaturePath.FeaturePath和FlowSegPath.FlowSegPath类的实例。
#forward方法接收两个输入图像img1和img2，然后在_pyramid和_flow_model中进行特征提取和光流计算，最终返回光流向前和向后的结果以及特征金字塔列表。
#compute_loss方法计算了两个输入图像之间的光流向前和向后，然后计算了特征的一致性、光流的平滑性、相似度权重和分割损失。如果有向后的光流，则计算光流的一致性损失。
#最终将这些损失加权求和，返回总的损失值。
#这段代码是一个用于深度学习的图像处理程序，主要是用于计算两个图像之间的光流，同时计算一些损失函数来训练模型。
#具体来说，函数forward输入两张图片img1和img2，并且返回它们之间的光流flow_forward，以及这两张图片的特征金字塔列表fp1和fp2。
# 函数使用了一个神经网络模型来计算光流，其中先对两张图片进行特征金字塔计算，再将这些特征金字塔作为输入传递给模型来计算光流。
#函数compute_loss用于计算损失函数，输入两张图片img1和img2，以及一些计算过程中需要用到的特征，如特征金字塔列表features1和features2，以及计算得到的光流flows_f和分割结果segs_f。
# 该函数的目标是计算一些损失函数，如特征一致性损失、光流平滑度损失、结构相似性损失等，用于训练模型。
# 此外，该函数还可接受反向计算得到的光流flows_b和分割结果segs_b，以计算分割结果一致性损失。函数最终返回总的损失值。
#这段代码定义了一个名为 OpticalSegFLow 的类，继承自 nn.Module。该类包含一个构造函数 __init__ 和两个方法 forward 和 compute_loss。
#构造函数 __init__ 接收四个参数：device 表示设备，num_channels 表示输入图片的通道数，默认为 1，num_levels 表示 feature pyramid 的层数，默认为 6，use_cost_volume 表示是否使用 cost volume，默认为 True。构造函数中的第一行调用了父类的构造函数。
#接下来，构造函数初始化了两个模型：FeaturePath.FeaturePath 和 FlowSegPath.FlowSegPath。
# 这两个模型都是自定义模型，FeaturePath.FeaturePath 是提取 feature pyramid 的模型，FlowSegPath.FlowSegPath 是计算光流和语义分割的模型。
#forward 方法接收两个参数 img1 和 img2，表示两张输入的图片。在该方法中，首先使用 self._pyramid 模型提取两张图片的 feature pyramid，然后使用 self._flow_model 模型计算两张图片的光流和语义分割结果。
#最后返回正向光流、正向语义分割结果、第一张图片的 feature pyramid、第二张图片的 feature pyramid、反向光流和反向语义分割结果。
#compute_loss 方法接收多个参数，包括两张输入图片 img1 和 img2，
#以及 forward 方法计算出的正向光流、正向语义分割结果、第一张图片的 feature pyramid、第二张图片的 feature pyramid、反向光流和反向语义分割结果。
# 此方法使用 net_utils 模块中定义的一系列函数计算损失值，包括特征一致性损失、平滑性损失、SSIM 权重损失和分割损失等，并返回总损失值。如果 flows_b 不为 None，则还会计算分割一致性损失，并将其加入到总损失中。      