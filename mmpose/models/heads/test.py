# # Copyright (c) OpenMMLab. All rights reserved.
# import torch
# import torch.nn as nn
# from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
#                       constant_init, normal_init)
# from mmpose.core.evaluation import pose_pck_accuracy,keypoint_epe
# from mmpose.core.post_processing import flip_back
# from mmpose.models.builder import build_loss
# from mmpose.models.utils.ops import resize
# from ..builder import HEADS
# from .topdown_heatmap_base_head import TopdownHeatmapBaseHead
# import torch.nn.functional as F

# class myConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1):
#         super(myConv2d, self).__init__()
#         padding = (kernel_size-1)//2
#         self.conv = nn.Conv2d(in_channels, out_channels,
#                               kernel_size=kernel_size, padding=padding)

#     def forward(self, x):
#         return self.conv(x)


# class dilatedConv(nn.Module):
#     ''' stride == 1 '''

#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
#         super(dilatedConv, self).__init__()
#         # f = (kernel_size-1) * d +1
#         # new_width = (width - f + 2 * padding)/stride + stride
#         padding = (kernel_size-1) * dilation // 2
#         self.conv = nn.Conv2d(in_channels, out_channels,
#                               kernel_size, dilation=dilation, padding=padding)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))


# class globalNet(nn.Module):
#     def __init__(self, in_channel, out_channel, mid_channel=128,scale_factor=0.25, kernel_size=3, dilations=None):
#         super(globalNet, self).__init__()
#         self.scale_factor = scale_factor
#         mid_channel = mid_channel
#         if dilations is None:
#             dilations = [1, 2, 5]
#         self.in_conv = myConv2d(in_channel, mid_channel//2,
#                               kernel_size=3)
#         self.in_local = myConv2d(out_channel, (mid_channel+1)//2,
#                               kernel_size=3)
#         self.out_conv = myConv2d(mid_channel, out_channel,
#                               kernel_size=3) 
#         convs = [dilatedConv(mid_channel, mid_channel,
#                                  kernel_size, dilation) for dilation in dilations]
#         self.convs = nn.Sequential(*convs)

#     def forward(self, x, local_feature):
#         # size = x.size()[2:]
#         # sf = self.scale_factor
#         # x = F.interpolate(x, scale_factor=sf)
#         # local_feature = F.interpolate(local_feature, scale_factor=sf)
#         # x = self.in_conv(x)
#         # local_feature = self.in_local(local_feature)
#         # fuse = torch.cat((x, local_feature), dim=1)
#         # x = self.convs(fuse)
#         # x = self.out_conv(x)
#         # x = F.interpolate(x, size=size)

#         size = x.size()[2:]
#         sf = self.scale_factor
#         x = F.interpolate(x, scale_factor=sf)
#         local_feature = F.interpolate(local_feature, scale_factor=sf/2)
#         x = self.in_conv(x)
#         local_feature = self.in_local(local_feature)
#         fuse = torch.cat((x, local_feature), dim=1)
#         x = self.convs(fuse)
#         x = self.out_conv(x)
#         x = F.interpolate(x, size=size)
#         return torch.sigmoid(x)
    

# @HEADS.register_module()
# class TopdownSimple_Heatmap_ConvHead_global(TopdownHeatmapBaseHead):
#     """Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple
#     Baselines for Human Pose Estimation and Tracking``.

#     TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
#     and a simple conv2d layer.

#     Args:
#         in_channels (int): Number of input channels
#         out_channels (int): Number of output channels
#         num_deconv_layers (int): Number of deconv layers.
#             num_deconv_layers should >= 0. Note that 0 means
#             no deconv layers.
#         num_deconv_filters (list|tuple): Number of filters.
#             If num_deconv_layers > 0, the length of
#         num_deconv_kernels (list|tuple): Kernel sizes.
#         in_index (int|Sequence[int]): Input feature index. Default: 0
#         input_transform (str|None): Transformation type of input features.
#             Options: 'resize_concat', 'multiple_select', None.
#             Default: None.

#             - 'resize_concat': Multiple feature maps will be resized to the
#                 same size as the first one and then concat together.
#                 Usually used in FCN head of HRNet.
#             - 'multiple_select': Multiple feature maps will be bundle into
#                 a list and passed into decode head.
#             - None: Only one select feature map is allowed.
#         align_corners (bool): align_corners argument of F.interpolate.
#             Default: False.
#         loss_keypoint (dict): Config for keypoint loss. Default: None.
#     """

#     def __init__(self,
#                  in_channels,
#                  feat_channels,
#                  out_channels,

#                  loss_keypoint=None,
#                  train_cfg=None,
#                  test_cfg=None):
#         super().__init__()

#         # self.globalNet = globalNet(in_channels, out_channels, feat_channels, scale_factor=0.5, kernel_size=3, dilations=None)
#          #in : x:input_channel 3  output_channel:19 x->H/2,W/2  feat_channels:128
#         self.w1_conv = nn.Sequential( 
#             nn.Conv2d(3, 32, kernel_size=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True))
#         #256->64
#         self.w2_conv = nn.Sequential( 
#             nn.Conv2d(256, 64, kernel_size=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True))
        
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
#         self.in_channels = in_channels
#         self.loss = build_loss(loss_keypoint)
#         self.train_cfg = {} if train_cfg is None else train_cfg
#         self.test_cfg = {} if test_cfg is None else test_cfg
#         self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')
#         # keypointhead 
#         self.layer = nn.Sequential(
#             nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(feat_channels, out_channels, kernel_size=1))


#     def get_loss(self, output, target, target_weight):
#         """Calculate top-down keypoint loss.
#         Note:
#             - batch_size: N
#             - num_keypoints: K
#             - heatmaps height: H
#             - heatmaps weight: W
#         Args:
#             output (torch.Tensor[N,K,H,W]): Output heatmaps.
#             target (torch.Tensor[N,K,H,W]): Target heatmaps.
#             target_weight (torch.Tensor[N,K,1]):
#                 Weights across different joint types.
#         """

#         losses = dict()
#         assert not isinstance(self.loss, nn.Sequential)
#         losses['Multiresolution_loss_g'] = self.loss(output, target, target_weight)
#         return losses

#     def get_accuracy(self, output, target, target_weight, img, img_metas):
#         """Calculate accuracy for top-down keypoint loss.
#         Note:
#             - batch_size: N
#             - num_keypoints: K
#             - heatmaps height: H
#             - heatmaps weight: W
#         Args:
#             output (torch.Tensor[N,K,H,W]): Output heatmaps.
#             target (torch.Tensor[N,K,H,W]): Target heatmaps.
#             target_weight (torch.Tensor[N,K,1]):
#                 Weights across different joint types.
#         """
#         assert img.size(0) == len(img_metas)
#         batch_size, _, img_height, img_width = img.shape
#         if batch_size > 1:
#             assert 'bbox_id' in img_metas[0]
#         accuracy = dict()
#         if self.target_type == 'GaussianHeatmap':
#             output_point = TopdownHeatmapBaseHead.decode(self, img_metas, output.cpu().detach().numpy(),img_size=[img_width, img_height])
#             target_point = TopdownHeatmapBaseHead.decode(self, img_metas, target.cpu().detach().numpy(),img_size=[img_width, img_height])
#             #print('output_point:',output_point['preds'].shape)
#             #print('target_point:',target_point['preds'].shape)
#             avg_acc = keypoint_epe(
#                     output_point['preds'],
#                     target_point['preds'],
#                     target_weight.detach().cpu().numpy().squeeze(-1) > 0)
# #            _, avg_acc, _ = keypoint_epe(
# #                output.detach().cpu().numpy(),
# #                target.detach().cpu().numpy(),
# #                target_weight.detach().cpu().numpy().squeeze(-1) > 0)
#             accuracy['MRE_g'] = float(avg_acc)
#         return accuracy

#     def forward(self, x,local_feature,local_output):
#         # print('x.shape',x.shape) torch.Size([2, 3, 1216, 1024])
#         # print('local_feature',local_feature.shape) torch.Size([2, 19, 608, 512])
#         """Forward function."""
#         # x = x[1]
#         # x = F.interpolate(x, scale_factor=0.5)

#         # AdultUnder
#         x = F.interpolate(x, scale_factor=0.5)
#         x = self.w1_conv(x) #3->32
#         # x = F.interpolate(x, scale_factor=0.5)#3->32
#         # local_feature = self.w2_conv(local_feature) #256->64 可选
#         # local_feature = self.upsample2(local_feature)#H/2xW/2->HxW
#         # local_output = self.upsample2(local_output)#H/2xW/2->HxW
#         out = torch.cat((local_feature, x, local_output), 1)
#         out = self.layer(out)#3x3-->1x1conv 133->37
#         out = self.upsample2(out)

#         # # hand
#         # x = self.w1_conv(x) #3->32
#         # # x = F.interpolate(x, scale_factor=0.5)#3->32
#         # # local_feature = self.w2_conv(local_feature) #256->64 可选
#         # local_feature = self.upsample2(local_feature)#H/2xW/2->HxW
#         # local_output = self.upsample2(local_output)#H/2xW/2->HxW
#         # out = torch.cat((local_feature, x, local_output), 1)
#         # out = self.layer(out)#3x3-->1x1conv 133->37
#         # # out = self.upsample2(out)

#         # #ISBI2023
#         # # x = F.interpolate(x, scale_factor=0.5)#3->32
#         # x = self.w1_conv(x) #3->32
#         # x = F.interpolate(x, scale_factor=0.5)#3->32
#         # # local_feature = self.w2_conv(local_feature) #256->64 可选
#         # # local_feature = self.upsample2(local_feature)#H/2xW/2->HxW
#         # # local_output = self.upsample2(local_output)#H/2xW/2->HxW
#         # out = torch.cat((local_feature, x, local_output), 1)
#         # out = self.layer(out)#3x3-->1x1conv 133->37
#         # out = self.upsample2(out)

#         # x = self.w1_conv(x) #3->32
#         # # local_feature = self.w2_conv(local_feature) #256->64
#         # local_feature = self.upsample2(local_feature)#H/2xW/2->HxW
#         # local_output = self.upsample2(local_output)#H/2xW/2->HxW
#         # out = torch.cat((local_feature, x, local_output), 1)
#         # out = self.layer(out)#3x3-->1x1conv 32+256+19->19  64+32+19->19 115->19

#         # x = self.w1_conv(x)#3->32      
#         # x = F.interpolate(x, scale_factor=0.5)#3->32
#         # # x = F.interpolate(x, scale_factor=0.5)#3->32
#         # # local_feature = self.w2_conv(local_feature) #256->64
#         # out = torch.cat((local_feature, x, local_output), 1)
#         # out = self.layer(out)
#         # out = self.upsample2(out)
        
#         # x = F.interpolate(x, scale_factor=0.5)#3->32
#         # x = self.w1_conv(x)#3->32
#         # out = torch.cat((local_feature, x, local_output), 1)
        
#         # out = self.layer(out)
#         # out = self.upsample2(out)
#         #-----
#         # x = F.interpolate(x, scale_factor=0.5)#3->32
#         # x = self.w1_conv(x)#3->32
#         # local_feature = self.w2_conv(local_feature) #256->64
#         # out = torch.cat((local_feature, x, local_output), 1)
        
#         # out = self.upsample2(out)
#         # out = self.layer(out)


#         # out = self.upsample2(out)

#         # global_feature = self.globalNet(x,local_feature)
#         # print('global_feature',global_feature.shape)#torch.Size([2, 19, 608, 512])
#         # out = global_feature*local_feature
#         # print('out',out.shape)#torch.Size([2, 19, 608, 512])
#         # x  = self.layer(x)
#         # x  = self.layer(x[1])
#         # print('x------------------',x.shape)#x------------------ torch.Size([2, 76, 304, 256]) torch.Size([2, 19, 608, 512])
 
#         return out

#     def inference_model(self, x,local_feature, local_output, flip_pairs=None):
#         """Inference function.
#         Returns:
#             output_heatmap (np.ndarray): Output heatmaps.
#         Args:
#             x (torch.Tensor[N,K,H,W]): Input features.
#             flip_pairs (None | list[tuple]):
#                 Pairs of keypoints which are mirrored.
#         """
#         # output = self.forward(x,local_feature)
#         output = self.forward(x,local_feature, local_output)
#         if flip_pairs is not None:
#             output_heatmap = flip_back(
#                 output.detach().cpu().numpy(),
#                 flip_pairs,
#                 target_type=self.target_type)
#             # feature is not aligned, shift flipped heatmap for higher accuracy
#             if self.test_cfg.get('shift_heatmap', False):
#                 output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
#         else:
#             output_heatmap = output.detach().cpu().numpy()
#         return output_heatmap

#     def init_weights(self):
#         """Initialize model weights."""
#         for _, m in self.layer.named_modules():
#             if isinstance(m, nn.Conv2d):
#                 normal_init(m, std=0.001, bias=0)
#         for _, m in self.w1_conv.named_modules():
#             if isinstance(m, nn.Conv2d):
#                 normal_init(m, std=0.001, bias=0)
#         # for _, m in self.globalNet.named_modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         normal_init(m, std=0.001, bias=0)
# # 