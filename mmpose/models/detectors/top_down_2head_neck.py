# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import cv2
import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core import imshow_bboxes, imshow_keypoints
from .. import builder
from ..builder import POSENETS
from .base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class TopDown_2Head_Neck(BasePose):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 distance_head=None,
                 keypoint_head=None,
                 keypoint_head_global = None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super().__init__()
        self.fp16_enabled = False

        self.backbone = builder.build_backbone(backbone)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if neck is not None:
            self.neck = builder.build_neck(neck)
        
        if keypoint_head_global is not None:
            keypoint_head_global['train_cfg'] = train_cfg
            keypoint_head_global['test_cfg'] = test_cfg

            if 'loss_keypoint' not in keypoint_head_global and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head_global['loss_keypoint'] = loss_pose

            self.keypoint_head_global = builder.build_head(keypoint_head_global)

        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg
            keypoint_head['test_cfg'] = test_cfg

            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose

            self.keypoint_head = builder.build_head(keypoint_head)

        if distance_head is not None:
            distance_head['train_cfg'] = train_cfg
            distance_head['test_cfg'] = test_cfg

            if 'loss_keypoint_reg' not in distance_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                distance_head['Distance_loss'] = loss_pose

            self.distance_head = builder.build_head(distance_head)

        self.init_weights(pretrained=pretrained)

    @property
    def with_neck(self):
        """Check if has neck."""
        return hasattr(self, 'neck')
    @property
    def with_keypoint_global(self):
        """Check if has keypoint_head_global."""
        return hasattr(self, 'keypoint_head_global')
    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')
    @property
    def with_distance(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'distance_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        if self.with_neck:
            self.neck.init_weights()
        if self.with_keypoint_global:
            self.keypoint_head_global.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()
        if self.with_distance:
            self.distance_head.init_weights()

    @auto_fp16(apply_to=('img',))
    def forward(self,
                img,
                target=None,
                target_weight=None,
                target_reg=None,
                target_weight_reg=None,
                target_reg_down=None, 
                target_weight_reg_down=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_img_channel: C (Default: 3)
            - img height: imgH
            - img width: imgW
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses. \
                Otherwise, return predicted poses, boxes, image paths \
                and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, target_reg, target_weight_reg,target_reg_down, target_weight_reg_down, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, target, target_weight, target_reg, target_weight_reg,target_reg_down, target_weight_reg_down, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        features = self.backbone(img)
        features_neck = self.neck(features)
        if self.with_keypoint:
            output_keypoint = self.keypoint_head(features_neck)
        if self.with_distance:
            # output_distance, out_feature = self.distance_head(features_neck,output_keypoint)
            output_distance, out_feature = self.distance_head(features_neck)
        if self.with_keypoint_global:
            output_global = self.keypoint_head_global(img,out_feature, output_distance)

        # if return loss
        losses = dict()
  
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output_keypoint, target, target_weight)
                # output_keypoint, target_reg, target_weight_reg)
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output_keypoint, target, target_weight, img, img_metas)
                # output_keypoint, target_reg, target_weight_reg, img, img_metas)
            losses.update(keypoint_accuracy)
        if self.with_distance:
            distance_losses = self.distance_head.get_loss(
                output_distance, target_reg, target_weight_reg)
            losses.update(distance_losses)
            distance_accuracy = self.distance_head.get_accuracy(
                output_distance, target_reg, target_weight_reg, img, img_metas)
            losses.update(distance_accuracy)
        if self.with_keypoint_global:
            global_losses = self.keypoint_head_global.get_loss(
                output_global, target_reg_down, target_weight_reg_down)
                # output_global, target_reg, target_weight_reg)
            losses.update(global_losses)
            global_accuracy = self.keypoint_head_global.get_accuracy(
                output_global, target_reg_down, target_weight_reg_down, img, img_metas)
                # output_global, target_reg, target_weight_reg, img, img_metas)
            losses.update(global_accuracy)

        return losses

    def forward_test(self, img, img_metas, return_heatmap=True, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)
        if self.with_distance:
            output_heat2x = self.distance_head.inference_model(features, flip_pairs=None)
        if self.with_keypoint_global:
            output_distance,out_feature = self.distance_head(features)
            output_heat2x_global = self.keypoint_head_global.inference_model(img,out_feature,output_distance, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)
            if self.with_neck:
                features_flipped = self.neck(features_flipped)
            if self.with_keypoint:
                output_flipped_heatmap = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output_heatmap = (output_heatmap +
                                  output_flipped_heatmap) * 0.5
            if self.with_distance:
                output_fliped_heatmap_2x = self.distance_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output_heat2x = (output_heat2x +
                                 output_fliped_heatmap_2x) * 0.5
            if self.with_keypoint_global:
                output_distance,out_feature = self.distance_head(features_flipped)
                output_fliped_heatmap_global = self.keypoint_head_global.inference_model(
                    img_flipped,out_feature,output_distance, img_metas[0]['flip_pairs'])
                output_heat2x_global = (output_heat2x_global +
                                 output_fliped_heatmap_global) * 0.5
 
        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_heatmap, img_size=[img_width, img_height])
            if not return_heatmap:
                # print('no_heat')
                output_heatmap = None

            result['output_heatmap'] = output_heatmap
        if self.with_distance:
            keypoint_result_2x = self.distance_head.decode(
                img_metas, output_heat2x, img_size=[img_width, img_height])
        if self.with_keypoint_global:
            keypoint_result_global = self.keypoint_head_global.decode(
                img_metas, output_heat2x_global, img_size=[img_width, img_height])
        # pred_point_down = keypoint_result_down['preds']
        if self.with_keypoint:
            pred_point4x = keypoint_result['preds']
        if self.with_distance:
            pred_point2x = keypoint_result_2x['preds']
        if self.with_keypoint_global:
            pred_point2x_global = keypoint_result_global['preds']

        if self.with_keypoint_global:
            pred_avg =((pred_point4x) + (pred_point2x)+(pred_point2x_global)) / 3
        else:
            pred_avg = ((pred_point4x) + (pred_point2x)) / 2 
        keypoint_result['preds'] = pred_avg
        result.update(keypoint_result)
        return result

    def forward_dummy(self, img):
        """Used for computing network FLOPs.
        See ``tools/get_flops.py``.
        Args:
            img (torch.Tensor): Input image.
        Returns:
            Tensor: Output heatmaps.
        """
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output_heatmap = self.keypoint_head(output)
        if self.with_distance:
            output_distance = self.distance_head(output)
        return output_heatmap

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.0,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=1,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    gt=None,
                    show_gt=False,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        #        if bbox_result:
        #            bboxes = np.vstack(bbox_result)
        #            # draw bounding boxes
        #            imshow_bboxes(
        #                img,
        #                bboxes,
        #                labels=bbox_labels,
        #                colors=bbox_color,
        #                text_color=text_color,
        #                thickness=bbox_thickness,
        #                font_scale=font_scale,
        #                show=False)
        if show_gt:
            img = mmcv.imread(img)
            joints_3d = gt
            for i, key_cor in enumerate(joints_3d):
                x, y = key_cor
                #               print('(x,y):',x,' , ',y)
                img = cv2.drawMarker(img, (int(float(x)), int(float(y))), color=(0, 205, 0),
                                     markerType=cv2.MARKER_TILTED_CROSS, thickness=2, markerSize=10)
                # img = cv2.circle(img, (int(x),int(y)), radius=5, color=(255, 0, 0), thickness=-1)

        if pose_result:
            imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img

