# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
from mmpose.core.post_processing import (affine_transform, fliplr_joints,
                                         get_affine_transform, get_warp_matrix,
                                         warp_affine_joints)
from mmpose.datasets.builder import PIPELINES
from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps,keypoints_from_regression
from mmpose.core.post_processing import transform_preds
from mmpose.core.evaluation import keypoint_epe
import math
from scipy.spatial import distance


@PIPELINES.register_module()
class TopDownRandomFlip:
    """Data augmentation with random image flip.

    Required keys: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'ann_info'.

    Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'flipped'.

    Args:
        flip (bool): Option to perform random flip.
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        if 'center' in results:
            center = results['center']

        # A flag indicating whether the image is flipped,
        # which can be used by child class.
        flipped = False
        if np.random.rand() <= self.flip_prob:
            flipped = True
            if not isinstance(img, list):
                img = img[:, ::-1, :]
            else:
                img = [i[:, ::-1, :] for i in img]
            if not isinstance(img, list):
                joints_3d, joints_3d_visible = fliplr_joints(
                    joints_3d, joints_3d_visible, img.shape[1],
                    results['ann_info']['flip_pairs'])
                if 'center' in results:
                    center[0] = img.shape[1] - center[0] - 1
            else:
                joints_3d, joints_3d_visible = fliplr_joints(
                    joints_3d, joints_3d_visible, img[0].shape[1],
                    results['ann_info']['flip_pairs'])
                if 'center' in results:
                    center[0] = img[0].shape[1] - center[0] - 1
        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        if 'center' in results:
            results['center'] = center
        results['flipped'] = flipped

        return results

@PIPELINES.register_module()
class TopDownGetRandomScaleRotation:
    """Data augmentation with random scaling & rotating.

    Required key: 'scale'.

    Modifies key: 'scale' and 'rotation'.

    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(self, rot_factor=40, scale_factor=0.5, rot_prob=0.6):
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        s = results['scale']

        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        s = s * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r = r_factor if np.random.rand() <= self.rot_prob else 0

        results['scale'] = s
        results['rotation'] = r

        return results

@PIPELINES.register_module()
class TopDownAffine:
    """Affine transform the image to make input.

    Required keys:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
    'rotation' and 'center'.

    Modified keys:'img', 'joints_3d', and 'joints_3d_visible'.

    Args:
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, use_udp=False):
        self.use_udp = use_udp

    def __call__(self, results):
        image_size = results['ann_info']['image_size']
        img = results['img']
        joints_3d = results['joints_3d']
        #print('joints_3d:',joints_3d)
        joints_3d_visible = results['joints_3d_visible']
        #joints_temp = joints_3d
        c = results['center']
        s = results['scale']
        r = results['rotation']
        img_path = results['image_file']
        #joints = np.zeros((len(joints_3d), 3), dtype=np.float32)
        if self.use_udp:
            trans = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]

            joints_3d[:, 0:2] = \
                warp_affine_joints(joints_3d[:, 0:2].copy(), trans)

        else:
            trans = get_affine_transform(c, s, r, image_size)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]
            for i in range(results['ann_info']['num_joints']):
                if joints_3d_visible[i, 0] > 0.0:
                    joints_3d[i,0:2] = affine_transform(joints_3d[i, 0:2], trans) 

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        #cv2.imwrite('/data2/Thanaporn/input_img.jpg',img)
        return results

@PIPELINES.register_module()
class TopDownGenerateTarget:
    """Generate the target heatmap.

    Required keys: 'joints_3d', 'joints_3d_visible', 'ann_info'.

    Modified keys: 'target', and 'target_weight'.

    Args:
        sigma: Sigma of heatmap gaussian for 'MSRA' approach.
        kernel: Kernel of heatmap gaussian for 'Megvii' approach.
        encoding (str): Approach to generate target heatmaps.
            Currently supported approaches: 'MSRA', 'Megvii', 'UDP'.
            Default:'MSRA'
        unbiased_encoding (bool): Option to use unbiased
            encoding methods.
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        keypoint_pose_distance: Keypoint pose distance for UDP.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
        target_type (str): supported targets: 'GaussianHeatmap',
            'CombinedTarget'. Default:'GaussianHeatmap'
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self,
                 sigma=2,
                 kernel=(11, 11),
                 valid_radius_factor=0.0546875,
                 target_type='GaussianHeatmap',
                 encoding='MSRA',
                 unbiased_encoding=False):
        self.sigma = sigma
        self.unbiased_encoding = unbiased_encoding
        self.kernel = kernel
        self.valid_radius_factor = valid_radius_factor
        self.target_type = target_type
        self.encoding = encoding

    def _msra_generate_target(self, cfg, joints_3d, joints_3d_visible, sigma):
        """Generate the target heatmap via "MSRA" approach.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray ([num_joints, 3])
            joints_3d_visible: np.ndarray ([num_joints, 3])
            sigma: Sigma of heatmap gaussian
        Returns:
            tuple: A tuple containing targets.

            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        W, H = cfg['heatmap_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        target = np.zeros((num_joints, H, W), dtype=np.float32)

        # 3-sigma rule
        tmp_size = sigma * 3

        if self.unbiased_encoding:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                # Check that any part of the gaussian is in-bounds
                ul = [mu_x - tmp_size, mu_y - tmp_size]
                br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] == 0:
                    continue

                x = np.arange(0, W, 1, np.float32)
                y = np.arange(0, H, 1, np.float32)
                y = y[:, None]

                if target_weight[joint_id] > 0.5:
                    target[joint_id] = np.exp(-((x - mu_x)**2 +
                                                (y - mu_y)**2) /
                                              (2 * sigma**2))
        else:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] > 0.5:
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, None]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized,
                    # we want the center value to equal 1
                    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], W)
                    img_y = max(0, ul[1]), min(br[1], H)

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                    
        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def _megvii_generate_target(self, cfg, joints_3d, joints_3d_visible,
                                kernel):
        """Generate the target heatmap via "Megvii" approach.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray ([num_joints, 3])
            joints_3d_visible: np.ndarray ([num_joints, 3])
            kernel: Kernel of heatmap gaussian

        Returns:
            tuple: A tuple containing targets.

            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """

        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        W, H = cfg['heatmap_size']
        heatmaps = np.zeros((num_joints, H, W), dtype='float32')
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)

        for i in range(num_joints):
            target_weight[i] = joints_3d_visible[i, 0]

            if target_weight[i] < 1:
                continue

            target_y = int(joints_3d[i, 1] * H / image_size[1])
            target_x = int(joints_3d[i, 0] * W / image_size[0])

            if (target_x >= W or target_x < 0) \
                    or (target_y >= H or target_y < 0):
                target_weight[i] = 0
                continue

            heatmaps[i, target_y, target_x] = 1
            heatmaps[i] = cv2.GaussianBlur(heatmaps[i], kernel, 0)
            maxi = heatmaps[i, target_y, target_x]

            heatmaps[i] /= maxi / 255

        return heatmaps, target_weight

    def _udp_generate_target(self, cfg, joints_3d, joints_3d_visible, factor,
                             target_type):
        """Generate the target heatmap via 'UDP' approach. Paper ref: Huang et
        al. The Devil is in the Details: Delving into Unbiased Data Processing
        for Human Pose Estimation (CVPR 2020).

        Note:
            - num keypoints: K
            - heatmap height: H
            - heatmap width: W
            - num target channels: C
            - C = K if target_type=='GaussianHeatmap'
            - C = 3*K if target_type=='CombinedTarget'

        Args:
            cfg (dict): data config
            joints_3d (np.ndarray[K, 3]): Annotated keypoints.
            joints_3d_visible (np.ndarray[K, 3]): Visibility of keypoints.
            factor (float): kernel factor for GaussianHeatmap target or
                valid radius factor for CombinedTarget.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Heatmap target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).

        Returns:
            tuple: A tuple containing targets.

            - target (np.ndarray[C, H, W]): Target heatmaps.
            - target_weight (np.ndarray[K, 1]): (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        heatmap_size = cfg['heatmap_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d_visible[:, 0]

        if target_type.lower() == 'GaussianHeatmap'.lower():
            target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = factor * 3

            # prepare for gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]

            for joint_id in range(num_joints):
                feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                mu_x_ac = joints_3d[joint_id][0] / feat_stride[0]
                mu_y_ac = joints_3d[joint_id][1] / feat_stride[1]
                x0 = y0 = size // 2
                x0 += mu_x_ac - mu_x
                y0 += mu_y_ac - mu_y
                g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * factor**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        elif target_type.lower() == 'CombinedTarget'.lower():
            target = np.zeros(
                (num_joints, 3, heatmap_size[1] * heatmap_size[0]),
                dtype=np.float32)
            feat_width = heatmap_size[0]
            feat_height = heatmap_size[1]
            feat_x_int = np.arange(0, feat_width)
            feat_y_int = np.arange(0, feat_height)
            feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
            feat_x_int = feat_x_int.flatten()
            feat_y_int = feat_y_int.flatten()
            # Calculate the radius of the positive area in classification
            #   heatmap.
            valid_radius = factor * heatmap_size[1]
            feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
            for joint_id in range(num_joints):
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                x_offset = (mu_x - feat_x_int) / valid_radius
                y_offset = (mu_y - feat_y_int) / valid_radius
                dis = x_offset**2 + y_offset**2
                keep_pos = np.where(dis <= 1)[0]
                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id, 0, keep_pos] = 1
                    target[joint_id, 1, keep_pos] = x_offset[keep_pos]
                    target[joint_id, 2, keep_pos] = y_offset[keep_pos]
            target = target.reshape(num_joints * 3, heatmap_size[1],
                                    heatmap_size[0])
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def __call__(self, results):
        """Generate the target heatmap."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        center = results['center']
        scale = results['scale']
        cfg = results['ann_info']
        #W, H = results['img_size_ori']
        #joints_temp = results['joints_temp']
        image_filename = results['image_file']
        #print('temp:',results['joints_temp'])
        assert self.encoding in ['MSRA', 'Megvii', 'UDP']
        if self.encoding == 'MSRA':
            if isinstance(self.sigma, list):
                num_sigmas = len(self.sigma)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                heatmap_size = cfg['heatmap_size']

                target = np.empty(
                    (0, num_joints, heatmap_size[1], heatmap_size[0]),
                    dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_sigmas):
                    target_i, target_weight_i = self._msra_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.sigma[i])
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._msra_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible,
                    self.sigma)

        elif self.encoding == 'Megvii':
            if isinstance(self.kernel, list):
                num_kernels = len(self.kernel)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                W, H = cfg['heatmap_size']

                target = np.empty((0, num_joints, H, W), dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_kernels):
                    target_i, target_weight_i = self._megvii_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.kernel[i])
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._megvii_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible,
                    self.kernel)

        elif self.encoding == 'UDP':
            if self.target_type.lower() == 'CombinedTarget'.lower():
                factors = self.valid_radius_factor
                channel_factor = 3
            elif self.target_type.lower() == 'GaussianHeatmap'.lower():
                factors = self.sigma
                channel_factor = 1
            else:
                raise ValueError('target_type should be either '
                                 "'GaussianHeatmap' or 'CombinedTarget'")
            if isinstance(factors, list):
                num_factors = len(factors)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                W, H = cfg['heatmap_size']

                target = np.empty((0, channel_factor * num_joints, H, W),
                                  dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_factors):
                    target_i, target_weight_i = self._udp_generate_target(
                        cfg, joints_3d, joints_3d_visible, factors[i],
                        self.target_type)
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._udp_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible, factors,
                    self.target_type)
        else:
            raise ValueError(
                f'Encoding approach {self.encoding} is not supported!')

        results['target'] = target
        results['target_weight'] = target_weight
        # num_joints = cfg['num_joints']
        # print('heatmap_target_shape:',target.shape)
        # for i in range(num_joints):
        #     target_ = np.expand_dims(target[i,:],axis=0)
        #     target_ = np.array(target_, dtype='float32')
        #     target_  = np.maximum(target_, 0)
        #     target_ /= torch.max(torch.from_numpy(target_))
        #     target_ = np.transpose(np.array( target_),(1,2,0))
        #     target_ = np.uint8(255*target_)
        #     cv2.imwrite('/data2/Thanaporn/heatmap/'+str(i)+'.jpg', target_)

        # f = open('/data2/Thanaporn/Landmark/ISBI2015_error_test.csv','a')
        # print(image_filename)
        # #print('center:',center)
        # #print('scale:',scale)
        # target = np.expand_dims(target, axis=0)
        # preds, maxvals = keypoints_from_heatmaps(
        #     heatmaps=target,
        #     center=[center],
        #     scale=[scale])
        
        # gt_temp = np.zeros((19, 2), dtype=np.float32)
        # gt_temp = joints_temp[:, :2]
        # gt_temp = np.expand_dims(gt_temp, axis=0)
        # target_temp = np.expand_dims((target_weight.squeeze(-1) > 0), axis=0)
        # #print((target_weight.squeeze(-1) > 0).shape)
        # #print('preds:',preds)
        # error = keypoint_epe(
        #             preds,
        #             gt_temp,
        #             target_temp)
        # f.writelines(image_filename+','+str(error)+'\n')
        #print('target_fromHeatmap:',preds)
        #print('error:',error)
    
        return results

@PIPELINES.register_module()
class TopDownRandomTranslation:
    """Data augmentation with random translation.

    Required key: 'scale' and 'center'.

    Modifies key: 'center'.

    Note:
        - bbox height: H
        - bbox width: W

    Args:
        trans_factor (float): Translating center to
            ``[-trans_factor, trans_factor] * [W, H] + center``.
        trans_prob (float): Probability of random translation.
    """

    def __init__(self, trans_factor=0.15, trans_prob=1.0):
        self.trans_factor = trans_factor
        self.trans_prob = trans_prob

    def __call__(self, results):
        """Perform data augmentation with random translation."""
        center = results['center']
        scale = results['scale']
        if np.random.rand() <= self.trans_prob:
            # reference bbox size is [200, 200] pixels
            center += self.trans_factor * np.random.uniform(
                -1, 1, size=2) * scale * 200
        results['center'] = center
        return results

@PIPELINES.register_module()
class Topdown_Heatmap_Target_Size:   
    def __init__(self, 
                size=2,
                sigma=2,
                kernel=(11, 11),
                valid_radius_factor=0.0546875,
                target_type='GaussianHeatmap',
                encoding='MSRA',#encoding='MSRA'
                unbiased_encoding=False):
        self.size=size
        self.sigma = sigma
        self.unbiased_encoding = unbiased_encoding
        self.kernel = kernel
        self.valid_radius_factor = valid_radius_factor
        self.target_type = target_type
        self.encoding = encoding

    def _msra_generate_target(self, cfg, joints_3d, joints_3d_visible, sigma):
        """Generate the target heatmap via "MSRA" approach.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray ([num_joints, 3])
            joints_3d_visible: np.ndarray ([num_joints, 3])
            sigma: Sigma of heatmap gaussian
        Returns:
            tuple: A tuple containing targets.

            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        W, H = cfg['heatmap_size']
        W = W*self.size
        H = H*self.size
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        target = np.zeros((num_joints, H, W), dtype=np.float32)

        # 3-sigma rule
        tmp_size = sigma * 3

        if self.unbiased_encoding:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                # Check that any part of the gaussian is in-bounds
                ul = [mu_x - tmp_size, mu_y - tmp_size]
                br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] == 0:
                    continue

                x = np.arange(0, W, 1, np.float32)
                y = np.arange(0, H, 1, np.float32)
                y = y[:, None]

                if target_weight[joint_id] > 0.5:
                    target[joint_id] = np.exp(-((x - mu_x)**2 +
                                                (y - mu_y)**2) /
                                              (2 * sigma**2))
        else:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] > 0.5:
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, None]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized,
                    # we want the center value to equal 1
                    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], W)
                    img_y = max(0, ul[1]), min(br[1], H)

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                    
        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def _megvii_generate_target(self, cfg, joints_3d, joints_3d_visible,
                                kernel):
        """Generate the target heatmap via "Megvii" approach.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray ([num_joints, 3])
            joints_3d_visible: np.ndarray ([num_joints, 3])
            kernel: Kernel of heatmap gaussian

        Returns:
            tuple: A tuple containing targets.

            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """

        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        W, H = cfg['heatmap_size']
        W = W*self.size
        H = H*self.size
        heatmaps = np.zeros((num_joints, H, W), dtype='float32')
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)

        for i in range(num_joints):
            target_weight[i] = joints_3d_visible[i, 0]

            if target_weight[i] < 1:
                continue

            target_y = int(joints_3d[i, 1] * H / image_size[1])
            target_x = int(joints_3d[i, 0] * W / image_size[0])

            if (target_x >= W or target_x < 0) \
                    or (target_y >= H or target_y < 0):
                target_weight[i] = 0
                continue

            heatmaps[i, target_y, target_x] = 1
            heatmaps[i] = cv2.GaussianBlur(heatmaps[i], kernel, 0)
            maxi = heatmaps[i, target_y, target_x]

            heatmaps[i] /= maxi / 255

        return heatmaps, target_weight

    def _udp_generate_target(self, cfg, joints_3d, joints_3d_visible, factor,
                             target_type):
        """Generate the target heatmap via 'UDP' approach. Paper ref: Huang et
        al. The Devil is in the Details: Delving into Unbiased Data Processing
        for Human Pose Estimation (CVPR 2020).

        Note:
            - num keypoints: K
            - heatmap height: H
            - heatmap width: W
            - num target channels: C
            - C = K if target_type=='GaussianHeatmap'
            - C = 3*K if target_type=='CombinedTarget'

        Args:
            cfg (dict): data config
            joints_3d (np.ndarray[K, 3]): Annotated keypoints.
            joints_3d_visible (np.ndarray[K, 3]): Visibility of keypoints.
            factor (float): kernel factor for GaussianHeatmap target or
                valid radius factor for CombinedTarget.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Heatmap target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).

        Returns:
            tuple: A tuple containing targets.

            - target (np.ndarray[C, H, W]): Target heatmaps.
            - target_weight (np.ndarray[K, 1]): (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        heatmap_size = cfg['heatmap_size']*self.size
        # heatmap_size[1] = heatmap_size[1]*self.size
        # heatmap_size[0] = heatmap_size[0]*self.size
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d_visible[:, 0]

        if target_type.lower() == 'GaussianHeatmap'.lower():
            target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = factor * 3

            # prepare for gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]

            for joint_id in range(num_joints):
                feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                mu_x_ac = joints_3d[joint_id][0] / feat_stride[0]
                mu_y_ac = joints_3d[joint_id][1] / feat_stride[1]
                x0 = y0 = size // 2
                x0 += mu_x_ac - mu_x
                y0 += mu_y_ac - mu_y
                g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * factor**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        elif target_type.lower() == 'CombinedTarget'.lower():
            target = np.zeros(
                (num_joints, 3, heatmap_size[1] * heatmap_size[0]),
                dtype=np.float32)
            feat_width = heatmap_size[0]
            feat_height = heatmap_size[1]
            feat_x_int = np.arange(0, feat_width)
            feat_y_int = np.arange(0, feat_height)
            feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
            feat_x_int = feat_x_int.flatten()
            feat_y_int = feat_y_int.flatten()
            # Calculate the radius of the positive area in classification heatmap.
            valid_radius = factor * heatmap_size[1]
            feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
            for joint_id in range(num_joints):
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                x_offset = (mu_x - feat_x_int) / valid_radius
                y_offset = (mu_y - feat_y_int) / valid_radius
                dis = x_offset**2 + y_offset**2
                keep_pos = np.where(dis <= 1)[0]
                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id, 0, keep_pos] = 1
                    target[joint_id, 1, keep_pos] = x_offset[keep_pos]
                    target[joint_id, 2, keep_pos] = y_offset[keep_pos]
            target = target.reshape(num_joints * 3, heatmap_size[1],
                                    heatmap_size[0])
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def __call__(self, results):
        """Generate the target heatmap."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        center = results['center']
        scale = results['scale']
        cfg = results['ann_info']
        W, H = results['img_size_ori']
        image_filename = results['image_file']
        assert self.encoding in ['MSRA', 'Megvii', 'UDP']
        if self.encoding == 'MSRA':
            if isinstance(self.sigma, list):
                num_sigmas = len(self.sigma)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                heatmap_size = cfg['heatmap_size']

                target = np.empty(
                    (0, num_joints, heatmap_size[1], heatmap_size[0]),
                    dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_sigmas):
                    target_i, target_weight_i = self._msra_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.sigma[i])
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._msra_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible,
                    self.sigma)

        elif self.encoding == 'Megvii':
            if isinstance(self.kernel, list):
                num_kernels = len(self.kernel)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                W, H = cfg['heatmap_size']

                target = np.empty((0, num_joints, H, W), dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_kernels):
                    target_i, target_weight_i = self._megvii_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.kernel[i])
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._megvii_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible,
                    self.kernel)

        elif self.encoding == 'UDP':
            if self.target_type.lower() == 'CombinedTarget'.lower():
                factors = self.valid_radius_factor
                channel_factor = 3
            elif self.target_type.lower() == 'GaussianHeatmap'.lower():
                factors = self.sigma
                channel_factor = 1
            else:
                raise ValueError('target_type should be either '
                                 "'GaussianHeatmap' or 'CombinedTarget'")
            if isinstance(factors, list):
                num_factors = len(factors)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                W, H = cfg['heatmap_size']

                target = np.empty((0, channel_factor * num_joints, H, W),
                                  dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_factors):
                    target_i, target_weight_i = self._udp_generate_target(
                        cfg, joints_3d, joints_3d_visible, factors[i],
                        self.target_type)
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._udp_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible, factors,
                    self.target_type)
        else:
            raise ValueError(
                f'Encoding approach {self.encoding} is not supported!')

        results['target_reg'] = target 
        results['target_weight_reg'] = target_weight

        return results
    
@PIPELINES.register_module()
class Topdown_Heatmap_Target_Size_down:   
    def __init__(self, 
                size=0.5,
                sigma=2,
                kernel=(11, 11),
                valid_radius_factor=0.0546875,
                target_type='GaussianHeatmap',
                encoding='MSRA',
                unbiased_encoding=False):
        self.size=size
        self.sigma = sigma
        self.unbiased_encoding = unbiased_encoding
        self.kernel = kernel
        self.valid_radius_factor = valid_radius_factor
        self.target_type = target_type
        self.encoding = encoding

    def _msra_generate_target(self, cfg, joints_3d, joints_3d_visible, sigma):
        """Generate the target heatmap via "MSRA" approach.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray ([num_joints, 3])
            joints_3d_visible: np.ndarray ([num_joints, 3])
            sigma: Sigma of heatmap gaussian
        Returns:
            tuple: A tuple containing targets.

            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        W, H = cfg['heatmap_size']
        W = W*self.size
        H = H*self.size
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        target = np.zeros((num_joints, H, W), dtype=np.float32)

        # 3-sigma rule
        tmp_size = sigma * 3

        if self.unbiased_encoding:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                # Check that any part of the gaussian is in-bounds
                ul = [mu_x - tmp_size, mu_y - tmp_size]
                br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] == 0:
                    continue

                x = np.arange(0, W, 1, np.float32)
                y = np.arange(0, H, 1, np.float32)
                y = y[:, None]

                if target_weight[joint_id] > 0.5:
                    target[joint_id] = np.exp(-((x - mu_x)**2 +
                                                (y - mu_y)**2) /
                                              (2 * sigma**2))
        else:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] > 0.5:
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, None]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized,
                    # we want the center value to equal 1
                    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], W)
                    img_y = max(0, ul[1]), min(br[1], H)

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                    
        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def _megvii_generate_target(self, cfg, joints_3d, joints_3d_visible,
                                kernel):
        """Generate the target heatmap via "Megvii" approach.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray ([num_joints, 3])
            joints_3d_visible: np.ndarray ([num_joints, 3])
            kernel: Kernel of heatmap gaussian

        Returns:
            tuple: A tuple containing targets.

            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """

        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        W, H = cfg['heatmap_size']
        W = W*self.size
        H = H*self.size
        heatmaps = np.zeros((num_joints, H, W), dtype='float32')
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)

        for i in range(num_joints):
            target_weight[i] = joints_3d_visible[i, 0]

            if target_weight[i] < 1:
                continue

            target_y = int(joints_3d[i, 1] * H / image_size[1])
            target_x = int(joints_3d[i, 0] * W / image_size[0])

            if (target_x >= W or target_x < 0) \
                    or (target_y >= H or target_y < 0):
                target_weight[i] = 0
                continue

            heatmaps[i, target_y, target_x] = 1
            heatmaps[i] = cv2.GaussianBlur(heatmaps[i], kernel, 0)
            maxi = heatmaps[i, target_y, target_x]

            heatmaps[i] /= maxi / 255

        return heatmaps, target_weight

    def _udp_generate_target(self, cfg, joints_3d, joints_3d_visible, factor,
                             target_type):
        """Generate the target heatmap via 'UDP' approach. Paper ref: Huang et
        al. The Devil is in the Details: Delving into Unbiased Data Processing
        for Human Pose Estimation (CVPR 2020).

        Note:
            - num keypoints: K
            - heatmap height: H
            - heatmap width: W
            - num target channels: C
            - C = K if target_type=='GaussianHeatmap'
            - C = 3*K if target_type=='CombinedTarget'

        Args:
            cfg (dict): data config
            joints_3d (np.ndarray[K, 3]): Annotated keypoints.
            joints_3d_visible (np.ndarray[K, 3]): Visibility of keypoints.
            factor (float): kernel factor for GaussianHeatmap target or
                valid radius factor for CombinedTarget.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Heatmap target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).

        Returns:
            tuple: A tuple containing targets.

            - target (np.ndarray[C, H, W]): Target heatmaps.
            - target_weight (np.ndarray[K, 1]): (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        heatmap_size = cfg['heatmap_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d_visible[:, 0]

        if target_type.lower() == 'GaussianHeatmap'.lower():
            target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = factor * 3

            # prepare for gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]

            for joint_id in range(num_joints):
                feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                mu_x_ac = joints_3d[joint_id][0] / feat_stride[0]
                mu_y_ac = joints_3d[joint_id][1] / feat_stride[1]
                x0 = y0 = size // 2
                x0 += mu_x_ac - mu_x
                y0 += mu_y_ac - mu_y
                g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * factor**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        elif target_type.lower() == 'CombinedTarget'.lower():
            target = np.zeros(
                (num_joints, 3, heatmap_size[1] * heatmap_size[0]),
                dtype=np.float32)
            feat_width = heatmap_size[0]
            feat_height = heatmap_size[1]
            feat_x_int = np.arange(0, feat_width)
            feat_y_int = np.arange(0, feat_height)
            feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
            feat_x_int = feat_x_int.flatten()
            feat_y_int = feat_y_int.flatten()
            # Calculate the radius of the positive area in classification heatmap.
            valid_radius = factor * heatmap_size[1]
            feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
            for joint_id in range(num_joints):
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                x_offset = (mu_x - feat_x_int) / valid_radius
                y_offset = (mu_y - feat_y_int) / valid_radius
                dis = x_offset**2 + y_offset**2
                keep_pos = np.where(dis <= 1)[0]
                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id, 0, keep_pos] = 1
                    target[joint_id, 1, keep_pos] = x_offset[keep_pos]
                    target[joint_id, 2, keep_pos] = y_offset[keep_pos]
            target = target.reshape(num_joints * 3, heatmap_size[1],
                                    heatmap_size[0])
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def __call__(self, results):
        """Generate the target heatmap."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        center = results['center']
        scale = results['scale']
        cfg = results['ann_info']
        W, H = results['img_size_ori']
        image_filename = results['image_file']
        assert self.encoding in ['MSRA', 'Megvii', 'UDP']
        if self.encoding == 'MSRA':
            if isinstance(self.sigma, list):
                num_sigmas = len(self.sigma)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                heatmap_size = cfg['heatmap_size']

                target = np.empty(
                    (0, num_joints, heatmap_size[1], heatmap_size[0]),
                    dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_sigmas):
                    target_i, target_weight_i = self._msra_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.sigma[i])
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._msra_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible,
                    self.sigma)

        elif self.encoding == 'Megvii':
            if isinstance(self.kernel, list):
                num_kernels = len(self.kernel)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                W, H = cfg['heatmap_size']

                target = np.empty((0, num_joints, H, W), dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_kernels):
                    target_i, target_weight_i = self._megvii_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.kernel[i])
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._megvii_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible,
                    self.kernel)

        elif self.encoding == 'UDP':
            if self.target_type.lower() == 'CombinedTarget'.lower():
                factors = self.valid_radius_factor
                channel_factor = 3
            elif self.target_type.lower() == 'GaussianHeatmap'.lower():
                factors = self.sigma
                channel_factor = 1
            else:
                raise ValueError('target_type should be either '
                                 "'GaussianHeatmap' or 'CombinedTarget'")
            if isinstance(factors, list):
                num_factors = len(factors)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                W, H = cfg['heatmap_size']

                target = np.empty((0, channel_factor * num_joints, H, W),
                                  dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_factors):
                    target_i, target_weight_i = self._udp_generate_target(
                        cfg, joints_3d, joints_3d_visible, factors[i],
                        self.target_type)
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._udp_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible, factors,
                    self.target_type)
        else:
            raise ValueError(
                f'Encoding approach {self.encoding} is not supported!')

        results['target_reg_down'] = target 
        results['target_weight_reg_down'] = target_weight

        return results
