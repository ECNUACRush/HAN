# # Copyright (c) OpenMMLab. All rights reserved.
from .ae_higher_resolution_head import AEHigherResolutionHead
from .ae_multi_stage_head import AEMultiStageHead
from .ae_simple_head import AESimpleHead
from .deconv_head import DeconvHead
from .deeppose_regression_head import DeepposeRegressionHead
from .hmr_head import HMRMeshHead
from .interhand_3d_head import Interhand3DHead
from .temporal_regression_head import TemporalRegressionHead
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead
from .topdown_heatmap_multi_stage_head import (TopdownHeatmapMSMUHead,
                                               TopdownHeatmapMultiStageHead)
from .topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from .vipnas_heatmap_simple_head import ViPNASHeatmapSimpleHead
from .voxelpose_head import CuboidCenterHead, CuboidPoseHead
from .topdown_convhead_heatmap import TopdownSimple_Heatmap_ConvHead
from .topdown_distance_deconvhead import TopdownSimple_Distance_DeConvHead
from .topdown_convhead_heatmap_down import TopdownSimple_Heatmap_ConvHead_down
from .topdown_convhead_heatmap_downx2 import TopdownSimple_Heatmap_ConvHead_downx2
from .topdown_globalhead_heatmap import TopdownSimple_Heatmap_ConvHead_global

__all__ = [
    'TopdownHeatmapSimpleHead', 'TopdownHeatmapMultiStageHead',
    'TopdownHeatmapMSMUHead', 'TopdownHeatmapBaseHead',
    'AEHigherResolutionHead', 'AESimpleHead', 'AEMultiStageHead',
    'DeepposeRegressionHead', 'DeconvHead', 'TemporalRegressionHead', 'Interhand3DHead',
    'HMRMeshHead', 'ViPNASHeatmapSimpleHead', 'CuboidCenterHead',
    'CuboidPoseHead','TopdownSimple_Heatmap_ConvHead', 'TopdownSimple_Distance_DeConvHead','TopdownSimple_Heatmap_ConvHead_down','TopdownSimple_Heatmap_ConvHead_downx2','TopdownSimple_Heatmap_ConvHead_global'
]
