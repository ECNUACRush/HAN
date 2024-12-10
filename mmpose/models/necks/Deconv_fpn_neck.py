# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
#                       constant_init, normal_init, xavier_init)
# from torch.utils.checkpoint import checkpoint

# from ..builder import NECKS


# class ConvBlock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
#         self.norm = nn.BatchNorm2d(dim)
#         self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = self.norm(x)
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

#         x = input + x
#         return x


# @NECKS.register_module()
# class Deconv_FPN_Neck(nn.Module):
#     """

#     Args:
#         in_channels (list): number of channels for each branch.
#         out_channels (int): output channels of feature pyramids.
#         num_outs (int): number of output stages.
#         pooling_type (str): pooling for generating feature pyramids
#             from {MAX, AVG}.
#         conv_cfg (dict): dictionary to construct and config conv layer.
#         norm_cfg (dict): dictionary to construct and config norm layer.
#         with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
#             memory while slowing down the training speed.
#         stride (int): stride of 3x3 convolutional layers
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  num_deconv_layers=2, #4
#                  num_deconv_filters=(256, 256),
#                  num_deconv_kernels=(4, 4),
#                  in_index=0,
#                  extra=None,
#                  input_transform=None,
#                  align_corners=False):
#         super(Deconv_FPN_Neck, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.align_corners = align_corners
#         self.upsample_cfg = dict(mode='nearest')
#         self.up = nn.Upsample(
#                 scale_factor=2, mode='bilinear', align_corners=True)

#         self.lateral_convs = nn.ModuleList()
#         self.c_convs = nn.ModuleList()
#         self.d_convs = nn.ModuleList()
#         self.conv_block = nn.ModuleList()
#         # 64 128 320 512 1024 1-5 
#         self.pre_interpolate = nn.ModuleList([nn.Sequential(
#             nn.Conv2d(out_channels[i],out_channels[i],3,1,1,1,out_channels[i]),
#             nn.Conv2d(out_channels[i],out_channels[i-1],1),
#             nn.BatchNorm2d(out_channels[i-1]),
#             nn.ReLU()
#         ) for i in range( len(out_channels)-1,0, -1)])
#         # ConvBlock(out_channels)
#         for i in range(num_deconv_layers):#4 0-1-2-3
#             l_conv = build_conv_layer(
#                 dict(type='Conv2d'),
#                 in_channels=in_channels[i + 1], #1-2-3-4
#                 out_channels=out_channels,
#                 kernel_size=3,
#                 padding=1
#             )  # 横向 (64) 128 320 512 (1024) ->256
#             c_conv = build_conv_layer(
#                 dict(type='Conv2d'),
#                 in_channels=in_channels[i],
#                 out_channels=in_channels[i],
#                 kernel_size=3,
#                 padding=1,
#             )
#             d_conv = build_conv_layer(
#                 dict(type='Conv2d'),
#                 in_channels=in_channels[i]*2,
#                 out_channels=in_channels[i],
#                 kernel_size=3,
#                 padding=1,
#             )
#             self.lateral_convs.append(l_conv)
#             self.c_convs.append(c_conv)
#             self.d_convs.append(d_conv)
#         # if num_deconv_layers > 0:
#         #     self.deconv_layers = self._make_deconv_layer(
#         #         num_deconv_layers,
#         #         num_deconv_filters,
#         #         num_deconv_kernels,
#         #     )

#     def init_weights(self):
#         """Initialize model weights."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution='uniform')

#     def forward(self, x):
#         """Forward function."""
#         # x = x[1:] # 64 128 320 512 1024
#         # laterals = [
#         #     lateral_conv(x[i])
#         #     for i, lateral_conv in enumerate(self.lateral_convs)
#         # ]
#         used_backbone_levels = len(x)
#         num = 0
#         outs = x
#         for i in range(used_backbone_levels-1 , 0, -1): # 4 3 2 1
#             # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
#             #  it cannot co-exist with `size` in `F.interpolate`.
#             if 'scale_factor' in self.upsample_cfg:
#                 x[i - 1] += F.interpolate(x[i],
#                                                  **self.upsample_cfg)
#                 # laterals[i - 1] += F.interpolate(laterals[i],
#                 #                                  **self.upsample_cfg)
#             else:
#                 prev_shape = x[i - 1].shape[2:] #512的shape
#                 # deconv_in = laterals[i]
#                 # for a in range(3):
#                 #     deconv_in =self.deconv_layers[a+(3*num)](deconv_in)

#                 # laterals[i - 1] += deconv_in
#                 # num+=1
#                 up_x = x[i]
#                 up_x= self.pre_interpolate[num](up_x) #1024->512
#                 num+=1
#                 up_x = F.interpolate(
#                     up_x, size=prev_shape, **self.upsample_cfg) #up
#                 # print('up_x',up_x.shape)
#                 # print('x[i-1]',x[i-1].shape)
#                 x[i-1] = torch.cat([up_x,x[i-1]], 1)
#                 # print('x[i-1]',x[i-1].shape)
#                 x[i-1] = self.d_convs[i-1](x[i-1]) #cat1024-512
#                 # print('x[i-1]',x[i-1].shape)
#                 x[i-1] =self.c_convs[i-1](x[i-1]) # 3x3卷积 
#                 outs[i-1] = x[i-1]
#                 # print('x[i-1]',x[i-1].shape)

#                 # laterals[i - 1] += F.interpolate(
#                 #     laterals[i], size=prev_shape, **self.upsample_cfg)  #
#                 # print('1',laterals[i - 1].shape)
#                 # laterals[i - 1] = self.conv_block(laterals[i - 1])
#                 # print('2',laterals[i - 1].shape)
#         # outs = [
#         #     self.c_convs[i](laterals[i]) for i in range(used_backbone_levels)
#         # ]
#         # out = outs[0]
#         # for i in outs:
#         #     print(i.shape)

#         return outs

#     def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
#         """Make deconv layers."""
#         if num_layers != len(num_filters):
#             error_msg = f'num_layers({num_layers}) ' \
#                         f'!= length of num_filters({len(num_filters)})'
#             raise ValueError(error_msg)
#         if num_layers != len(num_kernels):
#             error_msg = f'num_layers({num_layers}) ' \
#                         f'!= length of num_kernels({len(num_kernels)})'
#             raise ValueError(error_msg)

#         layers = []
#         for i in range(num_layers):
#             kernel, padding, output_padding = \
#                 self._get_deconv_cfg(num_kernels[i])

#             planes = num_filters[i]
#             layers.append(
#                 build_upsample_layer(
#                     dict(type='deconv'),
#                     in_channels=planes,
#                     out_channels=planes,
#                     kernel_size=kernel,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=False))
#             layers.append(nn.BatchNorm2d(planes))
#             layers.append(nn.ReLU(inplace=True))
#             self.in_channels = planes

#         return nn.Sequential(*layers)

#     @staticmethod
#     def _get_deconv_cfg(deconv_kernel):
#         """Get configurations for deconv layers."""
#         if deconv_kernel == 4:
#             padding = 1
#             output_padding = 0
#         elif deconv_kernel == 3:
#             padding = 1
#             output_padding = 1
#         elif deconv_kernel == 2:
#             padding = 0
#             output_padding = 0
#         else:
#             raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

#         return deconv_kernel, padding, output_padding
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init, xavier_init)
from torch.utils.checkpoint import checkpoint

from ..builder import NECKS


class ConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# @NECKS.register_module()
# class Deconv_FPN_Neck(nn.Module):
#     """

#     Args:
#         in_channels (list): number of channels for each branch.
#         out_channels (int): output channels of feature pyramids.
#         num_outs (int): number of output stages.
#         pooling_type (str): pooling for generating feature pyramids
#             from {MAX, AVG}.
#         conv_cfg (dict): dictionary to construct and config conv layer.
#         norm_cfg (dict): dictionary to construct and config norm layer.
#         with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
#             memory while slowing down the training speed.
#         stride (int): stride of 3x3 convolutional layers
#     """

#     def __init__(self,
#                  in_channels = [64,128,320,512,1024],
#                  out_channels = [64,128,320,512,1024],
#                  num_deconv_layers=4,
#                  num_deconv_filters=(256, 256),
#                  num_deconv_kernels=(4, 4),
#                  in_index=0,
#                  extra=None,
#                  input_transform=None,
#                  bilinear=True,
#                  align_corners=False):
#         super(Deconv_FPN_Neck, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.align_corners = align_corners
#         self.upsample_cfg = dict(mode='nearest')
#         self.Up = nn.ModuleList()
#         self.bilinear = bilinear
#         factor = 2 if bilinear else 1
#         # for i in range(num_deconv_layers, 0, -1):#4 3 2 1 [64,128,320,512,1024],
#         #     up = Up(in_channels[i], out_channels[i-1] // factor, bilinear)

#         self.up1 = Up(1024, 320, bilinear)
#         self.up2 = Up(640, 128, bilinear)
#         self.up3 = Up(256, 64, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.l_conv = build_conv_layer(
#                 dict(type='Conv2d'),
#                 in_channels=1024,
#                 out_channels=512,
#                 kernel_size=3,
#                 padding=1
#             )

#     def init_weights(self):
#         """Initialize model weights."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution='uniform')

#     def forward(self, x):
#         out = x
#         out[4] = self.l_conv(x[4])
#         out[3] = self.up1(out[4], x[3])
#         out[2] = self.up2(out[3], x[2])
#         out[1] = self.up3(out[2], x[1])
#         out[0] = self.up4(out[1], x[0])
#         # for o in out:
#         #     print(o.shape)

#         return out

@NECKS.register_module()
class Deconv_FPN_Neck(nn.Module):
    """

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        stride (int): stride of 3x3 convolutional layers
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=256,
                 num_deconv_layers=2,
                 num_deconv_filters=(256, 256),
                 num_deconv_kernels=(4, 4),
                 in_index=0,
                 extra=None,
                 input_transform=None,
                 align_corners=False):
        super(Deconv_FPN_Neck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.upsample_cfg = dict(mode='nearest')

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.lateral_convs = nn.ModuleList()
        self.c_convs = nn.ModuleList()
        # self.conv_block = ConvBlock(out_channels)
        self.conv_block = nn.ModuleList()
        for i in range(num_deconv_layers + 1):
            l_conv = build_conv_layer(
                dict(type='Conv2d'),
                in_channels=in_channels[i + 1],
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )  # 横向512 320 128 64 ->256

            c_conv = build_conv_layer(
                dict(type='Conv2d'),
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
            self.lateral_convs.append(l_conv)
            self.c_convs.append(c_conv)
            # self.c_convs.append(double_conv)

        # if num_deconv_layers > 0:
        #     self.deconv_layers = self._make_deconv_layer(
        #         num_deconv_layers,
        #         num_deconv_filters,
        #         num_deconv_kernels,
        #     )

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        """Forward function."""
        x = x[1:] # 128 320 512 1024
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        num = 0
        for i in range(used_backbone_levels - 1, 0, -1): #4 3 2 1
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:] #512的shape
                # deconv_in = laterals[i]
                # for a in range(3):
                #     deconv_in =self.deconv_layers[a+(3*num)](deconv_in)

                # laterals[i - 1] += deconv_in
                # num+=1
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)  #
                
                # laterals[i - 1] = self.c_convs[i-1](laterals[i-1])

                # print('1',laterals[i - 1].shape)
                # laterals[i - 1] = self.conv_block(laterals[i - 1])
                # print('2',laterals[i - 1].shape)
        outs = [
            self.c_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # out = self.double_conv(laterals[0])

        return outs[0]

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding


