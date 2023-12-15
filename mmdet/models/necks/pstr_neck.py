# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2dPack
from mmengine.model import BaseModule

from mmdet.utils import OptConfigType, OptMultiConfig
from mmdet.registry import MODELS


@MODELS.register_module()
class PSTRMapper(BaseModule):
    r"""Channel Mapper to reduce/increase channels of backbone features.

    The PSTR Mapper adds an additional DeformConv2dPack on top of standard
    ConvModule.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): kernel_size for reducing channels (used
            at each scale). Default: 1.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Default: None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Default: None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Default: dict(type='ReLU').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or dict],
            optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        kernel_size: int = 1,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = dict(
            type="Xavier",
            layer="Conv2d",
            distribution="uniform",
        ),
    ):
        super(PSTRMapper, self).__init__(init_cfg)
        assert isinstance(in_channels, list)

        self.convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1 if i == 2 else kernel_size,
                padding=0 if i == 2 else (kernel_size - 1) // 2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            conv = DeformConv2dPack(
                out_channels,
                out_channels,
                3,
                padding=1,
                stride=2 if i == 0 else 1,
            )
            self.lateral_convs.append(l_conv)
            self.convs.append(conv)

    def forward(self, inputs):
        """Forward function."""
        n_convs = len(self.convs)
        assert len(inputs) == n_convs

        previous_shape = inputs[-2].shape[2:]

        outs = []
        for i in range(n_convs - 1, -1, -1):
            lateral = self.lateral_convs[i](inputs[i])

            out = (
                F.interpolate(lateral, size=previous_shape, mode="nearest")
                if i == n_convs - 1
                else lateral
            )
            out = self.convs[i](out)

            outs.append(out)

        return tuple(outs)

