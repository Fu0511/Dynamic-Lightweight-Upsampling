from mmcv.ops import deform_conv, roi_align
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer
from mmengine.model import xavier_init
from mmcv.ops.carafe import CARAFEPack
from custom.ops.dlu import DLUPack
from mmengine.model import BaseModule, ModuleList

# from mmdet.models.builder import NECKS

from mmengine.registry import MODELS

import torch.nn.functional as F

@MODELS.register_module()
class UNIFIED_FPN(BaseModule):
    """FPN_CARAFE is a more flexible implementation of FPN. It allows more
    choice for upsample methods during the top-down pathway.

    It can reproduce the performance of ICCV 2019 paper
    CARAFE: Content-Aware ReAssembly of FEatures
    Please refer to https://arxiv.org/abs/1905.02188 for more details.

    Args:
        in_channels (list[int]): Number of channels for each input feature map.
        out_channels (int): Output channels of feature pyramids.
        num_outs (int): Number of output stages.
        start_level (int): Start level of feature pyramids.
            (Default: 0)
        end_level (int): End level of feature pyramids.
            (Default: -1 indicates the last level).
        norm_cfg (dict): Dictionary to construct and config norm layer.
        activate (str): Type of activation function in ConvModule
            (Default: None indicates w/o activation).
        order (dict): Order of components in ConvModule.
        upsample (str): Type of upsample layer.
        upsample_cfg (dict): Dictionary to construct and config upsample layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=None,
                 act_cfg=None,
                 order=('conv', 'norm', 'act'),
                 upsample_cfg=dict(
                     type='carafe',
                     up_kernel=5,
                     up_group=1,
                     encoder_kernel=3,
                     encoder_dilation=1),
                 init_cfg=None):
        # assert init_cfg is None, 'To prevent abnormal initialization ' \
        #                          'behavior, init_cfg is not allowed to be set'
        super(UNIFIED_FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_bias = norm_cfg is None
        self.upsample_cfg = upsample_cfg.copy()
        self.upsample = self.upsample_cfg.get('type')
        self.relu = nn.ReLU(inplace=False)

        self.order = order
        assert order in [('conv', 'norm', 'act'), ('act', 'conv', 'norm')]

        assert self.upsample in [
            'nearest', 'bilinear', 'deconv', 'pixel_shuffle', 'carafe', 'dlu', None
        ]
        if self.upsample in ['deconv', 'pixel_shuffle']:
            assert hasattr(
                self.upsample_cfg,
                'upsample_kernel') and self.upsample_cfg.upsample_kernel > 0
            self.upsample_kernel = self.upsample_cfg.pop('upsample_kernel')
        elif self.upsample == 'carafev3':
                self.up_kernel = self.upsample_cfg.get('up_kernel')
                self.up_group = self.upsample_cfg.get('up_group')

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()
        self.upsample_modules = ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False,
                order=self.order)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False,
                order=self.order)
            if i != self.backbone_end_level - 1:
                upsample_cfg_ = self.upsample_cfg.copy()
                if self.upsample == 'deconv':
                    upsample_cfg_.update(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=self.upsample_kernel,
                        stride=2,
                        padding=(self.upsample_kernel - 1) // 2,
                        output_padding=(self.upsample_kernel - 1) // 2)
                elif self.upsample == 'pixel_shuffle':
                    upsample_cfg_.update(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        scale_factor=2,
                        upsample_kernel=self.upsample_kernel)
                elif self.upsample == 'carafe' or self.upsample == 'dlu':
                    upsample_cfg_.update(channels=out_channels, scale_factor=2)
                else:
                    # suppress warnings
                    align_corners = (None
                                     if self.upsample == 'nearest' else False)
                    upsample_cfg_.update(
                        scale_factor=2,
                        mode=self.upsample,
                        align_corners=align_corners)
                upsample_module = build_upsample_layer(upsample_cfg_)
                self.upsample_modules.append(upsample_module)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_out_levels = (
            num_outs - self.backbone_end_level + self.start_level)
        if extra_out_levels >= 1:
            for i in range(extra_out_levels):
                in_channels = (
                    self.in_channels[self.backbone_end_level -
                                     1] if i == 0 else out_channels)
                extra_l_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=self.with_bias,
                    act_cfg=act_cfg,
                    inplace=False,
                    order=self.order)
                if self.upsample == 'deconv':
                    upsampler_cfg_ = dict(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=self.upsample_kernel,
                        stride=2,
                        padding=(self.upsample_kernel - 1) // 2,
                        output_padding=(self.upsample_kernel - 1) // 2)
                elif self.upsample == 'pixel_shuffle':
                    upsampler_cfg_ = dict(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        scale_factor=2,
                        upsample_kernel=self.upsample_kernel)
                elif self.upsample == 'carafe' or self.upsample == 'dlu':
                    upsampler_cfg_ = dict(
                        channels=out_channels,
                        scale_factor=2,
                        # groups=out_channels,
                        **self.upsample_cfg)
                elif self.upsample == 'carafev3':
                    upsampler_cfg_ = dict(
                        channels=out_channels,
                        **self.upsample_cfg)

                # 自定义添加
                elif self.upsample == 'deform_se4':
                    upsampler_cfg_ = dict(
                        scale_factor=2,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        groups=out_channels,
                        **self.upsample_cfg)
                elif self.upsample == 'deform_region':
                    upsampler_cfg_ = dict(
                        scale_factor=2,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        groups=out_channels,
                        **self.upsample_cfg)
                elif self.upsample == 'deform_weighted_kernel':
                    upsampler_cfg_ = dict(
                        scale_factor=2,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        groups=out_channels,
                        **self.upsample_cfg)
                elif self.upsample == 'deform_weighted_kernel_v2':
                    upsampler_cfg_ = dict(
                        scale_factor=2,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        groups=out_channels,
                        **self.upsample_cfg)
                elif self.upsample == 'deform_weighted_kernel_v2_cascade':
                    # 有两种实现方式，指数增长与线性增长，此处使用指数增长
                    roi_size = self.upsample_cfg["roi_size"] * 2 ** (i + self.backbone_end_level - self.start_level - 1)
                    upsampler_cfg_ = dict(
                        scale_factor=2,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        groups=out_channels,
                        **self.upsample_cfg)
                    upsampler_cfg_.update(roi_size=roi_size, type="deform_weighted_kernel_v2")
                elif self.upsample == 'conv_offset_2d_upsample':
                    upsampler_cfg_ = dict(
                        scale_factor=2,
                        in_channels=out_channels,
                        **self.upsample_cfg)

                else:
                    # suppress warnings
                    align_corners = (None
                                     if self.upsample == 'nearest' else False)
                    upsampler_cfg_ = dict(
                        scale_factor=2,
                        mode=self.upsample,
                        align_corners=align_corners)
                upsampler_cfg_['type'] = self.upsample #好像有点多余
                upsample_module = build_upsample_layer(upsampler_cfg_)
                extra_fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.with_bias,
                    act_cfg=act_cfg,
                    inplace=False,
                    order=self.order)
                self.upsample_modules.append(upsample_module)
                self.fpn_convs.append(extra_fpn_conv)
                self.lateral_convs.append(extra_l_conv)
        if "cascade" in self.upsample:
            self.upsample_modules = self.upsample_modules[::-1]

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of module."""
        super(UNIFIED_FPN, self).init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()
            elif isinstance(m, DLUPack):
                m.init_weights()

    def slice_as(self, src, dst):
        """Slice ``src`` as ``dst``

        Note:
            ``src`` should have the same or larger size than ``dst``.

        Args:
            src (torch.Tensor): Tensors to be sliced.
            dst (torch.Tensor): ``src`` will be sliced to have the same
                size as ``dst``.

        Returns:
            torch.Tensor: Sliced tensor.
        """
        assert (src.size(2) >= dst.size(2)) and (src.size(3) >= dst.size(3))
        if src.size(2) == dst.size(2) and src.size(3) == dst.size(3):
            return src
        else:
            return src[:, :, :dst.size(2), :dst.size(3)]

    def tensor_add(self, a, b):
        """Add tensors ``a`` and ``b`` that might have different sizes."""
        if a.size() == b.size():
            c = a + b
        else:
            c = a + self.slice_as(b, a)
        return c

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i <= self.backbone_end_level - self.start_level:
                input = inputs[min(i + self.start_level, len(inputs) - 1)]
            else:
                input = laterals[-1]
            lateral = lateral_conv(input)
            laterals.append(lateral)

        # build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            if self.upsample is not None:
                    upsample_feat = self.upsample_modules[i - 1](laterals[i])
            else:
                upsample_feat = laterals[i]
            laterals[i - 1] = self.tensor_add(laterals[i - 1], upsample_feat)

        # build outputs
        num_conv_outs = len(self.fpn_convs)
        outs = []
        for i in range(num_conv_outs):
            out = self.fpn_convs[i](laterals[i])
            outs.append(out)
        return tuple(outs)


if __name__ == "__main__":
    import torch
    from custom.ops.carafe import CustomCARAFEPack
    # import custom.ops.deform_upsample_block_adaptivesoftmax_se4_usegroup
    in_channels = [10,20,40,80]
    fpn = UNIFIED_FPN(
                in_channels=in_channels,
                out_channels=256,
                num_outs=5,
                start_level=0,
                end_level=-1,
                norm_cfg=None,
                act_cfg=None,
                order=('conv', 'norm', 'act'),
                upsample_cfg=dict(
                    type='custom_carafe',
                    up_kernel=5,
                    up_group=1,
                    encoder_kernel=3,
                    encoder_dilation=1,
                    compressed_channels=64)).cuda()
    inputs = []
    for i in range(4):
        input_tensor = torch.randn(3,in_channels[i],10*2**(3-i), 10*2**(3-i)).cuda()
        inputs.append(input_tensor)

    outputs = fpn(inputs)
    
    for j in outputs:
        print(j.shape)
