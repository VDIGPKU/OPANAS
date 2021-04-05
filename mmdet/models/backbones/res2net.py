import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, constant_init, ConvModule, xavier_init, kaiming_init
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint

from ..builder import BACKBONES
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class Bottle2neck(_Bottleneck):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 scales=4,
                 base_width=26,
                 base_channels=64,
                 stage_type='normal',
                 sac=None,
                 **kwargs):
        """Bottle2neck block for Res2Net.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottle2neck, self).__init__(inplanes, planes, **kwargs)
        assert scales > 1, 'Res2Net degenerates to ResNet when scales = 1.'
        assert sac is None or isinstance(sac, dict)
        self.sac = sac
        self.with_sac = sac is not None

        width = int(math.floor(self.planes * (base_width / base_channels)))

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width * scales, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width * scales,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        if stage_type == 'stage' and self.conv2_stride != 1:
            self.pool = nn.AvgPool2d(
                kernel_size=3, stride=self.conv2_stride, padding=1)
        convs = []
        bns = []

        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)

        if self.with_sac:
            for i in range(scales - 1):
                convs.append(
                    build_conv_layer(
                        self.sac,
                        width,
                        width,
                        kernel_size=3,
                        stride=self.conv2_stride,
                        padding=self.dilation,
                        dilation=self.dilation,
                        bias=False))
                bns.append(
                    build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)
        elif not self.with_dcn or fallback_on_stride:
            for i in range(scales - 1):
                convs.append(
                    build_conv_layer(
                        self.conv_cfg,
                        width,
                        width,
                        kernel_size=3,
                        stride=self.conv2_stride,
                        padding=self.dilation,
                        dilation=self.dilation,
                        bias=False))
                bns.append(
                    build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            for i in range(scales - 1):
                convs.append(
                    build_conv_layer(
                        self.dcn,
                        width,
                        width,
                        kernel_size=3,
                        stride=self.conv2_stride,
                        padding=self.dilation,
                        dilation=self.dilation,
                        bias=False))
                bns.append(
                    build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width * scales,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.stage_type = stage_type
        self.scales = scales
        self.width = width
        delattr(self, 'conv2')
        delattr(self, self.norm2_name)



    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            spx = torch.split(out, self.width, 1)
            sp = self.convs[0](spx[0].contiguous())
            sp = self.relu(self.bns[0](sp))
            out = sp
            for i in range(1, self.scales - 1):
                if self.stage_type == 'stage':
                    sp = spx[i]
                else:
                    sp = sp + spx[i]
                sp = self.convs[i](sp.contiguous())
                sp = self.relu(self.bns[i](sp))
                out = torch.cat((out, sp), 1)

            if self.stage_type == 'normal' or self.conv2_stride == 1:
                out = torch.cat((out, spx[self.scales - 1]), 1)
            elif self.stage_type == 'stage':
                out = torch.cat((out, self.pool(spx[self.scales - 1])), 1)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out



class Res2Layer(nn.Sequential):
    """Res2Layer to build Res2Net style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottle2neck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        scales (int): Scales used in Res2Net. Default: 4
        base_width (int): Basic width of each scale. Default: 26
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 scales=4,
                 base_width=26,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False),
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1],
            )

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                scales=scales,
                base_width=base_width,
                stage_type='stage',
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    scales=scales,
                    base_width=base_width,
                    **kwargs))
        super(Res2Layer, self).__init__(*layers)


@BACKBONES.register_module()
class Res2Net(ResNet):
    arch_settings = {
        50: (Bottle2neck, (3, 4, 6, 3)),
        101: (Bottle2neck, (3, 4, 23, 3)),
        152: (Bottle2neck, (3, 8, 36, 3))
    }

    def __init__(self,
                 scales=4,
                 base_width=26,
                 style='pytorch',
                 deep_stem=True,
                 avg_down=True,
                 sac=None,
                 stage_with_sac=(False, False, False, False),
                 pretrained=None,
                 **kwargs):
        self.scales = scales
        self.base_width = base_width
        self.sac = sac
        self.stage_with_sac = stage_with_sac
        self.pretrained = pretrained
        super(Res2Net, self).__init__(
            style='pytorch', deep_stem=deep_stem, avg_down=avg_down, **kwargs)
        self.inplanes = self.stem_channels
        self.res_layers = []

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            sac = self.sac if self.stage_with_sac[i] else None
            if self.plugins is not None:
                stage_plugins = self.make_stage_plugins(self.plugins, i)
            else:
                stage_plugins = None
            planes = self.base_channels * 2 ** i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                sac=sac,
                plugins=stage_plugins)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

    def make_res_layer(self, **kwargs):
        return Res2Layer(
            scales=self.scales,
            base_width=self.base_width,
            base_channels=self.base_channels,
            **kwargs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.conv1.eval()
                for param in self.conv1.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.conv1 = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            #logger = get_root_logger()
            #load_checkpoint(self, filename=pretrained, map_location=lambda storage, loc: storage, strict=False, logger=logger)
            checkpoint = model_zoo.load_url(pretrained, map_location=lambda storage, loc: storage)
            self.load_state_dict(checkpoint, strict=False)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            # if self.dcn is not None:
            #     for m in self.modules():
            #         if isinstance(m, Bottle2neck) and hasattr(
            #                 m.conv2, 'conv_offset'):
            #             constant_init(m.conv2.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottle2neck):
                        constant_init(m.norm3, 0)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        outs = []
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)