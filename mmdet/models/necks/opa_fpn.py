import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init, kaiming_init

from mmdet.core import auto_fp16
from ..builder import NECKS
from .info_paths import OPS
import torch
from torch.autograd import Variable

@NECKS.register_module()
class OPA_FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 stack = 5,
                 edge_num = 0,
                 primitives = ['none', 'skip_connect', 'TD', 'BU', 'FS', 'SE'],
                 paths=None,
                 search=True,
                 ):
        super(OPA_FPN, self).__init__()
        assert isinstance(in_channels, list)
        #assert stack == edge_num or edge_num == stack * (stack + 1) // 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.act_cfg = act_cfg
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.stack = stack
        self.primitives = primitives
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
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.information_path = nn.ModuleList()

        self.features = nn.ModuleList()
        self.edge_num = edge_num
        self.paths = paths
        self.search = search
        if self.search:
            for i in range(self.stack):
                indgree = i + 1
                for edge in range(indgree):
                    self.features.append(nn.ModuleList())
                    for pre in self.primitives:
                        a = OPS[pre](out_channels, out_channels)
                        self.features[-1].append(a)
        else:
            for path in self.paths:
                self.features.append(OPS[path](out_channels, out_channels))

        self.topcontext = nn.Sequential(
            ConvModule(
                out_channels,
                out_channels,
                1,
                padding=0,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=self.act_cfg,
                inplace=False),
            nn.AdaptiveAvgPool2d(1))

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                extra_fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs, architecture=None):
        assert len(inputs) == len(self.in_channels)
        # build laterals

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        if self.add_extra_convs:
            laterals.append(self.fpn_convs[-2](laterals[-1]))
        used_backbone_levels = len(laterals)
        top = F.interpolate(self.topcontext(laterals[-1]), size=laterals[-1].shape[2:], mode='nearest')
        laterals[-1] = top + laterals[-1]

        info_paths = []
        info_paths.append(laterals)

        for step in range(self.stack):
            _step = step * (step + 1) // 2
            laterals_mid = [laterals[i].new_zeros(laterals[i].shape) for i in range(4)]
            for j in range(step+1):
                if self.search:
                    arch = architecture[_step+j]
                    temp = self.features[_step+j][arch](info_paths[j])
                else:
                    temp = self.features[_step+j](info_paths[j])

                for i in range(4):
                    laterals_mid[i] += temp[i]
            info_paths.append(laterals_mid)

        outs = info_paths[-1]

        for i in range(1, len(info_paths)-1):
            out = info_paths[i]
            for j in range(4):
                outs[j] += out[j]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.relu_before_extra_convs:
                    outs.append(self.fpn_convs[-1](F.relu(outs[-1])))
                else:
                    outs.append(self.fpn_convs[-1](outs[-1]))

        return tuple(outs)
