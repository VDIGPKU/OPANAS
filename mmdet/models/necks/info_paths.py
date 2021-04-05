import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init, normal_init
import torch
from mmdet.core import auto_fp16
from ..builder import NECKS
from mmcv.ops import DeformConv2d, deform_conv2d
from torch.nn.modules.utils import _pair
from torch.nn import init as init
OPS = {
    'none': lambda in_channels, out_channels: None_(),
    'skip_connect' :lambda in_channels, out_channels: Skip_(),
    'TD' : lambda  in_channels, out_channels: Top_down(in_channels, out_channels),
    'BU' : lambda in_channels, out_channels: Bottom_up(in_channels, out_channels),
    'FS' : lambda in_channels, out_channels: Fuse_split(in_channels, out_channels),
    'SE': lambda in_channels, out_channels: Scale_equalize(in_channels, out_channels),

}


class Top_down(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super(Top_down, self).__init__()
        self.tdm_convs = nn.ModuleList()
        for i in range(4):
            tdm_conv = deform_conv(
                in_channels,
                out_channels,
                3,
                padding=1)
            self.tdm_convs.append(tdm_conv)


    def forward(self, inputs):
        # build top-down path

        topdown = []
        topdownconv = self.tdm_convs[-1](1, inputs[-1])
        if topdownconv.shape[2:] != inputs[-1].shape:
            topdownconv = F.interpolate(topdownconv, size=inputs[-1].shape[2:], mode='nearest')

        topdown.append(topdownconv)
        for i in range(3, 0, -1):
            temp = self.tdm_convs[i - 1](i - 1, inputs[i - 1] + F.interpolate(
                topdownconv.clone(), size=inputs[i - 1].shape[2:], mode='nearest'))
            topdown.insert(0, temp)
            topdownconv = temp
        return topdown





class Bottom_up(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super(Bottom_up, self).__init__()
        self.bun_convs = nn.ModuleList()
        for i in range(4):
            bun_conv = deform_conv(
                in_channels,
                out_channels,
                3,
                padding=1,
            )
            self.bun_convs.append(bun_conv)


    def forward(self, inputs):
        # build bottom-up path

        botomup = []
        for i in range(4):
            if i == 0:
                bum = inputs[0]
            elif i == 3:
                bb = F.max_pool2d(botomup[-1].clone(), 2, stride=2)
                if bb.shape[2:] != inputs[-1].shape[2:]:
                    bb = F.interpolate(
                        bb, size=inputs[-1].shape[2:], mode='nearest')
                bum = bb + inputs[-1]
            else:
                bum = inputs[i] + F.max_pool2d(botomup[i - 1].clone(), 2, stride=2)

            botomup.append(self.bun_convs[i](i, bum))

        return botomup




class Fuse_split(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super(Fuse_split, self).__init__()
        self.fuse = nn.ModuleList([deform_conv(
            out_channels * 2,
            out_channels,
            3,
            padding=1
        )] * 2)
        self.in_channels = in_channels
        self.out_channels = out_channels



    def forward(self, inputs):
        # build fusing-splitting path

        fussplit = []
        fuse1 = inputs[1] + F.max_pool2d(inputs[0], 2, stride=2)
        fuse2 = F.interpolate(
            inputs[-1], size=inputs[2].shape[2:], mode='nearest') + inputs[2]
        fuseconv1 = self.fuse[0](1, torch.cat([fuse1.clone(), F.interpolate(
            fuse2.clone(), size=fuse1.shape[2:], mode='nearest')], 1))
        fuseconv2 = self.fuse[1](1, torch.cat([F.max_pool2d(fuse1.clone(), 2, stride=2), fuse2.clone()], 1))

        fussplit.append(F.interpolate(
            fuseconv1.clone(), size=inputs[0].shape[2:], mode='nearest'))
        fussplit.append(fuseconv1)
        fussplit.append(fuseconv2)
        fussplit.append(F.max_pool2d(fuseconv2.clone(), 2, stride=2, ceil_mode=False))
        if fussplit[-1].shape[2:] != inputs[-1].shape[2:]:
            fussplit[-1] = F.interpolate(fussplit[-1].clone(), size=inputs[-1].shape[2:], mode='nearest')
        return fussplit




class None_(nn.Module):
      def __init__(self,
                   ):
            super(None_, self).__init__()

            self.size =0
            self.fp = 0
      def forward(self, inputs):

            outs = []
            for x in inputs:
              outs.append(x.new_zeros(x.shape))
            return outs

class Skip_(nn.Module):
      def __init__(self):
            super(Skip_, self).__init__()

            self.size = 0
            self.fp = 0
      def forward(self, inputs):
        return inputs

class Scale_equalize(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        kernel_size=[3, 3, 3],
        dilation=[1, 1, 1],
        groups=[1, 1, 1],

    ):
        super(Scale_equalize, self).__init__()

        self.Pconv = nn.ModuleList()
        self.Pconv.append(
            deform_conv(in_channels,
                      out_channels,
                      kernel_size=kernel_size[0],
                      dilation=dilation[0],
                      groups=groups[0],
                      padding=(kernel_size[0] + (dilation[0] - 1) * 2) // 2))
        self.Pconv.append(
            deform_conv(in_channels,
                      out_channels,
                      kernel_size=kernel_size[1],
                      dilation=dilation[1],
                      groups=groups[1],
                      padding=(kernel_size[1] + (dilation[1] - 1) * 2) // 2))

        self.Pconv.append(
            deform_conv(in_channels,
                      out_channels,
                      kernel_size=kernel_size[2],
                      dilation=dilation[2],
                      groups=groups[2],
                      padding=(kernel_size[2] + (dilation[2] - 1) * 2) // 2,
                      stride=2))

        self.relu = nn.ReLU()
        self.init_weights()



    def init_weights(self):
        for m in self.Pconv:
            init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        next_x = []
        for level, feature in enumerate(x):

            temp_fea = self.Pconv[1](level, feature)
            if level > 0:
                temp_fea += self.Pconv[2](level, x[level - 1])
            if level < len(x) - 1:
                temp_fea += F.interpolate(
                    self.Pconv[0](level, x[level + 1]),
                    size=temp_fea.size()[2:], mode='nearest'
                )
            next_x.append(temp_fea)

        next_x = [self.relu(item) for item in next_x]
        return next_x




class deform_conv(DeformConv2d):
    def __init__(self, *args,  **kwargs,):
        super(deform_conv, self).__init__( *args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 2 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.start_level = 1

    def init_offset(self):

        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, i, x):

        if i < self.start_level:
            return torch.nn.functional.conv2d(x, self.weight, bias=self.bias, stride=self.stride,
                                              padding=self.padding,
                                              dilation=self.dilation, groups=self.groups)

        offset = self.conv_offset(x)
        return deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups) + self.bias.unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1)



