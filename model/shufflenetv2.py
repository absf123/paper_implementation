# 원본 코드 : https://github.com/ericsun99/Shufflenet-v2-Pytorch/blob/master/ShuffleNetV2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width, depth = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width, depth)
    # print(x.shape) # torch.Size([2, 2, 24, 7, 8, 7])
    x = torch.transpose(x, 1, 2).contiguous()   # x[1], x[2] 바꿈
    # print(x.shape) # torch.Size([2, 24, 2, 7, 8, 7])
    # flatten
    x = x.view(batchsize, -1, height, width, depth)
    # print(x.shape)  # torch.Size([2, 48, 7, 8, 7]), group=2를 하나로 24->48
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv3d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm3d(inp),
                # pw-linear
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)
    # 여기 부분을 이해 못함
    def forward(self, x):
        # fig3.(c)
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :, :]   # x.shape[1] -> channel
            x2 = x[:, (x.shape[1] // 2):, :, :, :]
            out = self._concat(x1, self.banch2(x2))
        # fig3.(d)
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, in_ch=1, input_size=224, width_mult=1.):
        super(ShuffleNetV2, self).__init__()

        assert input_size % 32 == 0

        # self.stage_repeats = [4, 8, 4]
        self.stage_repeats = [2, 4, 2]
        self.in_ch = in_ch
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:  # x0.5
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:  # x1.0
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(self.in_ch, input_channel, 2)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    # inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))    # stride=2
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool3d(int(input_size / 32)))

        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

def shufflenetv2(width_mult=1.):
    model = ShuffleNetV2(width_mult=width_mult)
    return model

if __name__ == "__main__":
    from torchsummary import summary
    """Testing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShuffleNetV2(n_class=2, in_ch=1, input_size=64, width_mult=0.5).to(device)
    # print(model)
    summary(model, (1, 56, 64, 56), 20)