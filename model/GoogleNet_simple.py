# https://github.com/jhcha08/Implementation_DeepLearningPaper/blob/master/%EC%8A%A4%ED%84%B0%EB%94%94%2020200205%20-%20CNN.%20Inception.ipynb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy

# GoogLeNet
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, initail):





class Inception_module(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, conv_block=None):
        super(Inception_module, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            # kernel_size=5가 아닌 3인 이유는 오류라고 합니다.
            # 실수로 3으로 설정하여 학습을 했고 5로 바꾸면 다시 학습을 시켜야 하므로
            # 이대로 냅두었다고 설명함
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        # torch.cat 으로 outputs를 하나로 통합
        return torch.cat(outputs, 1)

# Auxiliary classifier 정의하기
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 526 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x  512 x 4 x 4, aux2: N x 526 x 4 x 4
        x = self.conv(x)
        # N x 126 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2046
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_claseese)

        return x





# 기본 Convolution layer 정의하기 conv2d-bn-relu
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)