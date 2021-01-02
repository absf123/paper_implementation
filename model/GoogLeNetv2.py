# https://pytorch.org/hub/pytorch_vision_inception_v3/
# https://oi.readthedocs.io/en/latest/computer_vision/cnn/inception-v2,v3.html
# v3는 v2와 구조는 동일하고 Hyperparameter만 변경 됨 (SGD->RMSProp)

# https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py Inception v3 -> 참고하기 v2랑 구조는 동일

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats
import warnings

# channel, kernel_size, stride, padding...-> 논문이랑 다른 코드 참고하면서 수정 필요, 일단 뼈대만 완성 2021.1.2
# BN_auxiliary 아직 반영x

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.bn(self.conv(x))

        return F.relu(x, inplace=True)

# figure5, channel 조정은...? output size보고 맞춰야하나? -> 다른 코드 확인
class InceptionV2_A(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3red_, ch3x3_, pool_proj, conv_block=None):
        super(InceptionV2_A, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = nn.Conv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1),
            conv_block(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=ch3x3red_, kernel_size=1),
            conv_block(in_channels=ch3x3red_, out_channels=ch3x3red_, kernel_size=3, padding=1),
            conv_block(in_channels=ch3x3red_, out_channels=ch3x3_, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels=in_channels, out_channels=pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# figure6 (n=7)
class InceptionV2_B(nn.Module):
    def __init__(self, in_channels, ch1x1, ch1xNred, chNx1red, ch3x3, ch3x3red_, ch3x3_, pool_proj, conv_block=None):
        super(InceptionV2_B, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = nn.Conv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=ch1xNred, kernel_size=1),
            conv_block(in_channels=ch1xNred, out_channels=chNx1red, kernel_size=(1, 7), padding=1),
            conv_block(in_channels=chNx1red, out_channels=ch1xNred, kernel_size=(7, 1), padding=1),
            conv_block(in_channels=ch1xNred, out_channels=chNx1red, kernel_size=(1, 7), padding=1),
            conv_block(in_channels=chNx1red, out_channels=ch1xNred, kernel_size=(7, 1), padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=ch3x3red_, kernel_size=1),
            conv_block(in_channels=ch3x3red_, out_channels=ch3x3red_, kernel_size=3, padding=1),
            conv_block(in_channels=ch3x3red_, out_channels=ch3x3_, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels=in_channels, out_channels=pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# figure7, 이게 맞나...?
class InceptionV2_C(nn.Module):
    def __init__(self, in_channels, ch1x1, ch1x3red, ch3x1red, ch1x3, ch3x1, ch3x3, ch3x3red, pool_proj, conv_block=None):
        super(InceptionV2_C, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = nn.Conv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1)

        self.branch2a = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1),
            conv_block(in_channels=ch3x3red, out_channels=ch1x3red, kernel_size=3, padding=1),
            conv_block(in_channels=ch1x3red, out_channels=ch1x3, kernel_size=(1, 3), padding=1)
        )
        self.branch2b = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1),
            conv_block(in_channels=ch3x3red, out_channels=ch3x1red, kernel_size=3, padding=1),
            conv_block(in_channels=ch3x1red, out_channels=ch3x1, kernel_size=(3, 1), padding=1)
        )

        self.branch3a = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=ch1x3red, kernel_size=1),
            conv_block(in_channels=ch1x3red, out_channels=ch1x3, kernel_size=(1, 3), padding=1)
        )
        self.branch3b = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=ch3x1red, kernel_size=1),
            conv_block(in_channels=ch3x1red, out_channels=ch3x1, kernel_size=(3, 1), padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels=in_channels, out_channels=pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2a = self.branch2a(x)
        branch2b = self.branch2b(x)
        branch3a = self.branch3a(x)
        branch3b = self.branch3b(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2a, branch2b, branch3a, branch3b, branch4]
        return torch.cat(outputs, 1)

# auxiliary classifier, inceptionv2에서 바뀐점 반영x
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels=in_channels, out_channels=128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)

        return x


class googLeNet_InceptionV2(nn.Module):
    # inception V2
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=True, blocks=None):
        super(googLeNet_InceptionV2, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, InceptionV2_A, InceptionV2_B, InceptionV2_C, InceptionAux]

        if init_weights is None:
            # True : 구 버전 가중치 초기화, False : 최신 버전 가중치 초기화 -> fine tuning...?
            warnings.warn('The default weight initialization of GoogleNet will be changed in future releases of \
                             torchvision. If you wish to keep the old behavior (which leads to long initialization times due to scipy/scipy#11299), please set init_weights=True.',
                          FutureWarning)
            init_weights = True

        # https://wikidocs.net/21050 assert, assert는 뒤의 조건이 True가 아니면 AssertError를 발생한다., 방어적 프로그래밍
        assert len(blocks) == 5
        conv_block = blocks[0]
        InceptionV2_A_block = blocks[1]
        InceptionV2_B_block = blocks[2]
        InceptionV2_C_block = blocks[3]
        Inception_aux_block = blocks[4]

        self.aux_logits = aux_logits
        # transform_input = transform_input

        self.conv1 = conv_block(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = conv_block(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = conv_block(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, 2)
        self.conv4 = conv_block(in_channels=64, out_channels=80, kernel_size=3, stride=1)
        self.conv5 = conv_block(in_channels=64, out_channels=192, kernel_size=3, stride=2)
        self.conv6 = conv_block(in_channels=192, out_channels=288, kernel_size=3, stride=1)

        # 채널 수는 임시로 사용->논문보고 다시 맞춰야함
        self.inception_A = InceptionV2_A(288, 64, 96, 128, 16, 32, 32)
        self.inception_B = InceptionV2_B(288, 128, 128, 182, 32, 96, 64)
        self.inception_C = InceptionV2_C(288, 128, 128, 182, 16, 32, 64)


    def _initialize_weights(self):
        for m in self.modules():
            # isinstance(인스터스, class/data type) 함수로 특정 class/data type인지 확인
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # truncated normal distribution 생성
                X = stats.truncnorm(-2, 2, scale=0.01)

                # torch.as_tensor(data) data를 dtype tensor로 변환
                # .numel()는 입력 tensor의 전체 수를 반환
                # .rvs는 랜덤 표본 생성하는 사이킷런 함수
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)

                # value의 차원을 m.weight.size로 변경
                values = values.view(m.weight.size())

                with torch.no_grad():
                    m.weight.copy_(values)

            # BatchNorm2d에서는 weight=1, bias=0으로 설정
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):






