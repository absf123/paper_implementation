# https://pytorch.org/hub/pytorch_vision_inception_v3/
# https://oi.readthedocs.io/en/latest/computer_vision/cnn/inception-v2,v3.html
# v3는 v2와 구조는 동일하고 Hyperparameter만 변경 됨 (SGD->RMSProp)

# https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py Inception v3 -> 참고하기 v2랑 구조는 동일, paper랑 구조 다름 -> 이유는 모름 (버저닝이 굉장히 산만함)
# paper랑 너무 다름... 검색해도 v4 아니면 resnetv2,3 code만 나옴 -> 그냥 torchvision code로 만들어놓자

# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/inceptionv3.py 이 code가 그나마 비슷한듯 2021.1.9, paper구조랑은 다르긴함 -> 그래도 일단 torchvision code로 진행

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

# figure5, channel 조정은? -> torchvision code로 진행
class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_proj, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=48, kernel_size=1),
            conv_block(in_channels=48, out_channels=64, kernel_size=5, padding=1)
        )

        self.branch3x3 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=64, kernel_size=1),
            conv_block(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            conv_block(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        )

        self.branch_pool = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3 = self.branch3x3(x)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, 1)

# figure6 (n=7) -> not figure6, reference torchvision
class InceptionB(nn.Module):
    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = nn.Conv2d(in_channels=in_channels, out_channels=384, kernel_size=3, stride=2)

        self.branch3x3 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=64, kernel_size=1),
            conv_block(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            conv_block(in_channels=96, out_channels=96, kernel_size=3, stride=2),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)

        branch_pool = F.max_pool2d(x, kernel_siez=3, stride=2)

        outputs = [branch1x1, branch3x3, branch_pool]
        return torch.cat(outputs, 1)

# figure7, 이게 맞나...? -> not figure7, reference torchvision
class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = nn.Conv2d(in_channels=in_channels, out_channels=192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=c7, kernel_size=1),
            conv_block(in_channels=c7, out_channels=c7, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(in_channels=c7, out_channels=192, kernel_size=(7, 1), padding=(3, 0))
        )
        self.branch7x7db = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=c7, kernel_size=1),
            conv_block(in_channels=c7, out_channels=c7, kernel_size=(7, 1), padding=(3, 0)),
            conv_block(in_channels=c7, out_channels=c7, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(in_channels=c7, out_channels=c7, kernel_size=(7, 1), padding=(3, 0)),
            conv_block(in_channels=c7, out_channels=192, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7(x)
        branch7x7db = self.branch7x7db(x)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7db, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=192, kernel_size=1),
            conv_block(in_channels=192, out_channels=320, kernel_size=3, stride=2)
        )
        self.branch7x7 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=192, kernel_size=1),
            conv_block(in_channels=192, out_channels=192, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(in_channels=192, out_channels=192, kernel_size=(7, 1), padding=(3, 0)),
            conv_block(in_channels=192, out_channels=192, kernel_size=3, stride=2)
        )

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch7x7 = self.branch7x7(x)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch7x7, branch_pool]
        return torch.cat(outputs, 1)

# Sequential로 안한 이유가 이 module 때문인듯
class InceptionE(nn.Module):
    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


# auxiliary classifier, inceptionv2에서 바뀐점 반영x -> 2021.1.9 반영
class InceptionAux(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, conv_block = None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


# Inception-v3, 일단 aux는 output으로 x
class InceptionV3(nn.Module):
    # inception V2
    def __init__(self, num_classes=1000, aux_logits=True, transform_input = False, init_weights=True, blocks=None):
        super(InceptionV3, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux]

        if init_weights is None:
            # True : 구 버전 가중치 초기화, False : 최신 버전 가중치 초기화 -> fine tuning...?
            warnings.warn('The default weight initialization of GoogleNet will be changed in future releases of \
                             torchvision. If you wish to keep the old behavior (which leads to long initialization times due to scipy/scipy#11299), please set init_weights=True.',
                          FutureWarning)
            init_weights = True

        # https://wikidocs.net/21050 assert, assert는 뒤의 조건이 True가 아니면 AssertError를 발생한다., 방어적 프로그래밍
        assert len(blocks) == 7
        conv_block = blocks[0]
        inception_a_block = blocks[1]
        inception_b_block = blocks[2]
        inception_c_block = blocks[3]
        inception_d_block = blocks[4]
        inception_e_block = blocks[5]
        inception_aux_block = blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a_block(192, pool_proj=32)
        self.Mixed_5c = inception_a_block(256, pool_proj=64)
        self.Mixed_5d = inception_a_block(288, pool_proj=64)
        self.Mixed_6a = inception_b_block(288)
        self.Mixed_6b = inception_c_block(768, channels_7x7=128)
        self.Mixed_6c = inception_c_block(768, channels_7x7=160)
        self.Mixed_6d = inception_c_block(768, channels_7x7=160)
        self.Mixed_6e = inception_c_block(768, channels_7x7=192)
        self.AuxLogits = None
        if aux_logits:
            self.AuxLogits = inception_aux_block(768, num_classes)
        self.Mixed_7a = inception_d_block(768)
        self.Mixed_7b = inception_e_block(1280)
        self.Mixed_7c = inception_e_block(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)

        if init_weights:
            for m in self.modules():
                # isinstance(인스터스, class/data type) 함수로 특정 class/data type인지 확인
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    # truncated normal distribution 생성
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)

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

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = InceptionV3(num_classes=1000).to(device)
    summary(net, (3, 299, 299), 30)


