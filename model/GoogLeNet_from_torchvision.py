# torchvision package code....

import torch
from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
from torch import Tensor
import numpy
import warnings
from torch import Tensor
# from .utils import load_state_dict_from_url

__all__ = ['GoogLeNet', 'googlenet', "GoogLeNetOutputs", "_GoogLeNetOutputs"]

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


def googlenet(pretrained=False, progress=True, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = GoogLeNet(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['googlenet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None
            model.aux2 = None
        return model

    return GoogLeNet(**kwargs)


# https://github.com/jhcha08/Implementation_DeepLearningPaper/blob/master/%EC%8A%A4%ED%84%B0%EB%94%94%2020200205%20-%20CNN.%20Inception.ipynb
# https://deep-learning-study.tistory.com/389
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, init_weights=None, blocks=None):
        super(GoogLeNet, self).__init__()
        if blocks is None:
            # 정의된 3개의 block 가져오기
            blocks = [BasicConv2d, Inception_module, InceptionAux]
        if init_weights is None:
            # True : 구 버전 가중치 초기화, False : 최신 버전 가중치 초기화
            warnings.warn('The default weight initialization of GoogleNet will be changed in future releases of \
                          torchvision. If you wish to keep the old behavior (which leads to long initialization times due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True

        assert len(blocks) == 3
        conv_block = blocks[0]  # BasicConv2
        inception_block = blocks[1] # Inception_block
        inception_aux_block = blocks[2]  # InceptionAux

        self.auc_logits = aux_logits
        self.transfor_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        # ceil_mode = True : 천장함수 이용
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 95, 208, 16 ,48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes)
            self.aux2 = inception_aux_block(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()


    # 가중치 초기화 -> 논문에 있는 건가?
    def _initialize_weights(self):
        for m in self.modules():
            # isinstance(인스터스, class/data type) 함수로 특정 class/data type인지 확인
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
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

    def _transform_input(self, x):
    # type: #(Tensor) -> Tensor
        if self.transform_input:
            # unsqueeze(input, dimp) input을 dim 차원으로 변경
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) * (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) * (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) * (0.406 - 0.5) / 0.5
            # x를 하나로 통합
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # type: #(Tensor) -> Tensor
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 26 x 26
        x = self.inception3a(x)
        # N x 256 x 26 x 26
        x = self.inception3b(x)
        # N x 460 x 26 x 26
        x = self.maxpool3(x)
        # N x 460 x 26 x 26
        x = self.inception4a(x)
        # N x 512 x 14 x 14

        # torch.jit.annotate(Optional[Tensor], None)
        aux1 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux1 is not None:
            # training에서만 aux1 적용
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 526 x 14 x 14
        aux2 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux2, aux1

    @torch.jit.unused
    # aux2, aux1를 적용한 출력값 반환하기
    def eager_outputs(self, x, aux2, aux1):
        # type: #(Tensor, Optional[Tensor], Optional[Tensor]) -> GoogLeNetOutputs
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x

    def forward(self, x):
        # type: #(Tensor) -> GoogLeNetOutputs
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)

# inception module, branch 4개
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

if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GoogLeNet(num_classes=1000, aux_logits=True, transform_input=False, init_weights=None, blocks=None).to(device)
    summary(net, (3, 224, 224), 30)

# 하긴 했는데, 좀 더 simple 하게 할 수 없을까...?
# code 출처 : https://paperswithcode.com/method/googlenet
# 12.31에 다시 공부