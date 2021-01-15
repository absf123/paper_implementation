# https://arxiv.org/pdf/1704.04861.pdf%EF%BB%BF
# https://github.com/wjc852456/pytorch-mobilenet-v1/blob/master/benchmark.py 참고
# https://ysbsb.github.io/cnn/2020/02/20/MobileNet.html

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable



# group 설명 : https://gaussian37.github.io/dl-pytorch-conv2d/
class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()

        def conv_basic(in_ch, out_ch, stride):
            conv_basic = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            return conv_basic

        def conv_dw(in_ch, out_ch, stride_dw):
            conv_depthwise = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=stride_dw, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, stride=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            return conv_depthwise

        self.model = nn.Sequential(
            conv_basic(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),   # paper에서는 stride=2 로 나와있는데 그러면 계산이 안 됨 -> 1
            nn.AvgPool2d(7)
        )


        # debugging용
        # self.conv1 = conv_basic(3, 32, 2)
        # self.conv_dw1 = conv_dw(32, 64, 1)
        # self.conv_dw2 = conv_dw(64, 128, 2)
        # self.conv_dw3 = conv_dw(128, 128, 1)
        # self.conv_dw4 = conv_dw(128, 256, 2)
        # self.conv_dw5 = conv_dw(256, 256, 1)
        # self.conv_dw6 = conv_dw(256, 512, 2)
        # self.conv_dw7 = conv_dw(512, 512, 1)
        # self.conv_dw8 = conv_dw(512, 512, 1)
        # self.conv_dw9 = conv_dw(512, 512, 1)
        # self.conv_dw10 = conv_dw(512, 512, 1)
        # self.conv_dw11 = conv_dw(512, 512, 1)
        # self.conv_dw12 = conv_dw(512, 1024, 2)
        # self.conv_dw13 = conv_dw(1024, 1024, 2)  # paper에서는 stride=2 로 나와있는데 그러면 계산이 안 됨 -> 1, stride가 1이어야 output feature map이 7x7
        # self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)

        # debugging용, 논문에서 Table1. 제일 마지막 Conv dw를 보면 s2로 stride가 2로 나와있는데, 2이면 마지막 feature map이 4x4로 출력 됨. 오타 추정 (s2가 아니라 s1이 되어야 할듯)
        # x = self.conv1(x)
        # print(x.shape)
        # x = self.conv_dw1(x)
        # print(x.shape)
        # x = self.conv_dw2(x)
        # print(x.shape)
        # x = self.conv_dw3(x)
        # print(x.shape)
        # x = self.conv_dw4(x)
        # print(x.shape)
        # x = self.conv_dw5(x)
        # print(x.shape)
        # x = self.conv_dw6(x)
        # print(x.shape)
        # x = self.conv_dw7(x)
        # print(x.shape)
        # x = self.conv_dw8(x)
        # print(x.shape)
        # x = self.conv_dw9(x)
        # print(x.shape)
        # x = self.conv_dw10(x)
        # print(x.shape)
        # x = self.conv_dw11(x)
        # print(x.shape)
        # x = self.conv_dw12(x)
        # print(x.shape)
        # x = self.conv_dw13(x)
        # print(x.shape)

        # print(x.shape)
        x = x.view(-1, 1024)
        out = self.fc(x)
        return out

# 정확한 비교인지는 모르겠음
"""
  resnet18 : 0.017108
   alexnet : 0.004051
     vgg16 : 0.008626
squeezenet : 0.015003
 mobilenet : 0.024606
"""
def speed(model, name):
    t0 = time.time()
    input = torch.rand(1, 3, 224, 224).cuda()
    input = Variable(input, volatile=True)
    t1 = time.time()

    model(input)
    t2 = time.time()

    model(input)
    t3 = time.time()

    print('%10s : %f' % (name, t3 - t2))


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MobileNet(num_classes=1000).to(device)
    summary(net, (3, 224, 224), 30)


    resnet18 = models.resnet18().cuda()
    alexnet = models.alexnet().cuda()
    vgg16 = models.vgg16().cuda()
    squeezenet = models.squeezenet1_0().cuda()
    mobilenet = MobileNet().cuda()

    speed(resnet18, 'resnet18')
    speed(alexnet, 'alexnet')
    speed(vgg16, 'vgg16')
    speed(squeezenet, 'squeezenet')
    speed(mobilenet, 'mobilenet')