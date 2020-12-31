import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


# VGG
# https://eda-ai-lab.tistory.com/408?category=764743
# https://github.com/AhnYoungBin/vgg16_pytorch/blob/master/vgg16_torch.ipynb
cfg = {
    'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}



class VGGNet(nn.Module):
    def __init__(self, vgg_name, num_classes=1000, init_weighs=True):
        super(VGGNet, self).__init__()
        self.vgg_name = vgg_name
        self.features = self._make_layers(cfg[vgg_name])    #  conv layer
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))  # average pooling....? 논문 구조 설명에서는 maxpooling으로 되어있고, 뒤에 avgpooling이 나오긴 함, 일단 선언만
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        # 굳이 model안에 넣을 이유가 있을까? -> train시 net선언하고 그 코드에서 따로 init_weight적용하는 것은?
        if init_weighs:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)   # softmax는? 보통 CrossEntropyLoss()에 포함시켜서 train에 사용한다
        return out

    # 논문에서는 따로 언급은 없지만, 참고한 구현 코드에서 사용
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(2, 2)]
            else:
                conv2d = nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, stride=1, padding=1)
                # batch_norm이 true
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]   # inplace 메모리 감소 -> 따로 찾아볼것
                else:
                    layers += [conv2d,
                               nn.ReLU(inplace=True)]
                in_channels = x

        return nn.Sequential(*layers)


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = VGGNet('VGG16').to(device)
    summary(net, (3, 224, 224), 30)


