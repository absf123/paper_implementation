# brainNetCNN
# https://github.com/nicofarr/brainnetcnnVis_pytorch/blob/master/BrainNetCnnGoldMSI.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# example로 channel이랑 depth를 지정.. -> dataset 구성 마다 다를듯
# 위 code와 paper figure overview랑 살짝 차이가 있음(E2E1->E2E2 E2N->N2G channel수, dropout 위치&수), 그리고 batchnorm은 안쓰는듯

# BrainNetCNN
class E2EBlock(nn.Module):
    """ E2EBlock """
    def __init__(self, in_ch, out_ch, example, bias=False):
        super(E2EBlock, self).__init__()
        self.d = example.size(3) # tensor index size

        self.conv1 = nn.Conv2d(in_ch, out_ch, (1, self.d), bias=False)  # row
        self.conv2 = nn.Conv2d(in_ch, out_ch, (self.d, 1), bias=False)  # col

    def forward(self, x):
        row_filter = self.conv1(x)
        col_filter = self.conv2(x)
        return torch.cat([row_filter]*self.d, 3) + torch.cat([col_filter]*self.d, 2)     # NxCxHxW, row_filter는 W 방향으로 cat,  col_filter는 H 방향으로 cat



class BrainNetCNN(nn.Module):
    def __init__(self, example, num_classes=2):
        super(BrainNetCNN, self).__init__()
        self.d = example.size(3)   # tensor index size, dataset 마다 좀 다를 듯, brain region 개수(functional connectivity matrix map)

        self.E2Elayer1 = E2EBlock(1, 32, example, bias=False)
        self.E2Elayer2 = E2EBlock(32, 32, example, bias=False)
        self.E2Nlayer = nn.Conv2d(32, 64, (1, self.d))   # 왜 channel이 1이 되지 64가 아니라, 참고 code는 32->1, 1->256
        self.N2Glayer = nn.Conv2d(64, 256, (self.d, 1))
        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 30)
        self.lin3 = nn.Linear(30, 2)

    def forward(self, x):
        x = F.leaky_relu(self.E2Elayer1(x), negative_slope=0.33)
        x = F.leaky_relu(self.E2Elayer2(x), negative_slope=0.33)
        x = F.leaky_relu(self.E2Nlayer(x), negative_slope=0.33)
        x = F.dropout(F.leaky_relu(self.N2Glayer(x), negative_slope=0.33), p=0.5)
        # x = x.view(x.size(0), -1) # 참고 code
        x = torch.flatten(x, 1)
        x = F.dropout(F.leaky_relu(self.lin1(x), negative_slope=0.33), p=0.5)
        x = F.dropout(F.leaky_relu(self.lin2(x), negative_slope=0.33), p=0.5)
        # x = F.leaky_relu(self.lin3(x), negative_slope=0.33) # 참고 code
        prob = F.softmax(self.lin3(x))
        out = self.lin3(x)

        return prob, out, x

if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 정확하진 않음(실행이 안 됨), 실행은 아직 못함, input data 형태를 잘 모르겠음, 그냥 2D는 아님
    functional_connectivity = [1,1,90,90]
    example = torch.tensor(functional_connectivity)

    net = BrainNetCNN(example=example, num_classes=2).to(device)
    summary(net, (1, 90, 90), 30)