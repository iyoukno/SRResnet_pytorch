# -*- coding: utf-8 -*-
# @Time    : 2024/8/28 19:12
# @Author  : zjt
# @File    : net.py
# @info    :
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差模块"""

    def __init__(self, inChannals, outChannals):
        """初始化残差模块"""
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannals, outChannals, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(outChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannals)
        self.relu2 = nn.PReLU()
        # self.conv3 = nn.Conv2d(outChannals, outChannals, kernel_size=1, bias=False)
        self.relu = nn.PReLU()

    def forward(self, x):
        """前向传播过程"""
        resudial = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # out = self.conv3(x)

        out += resudial
        out = self.relu(out)
        return out


class SRResNet(nn.Module):
    """SRResNet模型(4x)"""

    def __init__(self):
        """初始化模型配置"""
        super(SRResNet, self).__init__()

        # 卷积模块1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, padding_mode='reflect', stride=1)
        self.relu = nn.PReLU()
        # 残差模块
        self.resBlock = nn.ModuleList(ResBlock(64,64) for i in range(6))
        # 卷积模块2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.PReLU()

        # 子像素卷积
        self.convPos1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=2, padding_mode='reflect')
        self.pixelShuffler1 = nn.PixelShuffle(2)
        self.reluPos1 = nn.PReLU()

        self.convPos2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.pixelShuffler2 = nn.PixelShuffle(2)
        self.reluPos2 = nn.PReLU()

        self.finConv = nn.Conv2d(64, 3, kernel_size=9, stride=1)

    def forward(self, x):
        """前向传播过程"""
        x = self.conv1(x)
        x = self.relu(x)
        residual = x

        out = x
        for m in self.resBlock:
            out = m(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        out = self.convPos1(out)
        out = self.pixelShuffler1(out)
        out = self.reluPos1(out)

        out = self.convPos2(out)
        out = self.pixelShuffler2(out)
        out = self.reluPos2(out)
        out = self.finConv(out)

        return out

if __name__ == '__main__':
    data = torch.randn(1, 3, 64, 64)
    s = SRResNet()
    ret = s(data)
    print(ret.shape)