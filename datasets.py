# -*- coding: utf-8 -*-
# @Time    : 2024/8/28 19:13
# @Author  : zjt
# @File    : datasets.py
# @info    :

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms
from PIL import Image

#图像处理操作，包括随机裁剪，转换张量
transform = transforms.Compose([transforms.RandomCrop(96),
                            transforms.ToTensor()])

class PreprocessDataset(Dataset):
    """预处理数据集类"""

    def __init__(self, imgPath, ex=1):
        """初始化预处理数据集类"""
        self.transforms = transform

        for _, _, files in os.walk(imgPath):
            # ex变量是用于扩充数据集的，在这里默认的是扩充十倍
            self.imgs = [os.path.join(imgPath, file) for file in files] * ex

        np.random.shuffle(self.imgs)  # 随机打乱

    def __len__(self):
        """获取数据长度"""
        return len(self.imgs)

    def __getitem__(self, index):
        """获取数据"""
        tempImg = self.imgs[index]
        tempImg = Image.open(tempImg)

        sourceImg = self.transforms(tempImg)  # 对原始图像进行处理
        cropImg = torch.nn.MaxPool2d(4)(sourceImg)
        return cropImg, sourceImg

if __name__ == '__main__':
    path = r'E:\BaiduNetdiskDownload\data\Urban100\HR'
    d = PreprocessDataset(path)
    crop_img, img = d[0]
    print(crop_img)