# -*- coding: utf-8 -*-
# @Time    : 2024/8/28 19:13
# @Author  : zjt
# @File    : train.py
# @info    :

import torch
from torch.optim import Adam
from net import SRResNet
from datasets import PreprocessDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer():
    def __init__(self, dataset_path, batch):
        # net
        self.net = SRResNet()

        # optimizer
        self.optimizer = Adam(self.net.parameters(), lr=0.01)

        # loss function
        self.loss_f = torch.nn.MSELoss()

        # datasets dataloader
        self.datasets = PreprocessDataset(dataset_path)
        self.dataloader = DataLoader(self.datasets, batch, shuffle=True)

    def train(self, epoch):
        net = self.net.to(DEVICE)
        net.train()

        min_loss = 999
        for e in range(epoch):
            e_loss = 0

            for idx, (corp_img, img) in tqdm(enumerate(self.dataloader)):
                corp_img, img = corp_img.to(DEVICE), img.to(DEVICE)

                out = net(corp_img)
                loss = self.loss_f(out, img)
                e_loss += loss

                # gradient update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            avg_loss = e_loss / len(self.dataloader)
            if avg_loss < min_loss:
                min_loss = avg_loss
                torch.save(net.state_dict(), f'./train_dir/best.pth')
            torch.save(net.state_dict(), f'./train_dir/{e}.pth')
            print(f'model save success, epoch: {e} loss {avg_loss}')

if __name__ == '__main__':
    ds_path = r'E:\BaiduNetdiskDownload\data\Urban100\HR'
    trainer = Trainer(ds_path,batch=4)
    trainer.train(100)