'''
@Project ：SRResnet_pytorch 
@File    ：test.py
@Author  ：yuk
@Date    ：2024/8/29 14:11 
description：
'''

import torch


from net import SRResNet
from torchvision import transforms
from PIL import Image
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])

# load model
m = SRResNet()
model_state_dict = torch.load('./train_dir/best.pth')
m.load_state_dict(state_dict=model_state_dict)
m.eval()
# read img
img = Image.open(r'E:\BaiduNetdiskDownload\data\Set14\image_SRF_4\LR\img_004.png')

img = transform(img)
img = torch.unsqueeze(img,dim=0)
out = m(img)
out = torch.squeeze(out,dim=0)
to_pil = transforms.ToPILImage()
i = to_pil(out)
i.save('out.jpg')