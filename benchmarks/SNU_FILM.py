import os
import sys
sys.path.append('.')
import torch
import torch.nn.functional as F
import numpy as np
from utils import read
from metric import calculate_psnr, calculate_ssim
from models.IFRNet import Model
# from models.IFRNet_L import Model
# from models.IFRNet_S import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = Model()
model.load_state_dict(torch.load('checkpoints/IFRNet/IFRNet_Vimeo90K.pth'))
# model.load_state_dict(torch.load('checkpoints/IFRNet_large/IFRNet_L_Vimeo90K.pth'))
# model.load_state_dict(torch.load('checkpoints/IFRNet_small/IFRNet_S_Vimeo90K.pth'))
model.eval()
model.cuda()

divisor = 20
scale_factor = 0.8

class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor=divisor):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

# Replace the 'path' with your SNU-FILM dataset absolute path.
path = '/home/ltkong/Datasets/SNU-FILM/'

psnr_list = []
ssim_list = []
file_list = []
test_file = "test-hard.txt" # test-easy.txt, test-medium.txt, test-hard.txt, test-extreme.txt
with open(os.path.join(path, test_file), "r") as f:
    for line in f:
        line = line.strip()
        file_list.append(line.split(' '))

for line in file_list:
    print(os.path.join(path, line[0]))
    I0_path = os.path.join(path, line[0])
    I1_path = os.path.join(path, line[1])
    I2_path = os.path.join(path, line[2])
    I0 = read(I0_path)
    I1 = read(I1_path)
    I2 = read(I2_path)
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    padder = InputPadder(I0.shape)
    I0, I2 = padder.pad(I0, I2)
    embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)

    I1_pred = model.inference(I0, I2, embt, scale_factor=scale_factor)
    I1_pred = padder.unpad(I1_pred)

    psnr = calculate_psnr(I1_pred, I1).detach().cpu().numpy()
    ssim = calculate_ssim(I1_pred, I1).detach().cpu().numpy()

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    
    print('Avg PSNR: {} SSIM: {}'.format(np.mean(psnr_list), np.mean(ssim_list)))
