import os
import sys
sys.path.append('.')
import torch
import numpy as np
from utils import read
from metric import calculate_psnr, calculate_ssim, calculate_ie
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

# Replace the 'path' with your Middlebury dataset absolute path.
path = '/home/ltkong/Datasets/Middlebury/'
sequence = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']

psnr_list = []
ssim_list = []
ie_list = []
for i in sequence:
    I0 = read(path + 'other-data/{}/frame10.png'.format(i))
    I1 = read(path + 'other-gt-interp/{}/frame10i11.png'.format(i))
    I2 = read(path + 'other-data/{}/frame11.png'.format(i))
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)

    I0_pad = torch.zeros([1, 3, 480, 640]).to(device)
    I2_pad = torch.zeros([1, 3, 480, 640]).to(device)
    h, w = I0.shape[-2:]
    I0_pad[:, :, :h, :w] = I0
    I2_pad[:, :, :h, :w] = I2

    I1_pred_pad = model.inference(I0_pad, I2_pad, embt)
    I1_pred = I1_pred_pad[:, :, :h, :w]

    psnr = calculate_psnr(I1_pred, I1).detach().cpu().numpy()
    ssim = calculate_ssim(I1_pred, I1).detach().cpu().numpy()
    ie = calculate_ie(I1_pred, I1).detach().cpu().numpy()

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    ie_list.append(ie)

    print('Avg PSNR: {} SSIM: {} IE: {}'.format(np.mean(psnr_list), np.mean(ssim_list), np.mean(ie_list)))
