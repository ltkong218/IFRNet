import os
import sys
sys.path.append('.')
import torch
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

# Replace the 'path' with your UCF101 dataset absolute path.
path = '/home/ltkong/Datasets/UCF101/ucf101_interp_ours/'
dirs = sorted(os.listdir(path))

psnr_list = []
ssim_list = []
for d in dirs:
    print(path + d + '/frame_00.png')
    I0 = read(path + d + '/frame_00.png')
    I1 = read(path + d + '/frame_01_gt.png')
    I2 = read(path + d + '/frame_02.png')
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)

    I1_pred = model.inference(I0, I2, embt)

    psnr = calculate_psnr(I1_pred, I1).detach().cpu().numpy()
    ssim = calculate_ssim(I1_pred, I1).detach().cpu().numpy()

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    
    print('Avg PSNR: {} SSIM: {}'.format(np.mean(psnr_list), np.mean(ssim_list)))
