import os
import numpy as np
import torch
from models.IFRNet import Model
from utils import read
from imageio import mimsave


model = Model().cuda().eval()
model.load_state_dict(torch.load('./checkpoints/IFRNet/IFRNet_Vimeo90K.pth'))

img0_np = read('./figures/img0.png')
img1_np = read('./figures/img1.png')

img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
embt = torch.tensor(1/2).view(1, 1, 1, 1).float().cuda()

imgt_pred = model.inference(img0, img1, embt)

imgt_pred_np = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

images = [img0_np, imgt_pred_np, img1_np]
mimsave('./figures/out_2x.gif', images, fps=3)
