import os
import sys
sys.path.append('.')
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.IFRNet import Model
# from models.IFRNet_L import Model
# from models.IFRNet_S import Model


img0 = torch.randn(1, 3, 256, 448).cuda()
img1 = torch.randn(1, 3, 256, 448).cuda()
embt = torch.tensor(1/2).float().view(1, 1, 1, 1).cuda()

model = Model().cuda().eval()

with torch.no_grad():
    for i in range(100):
        out = model.inference(img0, img1, embt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_stamp = time.time()
    for i in range(100):
        out = model.inference(img0, img1, embt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print('Time: {:.3f}s'.format((time.time() - time_stamp) / 100))

total = sum([param.nelement() for param in model.parameters()])
print('Parameters: {:.2f}M'.format(total / 1e6))
