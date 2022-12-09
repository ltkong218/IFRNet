import cv2
import torch
import numpy as np

from models.IFRNet import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read the video frame by frame

model = Model().to(device).eval()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
model.load_state_dict(torch.load('checkpoint/IFRNet/2022-11-10 09:09:26/IFRNet_best.pth', map_location=device))

video = cv2.VideoCapture('raw_video.mp4')

fps = video.get(cv2.CAP_PROP_FPS)
h, w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 2*fps, (w, h))

ret = True

ret, frame1 = video.read()

while ret:
    ret, frame2 = video.read()
    
    img0 = (torch.tensor(frame1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    img1 = (torch.tensor(frame2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    embt = torch.tensor(1/2).view(1, 1, 1, 1).float().to(device)
    
    imgt_pred = model.inference(img0, img1, embt)
    imgt_pred_np = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    
    # write the three frames to the output video
    output.write(img0.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    output.write(imgt_pred_np)
    
    frame1 = frame2
    
output.release()


    # do something with the frame