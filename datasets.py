import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils import read


def random_resize(img0, imgt, img1, flow, p=0.1):
    if random.uniform(0, 1) < p:
        img0 = cv2.resize(img0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        imgt = cv2.resize(imgt, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        flow = cv2.resize(flow, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) * 2.0
    return img0, imgt, img1, flow


def random_crop(img0, imgt, img1, flow, crop_size=(224, 224)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih-h+1)
    y = np.random.randint(0, iw-w+1)
    img0 = img0[x:x+h, y:y+w, :]
    imgt = imgt[x:x+h, y:y+w, :]
    img1 = img1[x:x+h, y:y+w, :]
    flow = flow[x:x+h, y:y+w, :]
    return img0, imgt, img1, flow


def random_reverse_channel(img0, imgt, img1, flow, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, :, ::-1]
        imgt = imgt[:, :, ::-1]
        img1 = img1[:, :, ::-1]
    return img0, imgt, img1, flow


def random_vertical_flip(img0, imgt, img1, flow, p=0.3):
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        imgt = imgt[::-1]
        img1 = img1[::-1]
        flow = flow[::-1]
        flow = np.concatenate((flow[:, :, 0:1], -flow[:, :, 1:2], flow[:, :, 2:3], -flow[:, :, 3:4]), 2)
    return img0, imgt, img1, flow


def random_horizontal_flip(img0, imgt, img1, flow, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        imgt = imgt[:, ::-1]
        img1 = img1[:, ::-1]
        flow = flow[:, ::-1]
        flow = np.concatenate((-flow[:, :, 0:1], flow[:, :, 1:2], -flow[:, :, 2:3], flow[:, :, 3:4]), 2)
    return img0, imgt, img1, flow


def random_rotate(img0, imgt, img1, flow, p=0.05):
    if random.uniform(0, 1) < p:
        img0 = img0.transpose((1, 0, 2))
        imgt = imgt.transpose((1, 0, 2))
        img1 = img1.transpose((1, 0, 2))
        flow = flow.transpose((1, 0, 2))
        flow = np.concatenate((flow[:, :, 1:2], flow[:, :, 0:1], flow[:, :, 3:4], flow[:, :, 2:3]), 2)
    return img0, imgt, img1, flow


def random_reverse_time(img0, imgt, img1, flow, p=0.5):
    if random.uniform(0, 1) < p:
        tmp = img1
        img1 = img0
        img0 = tmp
        flow = np.concatenate((flow[:, :, 2:4], flow[:, :, 0:2]), 2)
    return img0, imgt, img1, flow


class Vimeo90K_Train_Dataset(Dataset):
    def __init__(self, dataset_dir='/home/ltkong/Datasets/Vimeo90K/vimeo_triplet', augment=True):
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.img0_list = []
        self.imgt_list = []
        self.img1_list = []
        self.flow_t0_list = []
        self.flow_t1_list = []
        with open(os.path.join(dataset_dir, 'tri_trainlist.txt'), 'r') as f:
            for i in f:
                name = str(i).strip()
                if(len(name) <= 1):
                    continue
                self.img0_list.append(os.path.join(dataset_dir, 'sequences', name, 'im1.png'))
                self.imgt_list.append(os.path.join(dataset_dir, 'sequences', name, 'im2.png'))
                self.img1_list.append(os.path.join(dataset_dir, 'sequences', name, 'im3.png'))
                self.flow_t0_list.append(os.path.join(dataset_dir, 'flow', name, 'flow_t0.flo'))
                self.flow_t1_list.append(os.path.join(dataset_dir, 'flow', name, 'flow_t1.flo'))

    def __len__(self):
        return len(self.imgt_list)

    def __getitem__(self, idx):
        img0 = read(self.img0_list[idx])
        imgt = read(self.imgt_list[idx])
        img1 = read(self.img1_list[idx])
        flow_t0 = read(self.flow_t0_list[idx])
        flow_t1 = read(self.flow_t1_list[idx])
        flow = np.concatenate((flow_t0, flow_t1), 2).astype(np.float64)

        if self.augment == True:
            img0, imgt, img1, flow = random_resize(img0, imgt, img1, flow, p=0.1)
            img0, imgt, img1, flow = random_crop(img0, imgt, img1, flow, crop_size=(224, 224))
            img0, imgt, img1, flow = random_reverse_channel(img0, imgt, img1, flow, p=0.5)
            img0, imgt, img1, flow = random_vertical_flip(img0, imgt, img1, flow, p=0.3)
            img0, imgt, img1, flow = random_horizontal_flip(img0, imgt, img1, flow, p=0.5)
            img0, imgt, img1, flow = random_rotate(img0, imgt, img1, flow, p=0.05)
            img0, imgt, img1, flow = random_reverse_time(img0, imgt, img1, flow, p=0.5)

        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        flow = torch.from_numpy(flow.transpose((2, 0, 1)).astype(np.float32))
        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))

        return img0, imgt, img1, flow, embt


class Vimeo90K_Test_Dataset(Dataset):
    def __init__(self, dataset_dir='/home/ltkong/Datasets/Vimeo90K/vimeo_triplet'):
        self.dataset_dir = dataset_dir
        self.img0_list = []
        self.imgt_list = []
        self.img1_list = []
        self.flow_t0_list = []
        self.flow_t1_list = []
        with open(os.path.join(dataset_dir, 'tri_testlist.txt'), 'r') as f:
            for i in f:
                name = str(i).strip()
                if(len(name) <= 1):
                    continue
                self.img0_list.append(os.path.join(dataset_dir, 'sequences', name, 'im1.png'))
                self.imgt_list.append(os.path.join(dataset_dir, 'sequences', name, 'im2.png'))
                self.img1_list.append(os.path.join(dataset_dir, 'sequences', name, 'im3.png'))
                self.flow_t0_list.append(os.path.join(dataset_dir, 'flow', name, 'flow_t0.flo'))
                self.flow_t1_list.append(os.path.join(dataset_dir, 'flow', name, 'flow_t1.flo'))

    def __len__(self):
        return len(self.imgt_list)

    def __getitem__(self, idx):
        img0 = read(self.img0_list[idx])
        imgt = read(self.imgt_list[idx])
        img1 = read(self.img1_list[idx])
        flow_t0 = read(self.flow_t0_list[idx])
        flow_t1 = read(self.flow_t1_list[idx])
        flow = np.concatenate((flow_t0, flow_t1), 2)

        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        flow = torch.from_numpy(flow.transpose((2, 0, 1)).astype(np.float32))
        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))
        
        return img0, imgt, img1, flow, embt



def random_resize_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, p=0.1):
    if random.uniform(0, 1) < p:
        img0 = cv2.resize(img0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img3 = cv2.resize(img3, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img4 = cv2.resize(img4, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img5 = cv2.resize(img5, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img6 = cv2.resize(img6, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img7 = cv2.resize(img7, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img8 = cv2.resize(img8, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    return img0, img1, img2, img3, img4, img5, img6, img7, img8


def random_crop_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, crop_size=(224, 224)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih-h+1)
    y = np.random.randint(0, iw-w+1)
    img0 = img0[x:x+h, y:y+w, :]
    img1 = img1[x:x+h, y:y+w, :]
    img2 = img2[x:x+h, y:y+w, :]
    img3 = img3[x:x+h, y:y+w, :]
    img4 = img4[x:x+h, y:y+w, :]
    img5 = img5[x:x+h, y:y+w, :]
    img6 = img6[x:x+h, y:y+w, :]
    img7 = img7[x:x+h, y:y+w, :]
    img8 = img8[x:x+h, y:y+w, :]
    return img0, img1, img2, img3, img4, img5, img6, img7, img8


def center_crop_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, crop_size=(512, 512)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    img0 = img0[(ih//2-h//2):(ih//2+h//2), (iw//2-w//2):(iw//2+w//2), :]
    img1 = img1[(ih//2-h//2):(ih//2+h//2), (iw//2-w//2):(iw//2+w//2), :]
    img2 = img2[(ih//2-h//2):(ih//2+h//2), (iw//2-w//2):(iw//2+w//2), :]
    img3 = img3[(ih//2-h//2):(ih//2+h//2), (iw//2-w//2):(iw//2+w//2), :]
    img4 = img4[(ih//2-h//2):(ih//2+h//2), (iw//2-w//2):(iw//2+w//2), :]
    img5 = img5[(ih//2-h//2):(ih//2+h//2), (iw//2-w//2):(iw//2+w//2), :]
    img6 = img6[(ih//2-h//2):(ih//2+h//2), (iw//2-w//2):(iw//2+w//2), :]
    img7 = img7[(ih//2-h//2):(ih//2+h//2), (iw//2-w//2):(iw//2+w//2), :]
    img8 = img8[(ih//2-h//2):(ih//2+h//2), (iw//2-w//2):(iw//2+w//2), :]
    return img0, img1, img2, img3, img4, img5, img6, img7, img8


def random_reverse_channel_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, :, ::-1]
        img1 = img1[:, :, ::-1]
        img2 = img2[:, :, ::-1]
        img3 = img3[:, :, ::-1]
        img4 = img4[:, :, ::-1]
        img5 = img5[:, :, ::-1]
        img6 = img6[:, :, ::-1]
        img7 = img7[:, :, ::-1]
        img8 = img8[:, :, ::-1]
    return img0, img1, img2, img3, img4, img5, img6, img7, img8


def random_vertical_flip_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, p=0.3):
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        img1 = img1[::-1]
        img2 = img2[::-1]
        img3 = img3[::-1]
        img4 = img4[::-1]
        img5 = img5[::-1]
        img6 = img6[::-1]
        img7 = img7[::-1]
        img8 = img8[::-1]
    return img0, img1, img2, img3, img4, img5, img6, img7, img8


def random_horizontal_flip_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        img1 = img1[:, ::-1]
        img2 = img2[:, ::-1]
        img3 = img3[:, ::-1]
        img4 = img4[:, ::-1]
        img5 = img5[:, ::-1]
        img6 = img6[:, ::-1]
        img7 = img7[:, ::-1]
        img8 = img8[:, ::-1]
    return img0, img1, img2, img3, img4, img5, img6, img7, img8


def random_rotate_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, p=0.05):
    if random.uniform(0, 1) < p:
        img0 = img0.transpose((1, 0, 2))
        img1 = img1.transpose((1, 0, 2))
        img2 = img2.transpose((1, 0, 2))
        img3 = img3.transpose((1, 0, 2))
        img4 = img4.transpose((1, 0, 2))
        img5 = img5.transpose((1, 0, 2))
        img6 = img6.transpose((1, 0, 2))
        img7 = img7.transpose((1, 0, 2))
        img8 = img8.transpose((1, 0, 2))
    return img0, img1, img2, img3, img4, img5, img6, img7, img8


def random_reverse_time_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, p=0.5):
    if random.uniform(0, 1) < p:
        return img8, img7, img6, img5, img4, img3, img2, img1, img0
    else:
        return img0, img1, img2, img3, img4, img5, img6, img7, img8


class GoPro_Train_Dataset(Dataset):
    def __init__(self, dataset_dir='/home/ltkong/Datasets/GOPRO', interFrames=7, n_inputs=2, augment=True):
        self.dataset_dir = dataset_dir
        self.interFrames = interFrames
        self.n_inputs = n_inputs
        self.augment = augment
        self.setLength = (n_inputs-1)*(interFrames+1)+1
        video_list = [
            'GOPR0372_07_00', 'GOPR0374_11_01', 'GOPR0378_13_00', 'GOPR0384_11_01', 'GOPR0384_11_04', 'GOPR0477_11_00', 'GOPR0868_11_02', 'GOPR0884_11_00', 
            'GOPR0372_07_01', 'GOPR0374_11_02', 'GOPR0379_11_00', 'GOPR0384_11_02', 'GOPR0385_11_00', 'GOPR0857_11_00', 'GOPR0871_11_01', 'GOPR0374_11_00', 
            'GOPR0374_11_03', 'GOPR0380_11_00', 'GOPR0384_11_03', 'GOPR0386_11_00', 'GOPR0868_11_01', 'GOPR0881_11_00']
        self.frames_list = []
        self.file_list = []
        for video in video_list:
            frames = sorted(os.listdir(os.path.join(self.dataset_dir, video)))
            n_sets = (len(frames) - self.setLength)//(interFrames+1)  + 1
            videoInputs = [frames[(interFrames+1)*i:(interFrames+1)*i+self.setLength] for i in range(n_sets)]
            videoInputs = [[os.path.join(video, f) for f in group] for group in videoInputs]
            self.file_list.extend(videoInputs)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        imgpaths = [os.path.join(self.dataset_dir, fp) for fp in self.file_list[idx]]
        pick_idxs = list(range(0, self.setLength, self.interFrames+1))
        rem = self.interFrames%2
        gt_idx = list(range(self.setLength//2-self.interFrames//2, self.setLength//2+self.interFrames//2+rem)) 
        input_paths = [imgpaths[idx] for idx in pick_idxs]
        gt_paths = [imgpaths[idx] for idx in gt_idx]
        img0 = np.array(read(input_paths[0]))
        img1 = np.array(read(gt_paths[0]))
        img2 = np.array(read(gt_paths[1]))
        img3 = np.array(read(gt_paths[2]))
        img4 = np.array(read(gt_paths[3]))
        img5 = np.array(read(gt_paths[4]))
        img6 = np.array(read(gt_paths[5]))
        img7 = np.array(read(gt_paths[6]))
        img8 = np.array(read(input_paths[1]))

        if self.augment == True:
            img0, img1, img2, img3, img4, img5, img6, img7, img8 = random_resize_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, p=0.1)
            img0, img1, img2, img3, img4, img5, img6, img7, img8 = random_crop_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, crop_size=(224, 224))
            img0, img1, img2, img3, img4, img5, img6, img7, img8 = random_reverse_channel_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, p=0.5)
            img0, img1, img2, img3, img4, img5, img6, img7, img8 = random_vertical_flip_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, p=0.3)
            img0, img1, img2, img3, img4, img5, img6, img7, img8 = random_horizontal_flip_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, p=0.5)
            img0, img1, img2, img3, img4, img5, img6, img7, img8 = random_rotate_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, p=0.05)
            img0, img1, img2, img3, img4, img5, img6, img7, img8 = random_reverse_time_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, p=0.5)
        else:
            img0, img1, img2, img3, img4, img5, img6, img7, img8 = center_crop_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, crop_size=(512, 512))
            
        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img2 = torch.from_numpy(img2.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img3 = torch.from_numpy(img3.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img4 = torch.from_numpy(img4.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img5 = torch.from_numpy(img5.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img6 = torch.from_numpy(img6.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img7 = torch.from_numpy(img7.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img8 = torch.from_numpy(img8.transpose((2, 0, 1)).astype(np.float32) / 255.0)

        emb1 = torch.from_numpy(np.array(1/8).reshape(1, 1, 1).astype(np.float32))
        emb2 = torch.from_numpy(np.array(2/8).reshape(1, 1, 1).astype(np.float32))
        emb3 = torch.from_numpy(np.array(3/8).reshape(1, 1, 1).astype(np.float32))
        emb4 = torch.from_numpy(np.array(4/8).reshape(1, 1, 1).astype(np.float32))
        emb5 = torch.from_numpy(np.array(5/8).reshape(1, 1, 1).astype(np.float32))
        emb6 = torch.from_numpy(np.array(6/8).reshape(1, 1, 1).astype(np.float32))
        emb7 = torch.from_numpy(np.array(7/8).reshape(1, 1, 1).astype(np.float32))

        return img0, img1, img2, img3, img4, img5, img6, img7, img8, emb1, emb2, emb3, emb4, emb5, emb6, emb7


class GoPro_Test_Dataset(Dataset):
    def __init__(self, dataset_dir='/home/ltkong/Datasets/GOPRO', interFrames=7, n_inputs=2):
        self.dataset_dir = dataset_dir
        self.interFrames = interFrames
        self.n_inputs = n_inputs
        self.setLength = (n_inputs-1)*(interFrames+1)+1
        video_list = [
            'GOPR0384_11_00', 'GOPR0385_11_01', 'GOPR0410_11_00', 'GOPR0862_11_00', 'GOPR0869_11_00', 'GOPR0881_11_01', 'GOPR0384_11_05', 'GOPR0396_11_00', 
            'GOPR0854_11_00', 'GOPR0868_11_00', 'GOPR0871_11_00']
        self.frames_list = []
        self.file_list = []
        for video in video_list:
            frames = sorted(os.listdir(os.path.join(self.dataset_dir, video)))
            n_sets = (len(frames) - self.setLength)//(interFrames+1)  + 1
            videoInputs = [frames[(interFrames+1)*i:(interFrames+1)*i+self.setLength] for i in range(n_sets)]
            videoInputs = [[os.path.join(video, f) for f in group] for group in videoInputs]
            self.file_list.extend(videoInputs)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        imgpaths = [os.path.join(self.dataset_dir, fp) for fp in self.file_list[idx]]
        pick_idxs = list(range(0, self.setLength, self.interFrames+1))
        rem = self.interFrames%2
        gt_idx = list(range(self.setLength//2-self.interFrames//2, self.setLength//2+self.interFrames//2+rem)) 
        input_paths = [imgpaths[idx] for idx in pick_idxs]
        gt_paths = [imgpaths[idx] for idx in gt_idx]
        img0 = np.array(read(input_paths[0]))
        img1 = np.array(read(gt_paths[0]))
        img2 = np.array(read(gt_paths[1]))
        img3 = np.array(read(gt_paths[2]))
        img4 = np.array(read(gt_paths[3]))
        img5 = np.array(read(gt_paths[4]))
        img6 = np.array(read(gt_paths[5]))
        img7 = np.array(read(gt_paths[6]))
        img8 = np.array(read(input_paths[1]))

        img0, img1, img2, img3, img4, img5, img6, img7, img8 = center_crop_8x(img0, img1, img2, img3, img4, img5, img6, img7, img8, crop_size=(512, 512))

        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img2 = torch.from_numpy(img2.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img3 = torch.from_numpy(img3.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img4 = torch.from_numpy(img4.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img5 = torch.from_numpy(img5.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img6 = torch.from_numpy(img6.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img7 = torch.from_numpy(img7.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img8 = torch.from_numpy(img8.transpose((2, 0, 1)).astype(np.float32) / 255.0)

        emb1 = torch.from_numpy(np.array(1/8).reshape(1, 1, 1).astype(np.float32))
        emb2 = torch.from_numpy(np.array(2/8).reshape(1, 1, 1).astype(np.float32))
        emb3 = torch.from_numpy(np.array(3/8).reshape(1, 1, 1).astype(np.float32))
        emb4 = torch.from_numpy(np.array(4/8).reshape(1, 1, 1).astype(np.float32))
        emb5 = torch.from_numpy(np.array(5/8).reshape(1, 1, 1).astype(np.float32))
        emb6 = torch.from_numpy(np.array(6/8).reshape(1, 1, 1).astype(np.float32))
        emb7 = torch.from_numpy(np.array(7/8).reshape(1, 1, 1).astype(np.float32))

        return img0, img1, img2, img3, img4, img5, img6, img7, img8, emb1, emb2, emb3, emb4, emb5, emb6, emb7
