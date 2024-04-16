# encoding: utf-8
import numpy as np
import glob
import os
from torch.utils.data import Dataset
import torch
import random
import torchvision
from torchvision.transforms import Grayscale
from torchvision.io import read_video
import torch.nn.functional as F

class MyDataset(Dataset):
    def __init__(self, video_dir, label_dir, phases, args):
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.phases = phases
        self.args = args
        self.color_jitter = torchvision.transforms.ColorJitter(0.3, 0.3, 0.3)

        self.video_files = []
        self.label_files = []

        self.video_files += glob.glob(os.path.join(video_dir, '*.mpg'))
        self.label_files += glob.glob(os.path.join(label_dir, '*.align'))

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        label_file = self.label_files[idx]

        # Load video data
        #tensor = torch.load(video_file)
        inputs , _, _ = read_video(video_file, output_format="TCHW")
        #inputs = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in inputs]
        inputs = torch.mean(inputs.float() / 255, dim=1, keepdim=True)
        #inputs = np.stack(inputs, 0) / 255.0
        #batch_img = inputs[:, :, :, 0]  # 29, h, w



        num_frames, c, h, w = inputs.shape
        if num_frames < 75:
            padding = torch.zeros((75 - num_frames, c, h, w))
            #breaks here
            inputs = torch.cat((inputs, padding), dim=0)
            batch_img_padded = inputs
        else:
            batch_img_padded = inputs[:75]
            #batch_img_padded = batch_img
        #batch_img_padded = torch.FloatTensor(batch_img_padded[:, np.newaxis, ...])  # 75, 1 (C), h, w
        
        #if 'train' in self.phases:
            #batch_img = RandomDistort(batch_img, self.args.max_magnitude)
            #batch_img, remaining_list = RandomFrameDrop(batch_img, tensor.get('duration'))
            #batch_img = RandomCrop(batch_img, shaking_prob=self.args.shaking_prob)
            #batch_img = HorizontalFlip(batch_img)  # prob 0.5
            #batch_img_padded = torch.FloatTensor(batch_img_padded[:, np.newaxis, ...])  # 29, 1 (C), h, w
            #batch_img = self.color_jitter(batch_img)
        #else:
            #batch_img = CenterCrop(batch_img, (88, 88))
            #batch_img_padded = torch.FloatTensor(batch_img_padded[:, np.newaxis, ...])


        # Load label data
        with open(label_file, 'r') as f:
            lines = f.readlines()
            labels = []
            durations = []
            end = 0
            for line in lines:
                tmp_start, tmp_end, word = line.strip().split()
                end = int(tmp_end)
                if word == 'sil':
                    continue
                labels.append(word)
            durations = end
            labels = " ".join(labels)

        result = {}
        result['video'] = batch_img_padded
        result['label'] = labels
        #TODO: fix duration, it's an unkowk number
        result['duration'] = torch.tensor((0,1))

        return result

    def __len__(self):
        return len(self.video_files)
