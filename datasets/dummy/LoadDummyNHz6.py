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

from datasets.dummy.transformers import *



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
        # Create a mapping from words to numbers
        vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        self.char_to_num, self.num_to_char = self.word_to_number_mapping(vocab)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        label_file = self.label_files[idx]

        result = {}
        result['video'] = self.load_video(video_file)
        result['label'], result['duration'] = self.extract_label(label_file) 

        return result

    def word_to_number_mapping(self, chars):
        char_to_num = {}
        num_to_char = {}
        for i, char in enumerate(chars):
            char_to_num[char] = i
            num_to_char[i] = char
        return char_to_num, num_to_char
    
    def load_video(self, video_file, max_frames=75):
        inputs , _, _ = read_video(video_file)
        #inputs = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in inputs]
        inputs = torch.mean(inputs.float() / 255, dim=3, keepdim=True)
        num_frames, c, h, w = inputs.shape

        if 'train' in self.phases:
            inputs = RandomDistort(inputs, self.args['max_magnitude'])
            inputs, remaining_list = RandomFrameDrop(inputs, num_frames)
            #batch_img = RandomCrop(batch_img, shaking_prob=self.args.shaking_prob)
            inputs = HorizontalFlip(inputs)  # prob 0.5
            #batch_img_padded = torch.FloatTensor(batch_img_padded[:, np.newaxis, ...])  # 29, 1 (C), h, w
            inputs = self.color_jitter(inputs)

        if num_frames < max_frames:
            padding = torch.zeros((max_frames - num_frames, c, h, w))
            inputs = torch.cat((inputs, padding), dim=0)
        else:
            inputs = inputs[:max_frames]
        
        #batch_img_padded = batch_img
        #batch_img_padded = torch.FloatTensor(batch_img_padded[:, np.newaxis, ...])  # 75, 1 (C), h, w
        
        
        #else:
            #batch_img = CenterCrop(batch_img, (88, 88))
            #batch_img_padded = torch.FloatTensor(batch_img_padded[:, np.newaxis, ...])
        return inputs

    def extract_label(self, label_file, max_labels_length=40):
        #TODO:must be converterd in numbers
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
        # Convert the words into a list of numbers
        numerical_representation = [self.char_to_num[char] for char in labels]
        label_len = len(numerical_representation)
        numerical_representation = torch.tensor(numerical_representation)

        if label_len < max_labels_length:
            labels_padding = torch.zeros((max_labels_length - label_len))
            #breaks here
            padded_labels = torch.cat((numerical_representation, labels_padding), dim=0)
        else:
            padded_labels = numerical_representation[:max_labels_length]

        #TODO: fix duration, it's an unkowk number
        return padded_labels, torch.tensor((0,1))

    def __len__(self):
        return len(self.video_files)
