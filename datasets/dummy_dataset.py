# encoding: utf-8

#liplearner

import glob
import os
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision.io import read_video
import torch.nn.functional as F

from datasets.utils.transformers import *
from datasets.facelandmark.mediapipe.mouth_cropper import MouthCropper




class DummyDataset(Dataset):

    def __init__(self, video_dir, label_dir, phases, args):
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.phases = phases
        self.args = args
        self.color_jitter = torchvision.transforms.ColorJitter(0.3, 0.3, 0.3)

        self.video_files = []
        self.label_files = []

        self.video_files += glob.glob(os.path.join(video_dir, '*.pth'))
        self.label_files += glob.glob(os.path.join(label_dir, '*.align'))
        # Create a mapping from words to numbers
        vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        self.char_to_num, self.num_to_char = self.word_to_number_mapping(vocab)
        self.mouth_cropper = MouthCropper()

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        label_file = self.label_files[idx]

        result = {}
        result['video'], result['input_len'] = torch.load(video_file)
        result['label'], result['output_len']  = self.extract_label(label_file) 

        return result

    def word_to_number_mapping(self, chars):
        char_to_num = {}
        num_to_char = {}
        for i, char in enumerate(chars):
            char_to_num[char] = i
            num_to_char[i] = char
        return char_to_num, num_to_char

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

        return padded_labels, torch.tensor(label_len)

    def __len__(self):
        return len(self.video_files)
