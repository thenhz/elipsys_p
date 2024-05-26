# encoding: utf-8

#liplearner

import glob
import os
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision.io import read_video,write_video
import torch.nn.functional as F
from fractions import Fraction

from datasets.utils.transformers import *


class ConsistentRandomPerspective:
    def __init__(self, distortion_scale=0.5):
        self.distortion_scale = distortion_scale
        self.params = {}

    def __call__(self, img):
        if not self.params:
            _ , width, height = img.shape
            start_points, end_points = torchvision.transforms.RandomPerspective.get_params(width, height, self.distortion_scale)
            self.params = {'startpoints': start_points, 'endpoints': end_points}
        
        return torchvision.transforms.functional.perspective(img, **self.params)

class DummyDataset(Dataset):

    def __init__(self, video_dir, label_dir, phases, args):
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.phases = phases
        self.args = args
        self.video_resize_format = args['video_resize_format']
        self.max_frames_per_video = args['max_frames_per_video']
        self.color_jitter = torchvision.transforms.ColorJitter(0.3, 0.3, 0.3)

        self.video_files = []
        self.label_files = []

        if self.phases == 'train':
            self.video_files += glob.glob(os.path.join(video_dir, '*.mpg'))
            if self.args['short_train']:
                self.video_files = self.video_files[:args['batch']*2]
        elif self.phases == 'val':
            self.video_files += glob.glob(os.path.join(video_dir, '*.mpg'))[:args['batch']*1]
        else:
            raise ValueError("Invalid phase. It should be either 'train' or 'val'.")

        # Create label_files list matching video_files names but with '.align' extension
        self.label_files = [f"{os.path.splitext(os.path.basename(video_file))[0]}.align" for video_file in self.video_files]
        self.label_files = [os.path.join(self.label_dir, label_file) for label_file in self.label_files]
        # Create a mapping from words to numbers
        vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        self.char_to_num, self.num_to_char = self.word_to_number_mapping(vocab)
        #self.mouth_cropper = MouthCropper()

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        label_file = self.label_files[idx]

        result = {}
        result['video'], result['input_len'] = self.load_video(video_file)
        result['label'], result['output_len']  = self.extract_label(label_file) 

        return result

    def save_video(self, video_tensor, path, fps=24.0):
        # Convert the video tensor back to [0, 255] and to uint8 type
        video_tensor = (video_tensor * 255).byte()
        
        # Transpose the video tensor to (num_frames, height, width, channels)
        video_tensor = video_tensor.permute(0, 2, 3, 1)
        
        
        # Save the video
        write_video(path, video_tensor, fps=fps)

    def load_video(self, video_file):
        # Read the video using torchvision.io.read_video
        video, audio, info = read_video(video_file, output_format='TCHW', pts_unit='sec')
        # Normalize the video frames to be in the range [0, 1]
        video = video.float() / 255.0
        augment_transform = torchvision.transforms.Compose([])

        if self.args['model_name'] == 'resnet18':
            resize_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),                               # Resize to a slightly larger size
                torchvision.transforms.CenterCrop(self.video_resize_format[1]),   # Crop the center 224x224 pixels
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],      # Normalize with ImageNet mean
                                                 std=[0.229, 0.224, 0.225])      # Normalize with ImageNet std
            ])
            
            # Add augmentation if in training phase
            if self.phases == 'train':
                flip_transform = torchvision.transforms.RandomHorizontalFlip(p=0.5)  # 50% chance to flip
                distort_transform = torchvision.transforms.RandomApply([ConsistentRandomPerspective(distortion_scale=0.4)], p=0.5)  # Apply distortion with a probability of 50%
                augment_transform = torchvision.transforms.Compose([
                    flip_transform,
                    distort_transform
                ])
        # Apply the resize transform first to each frame in the video
        video = torch.stack([resize_transform(frame) for frame in video])
        
        # Apply augmentation transform to each frame in the video if in training phase
        if self.phases == 'train':
            video = torch.stack([augment_transform(frame) for frame in video])

        # Count the real number of loaded frames
        num_frames = video.shape[0]
        # Pad with zeros or slice the video frames to match self.args.max_frames
        if num_frames < self.max_frames_per_video:
            padding = torch.zeros((self.max_frames_per_video - num_frames, video.shape[1], video.shape[2], video.shape[3]))
            video = torch.cat((video, padding), dim=0)
        else:
            video = video[:self.max_frames_per_video]
        #self.save_video(video, "video.mp4")
        return video, torch.tensor(num_frames)


    def word_to_number_mapping(self, chars):
        char_to_num = {}
        num_to_char = {}
        for i, char in enumerate(chars):
            char_to_num[char] = i
            num_to_char[i] = char
        return char_to_num, num_to_char

    def extract_label(self, label_file, max_labels_length=40):
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
