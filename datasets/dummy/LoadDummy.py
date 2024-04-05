import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms import Grayscale

class DummyDataset(Dataset):
    def __init__(self, video_dir, label_dir, video_extension='.mpg', label_extension='.align', transform=None):
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.video_extension = video_extension
        self.label_extension = label_extension
        self.transform = transform
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith(video_extension)]
        self.label_files = [f.replace(video_extension, label_extension) for f in self.video_files]
        # Create a mapping from words to numbers
        vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        self.char_to_num, self.num_to_char = self.word_to_number_mapping(vocab)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # Load video frames
        video_frames = self.load_video(video_path)

        labels = self.load_labels(label_path)

        # Apply transformations if any
        if self.transform:
            video_frames = self.transform(video_frames)

        # Convert video frames to tensor
        video_tensor = torch.tensor(video_frames)
        label_tensor = torch.tensor(labels)

        return video_tensor, label_tensor

    
    def read_alignments(self, label_path):
        words = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    if parts[2] != 'sil':
                        words.append(parts[2])
        return ' '.join(words)

    """ def load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Crop the frame with the given coordinates
            cropped_frame = frame[190:236,80:220,:]
            # Convert the cropped frame to grayscale
            gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR)
            frames.append(gray_frame)
        cap.release()
        return frames """
    
    def load_video(self, video_path):

        # Read the video using torchvision.io.read_video
        video_frames, _, _ = read_video(video_path, output_format='TCHW')
         
        # Crop the frames with the given coordinates
        cropped_frames = [frame[:,190:236,80:220] for frame in video_frames]

        # Convert the cropped frames to grayscale
        # Convert the video frames to grayscale using the Grayscale transform
        gray_frames = [Grayscale()(frame) for frame in cropped_frames]


        # Check if the video is too short or too long
        num_frames = len(gray_frames)
        if num_frames < 75:
            # Pad the video with zeros if it is too short
            padding = torch.zeros((75 - num_frames, 1, 46, 140))
            gray_frames.extend(padding) 
        elif num_frames > 75:
            # Cut the exceeding frames if the video is too long
            gray_frames = gray_frames[:75]

        # Convert the cropped frames to a tensor
        video_tensor = torch.stack(gray_frames)

        return video_tensor.transpose(0, 1)
    
    def load_labels(self, label_path):
        # Load label and process the content
        alignments = self.read_alignments(label_path )

        # Convert the words into a list of numbers
        numerical_representation = [self.char_to_num[char] for char in alignments]

        # Pad the numerical representation with zeros if it is too short
        if len(numerical_representation) < 75:
            padding = [0] * (75 - len(numerical_representation))
            numerical_representation.extend(padding)

        return numerical_representation

    def word_to_number_mapping(self, chars):
        char_to_num = {}
        num_to_char = {}
        for i, char in enumerate(chars):
            char_to_num[char] = i
            num_to_char[i] = char
        return char_to_num, num_to_char


# Example usage:
# dataset = VideoDataset(video_dir='path/to/videos', label_dir='path/to/labels')
# dataset = DummyDataset(video_dir='path/to/videos', label_dir='path/to/labels')
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
