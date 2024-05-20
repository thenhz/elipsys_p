
import os
import torch
from utils.video_utils import save_numpy_video
from facelandmark.mediapipe.mouth_cropper import MouthCropper
from concurrent.futures import ThreadPoolExecutor

class PreprocessVideos:
    def __init__(self, video_folder, save_folder, max_frames=75):
        self.video_folder = video_folder
        self.save_folder = save_folder
        self.max_frames = max_frames
        self.mouth_cropper = MouthCropper()

    def load_video(self, video_file, max_frames=75):
        #inputs , _, _ = read_video(video_file) #T,H,W,C(3)
        inputs = self.mouth_cropper.crop_video(video_file) #T,H,W,C(3)
        #save_numpy_video(inputs.numpy(), "/code/datasets/eLipSys/data/s1/preprocessed/test.avi", (64,64))
        #inputs = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in inputs]
        inputs = torch.mean(inputs.float() / 255, dim=3, keepdim=True) #T,H,W,C(1) made a grey image
        num_frames, h, w, c = inputs.shape
        #needed just because I'm feeding straight the loaded data to the model
        inputs = inputs.permute(0, 3, 1, 2) #T,C,H,W
        if num_frames < max_frames:
            padding = torch.zeros((max_frames - num_frames, c, h, w))
            inputs = torch.cat((inputs, padding), dim=0)
        else:
            inputs = inputs[:max_frames]
        return inputs, torch.tensor(num_frames)

    def process_video(self, video_file):
        try:
            filepath = os.path.join(self.video_folder, video_file)
            # Load and preprocess video
            inputs, num_frames = self.load_video(filepath, self.max_frames)
            # Get the file name without extension
            base_filename = os.path.splitext(video_file)[0]
            save_path = os.path.join(self.save_folder, f"{base_filename}.pth")
            # Save preprocessed tensor
            torch.save((inputs, num_frames), save_path)
            #print(f"Saved preprocessed data at {save_path}")
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")

    def run(self):
        video_files = os.listdir(self.video_folder)  # get list of files in folder
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(self.process_video, video_files)

if __name__ == "__main__":
    video_folder = "/code/datasets/eLipSys/data/s1"
    save_folder = "/code/datasets/eLipSys/data/s1/preprocessed"
    preprocessor = PreprocessVideos(video_folder, save_folder, max_frames=75)
    preprocessor.run()
