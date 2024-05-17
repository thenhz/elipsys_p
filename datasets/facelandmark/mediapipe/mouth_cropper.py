import numpy as np
import torch
import cv2
import sys

from .blazebase import resize_pad, denormalize_detections
from .blazeface_landmark import BlazeFaceLandmark
from .blazeface import BlazeFace

class MouthCropper:
    def __init__(self, back_detector: bool = False):
        self.gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        
        self.back_detector = back_detector

        self.face_detector = BlazeFace(back_model=back_detector).to(self.gpu)
        if back_detector:
            self.face_detector.load_weights("datasets/facelandmark/mediapipe/blazefaceback.pth")
            self.face_detector.load_anchors("datasets/facelandmark/mediapipe/anchors_face_back.npy")
        else:
            self.face_detector.load_weights("datasets/facelandmark/mediapipe/blazeface.pth")
            self.face_detector.load_anchors("datasets/facelandmark/mediapipe/anchors_face.npy")

        self.face_regressor = BlazeFaceLandmark().to(self.gpu)
        self.face_regressor.load_weights("datasets/facelandmark/mediapipe/blazeface_landmark.pth")

    def get_mouth_region(self, frame, points, flags, size=(96,96)):
        lip_landmarks = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61, 185, 40, 
                        39, 37, 0, 267, 269, 270, 409, 291, 78, 95, 88, 178, 87, 14, 317,
                        402, 318, 324, 308, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

        for i in range(len(flags)):
            landmark, flag = points[i], flags[i]
            if flag>.5:
                lip_points = [points[0][i] for i in lip_landmarks]
                x_coordinates = [point[0] for point in lip_points]
                y_coordinates = [point[1] for point in lip_points]

                # Calculate width of bounding box
                width = int(max(x_coordinates) - min(x_coordinates))

                # Calculate the center of the bounding box
                center_x = min(x_coordinates) + width // 2
                center_y = min(y_coordinates) + width // 2  # Use width as height because the mouth is approximately in the center

                # Calculate new dimensions based on a fixed amount around the center
                start_x = int(center_x - width // 2)  
                start_y = int(center_y - width // 2)  
                end_x = start_x + width
                end_y = start_y + width

                mouth_region = frame[start_y:end_y, start_x:end_x]

                # Resize the mouth region to 96 x 96
                mouth_region_resized = cv2.resize(mouth_region, size)

        return mouth_region_resized

    def crop_video(self, video_file_path):
        frames = []
        capture = cv2.VideoCapture(video_file_path)
        mirror_img = False
        if capture.isOpened():
            hasFrame, frame = capture.read()
            frame_ct = 0
        else:
            hasFrame = False
        
        while hasFrame:
            frame_ct +=1
            frame = np.ascontiguousarray(frame[:,:,::-1])
            img1, img2, scale, pad = resize_pad(frame)

            if self.back_detector:
                normalized_face_detections = self.face_detector.predict_on_image(img1)
            else:
                normalized_face_detections = self.face_detector.predict_on_image(img2)
            face_detections = denormalize_detections(normalized_face_detections, scale, pad)

            xc, yc, scale, theta = self.face_detector.detection2roi(face_detections.cpu())
            img, affine, box = self.face_regressor.extract_roi(frame, xc, yc, theta, scale)
            flags, normalized_landmarks = self.face_regressor(img.to(self.gpu))
            landmarks = self.face_regressor.denormalize_landmarks(normalized_landmarks.cpu(), affine)

            cropped_img = self.get_mouth_region(frame,landmarks, flags)
            frames.append(cropped_img)
            hasFrame, frame = capture.read()

        capture.release()
        #convert frames to tensor in T H W C format
        frames = torch.stack([torch.from_numpy(np.transpose(frame,(2,0,1))) for frame in frames])
        return frames


