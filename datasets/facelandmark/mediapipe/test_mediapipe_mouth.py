#https://github.com/zmurez/MediaPipePyTorch/blob/master/blazeface_landmark.pth
import numpy as np
import torch
import cv2
import sys

from blazebase import resize_pad, denormalize_detections
from blazeface_landmark import BlazeFaceLandmark
from blazeface import BlazeFace

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS


def get_mouth_region(frame, points, flags):
    lip_landmarks = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    
    for i in range(len(flags)):
        landmark, flag = landmarks[i], flags[i]
        if flag>.5:
            #points = points[:,:2]
            lip_points = [points[0][i] for i in lip_landmarks]

            #draw_landmarks(frame, landmark[:,:2], FACE_CONNECTIONS, size=1)
            x_coordinates = [point[0] for point in lip_points]
            y_coordinates = [point[1] for point in lip_points]
            x_min, x_max = int(min(x_coordinates)), int(max(x_coordinates))
            y_min, y_max = int(min(y_coordinates)), int(max(y_coordinates))
            mouth_region = frame[y_min:y_max, x_min:x_max]
    return mouth_region
    """ points = points[:,:2]
    lip_landmarks = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    lip_points = [points[i] for i in lip_landmarks]
    x_coordinates = [point[0] for point in lip_points]
    y_coordinates = [point[1] for point in lip_points]
    x_min, x_max = int(min(x_coordinates)), int(max(x_coordinates))
    y_min, y_max = int(min(y_coordinates)), int(max(y_coordinates))
    mouth_region = img[y_min:y_max, x_min:x_max]
    return (mouth_region) """

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

back_detector = True

face_detector = BlazeFace(back_model=back_detector).to(gpu)

if back_detector:
    face_detector.load_weights("models/facelandmark/mediapipe/blazefaceback.pth")
    face_detector.load_anchors("models/facelandmark/mediapipe/anchors_face_back.npy")
else:
    face_detector.load_weights("models/facelandmark/mediapipe/blazeface.pth")
    face_detector.load_anchors("models/facelandmark/mediapipe/anchors_face.npy")


face_regressor = BlazeFaceLandmark().to(gpu)
face_regressor.load_weights("models/facelandmark/mediapipe/blazeface_landmark.pth")


video_file_path = "/code/datasets/eLipSys/data/s1/bras6n.mpg"
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

    if back_detector:
        normalized_face_detections = face_detector.predict_on_image(img1)
    else:
        normalized_face_detections = face_detector.predict_on_image(img2)

    face_detections = denormalize_detections(normalized_face_detections, scale, pad)


    xc, yc, scale, theta = face_detector.detection2roi(face_detections.cpu())
    img, affine, box = face_regressor.extract_roi(frame, xc, yc, theta, scale)
    flags, normalized_landmarks = face_regressor(img.to(gpu))
    landmarks = face_regressor.denormalize_landmarks(normalized_landmarks.cpu(), affine)

    cropped_img = get_mouth_region(frame,landmarks, flags)

    
    cv2.imwrite('test_sample/%04d.jpg'%frame_ct, cropped_img[:,:,::-1])

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
