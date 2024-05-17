import numpy as np
import torch
import cv2
import sys

from blazebase import resize_pad, denormalize_detections
from blazeface_landmark import BlazeFaceLandmark
from blazeface import BlazeFace

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

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

    for i in range(len(flags)):
        landmark, flag = landmarks[i], flags[i]
        if flag>.5:
            draw_landmarks(frame, landmark[:,:2], FACE_CONNECTIONS, size=1)


    draw_roi(frame, box)
    draw_detections(frame, face_detections)

    #cv2.imshow(WINDOW, frame[:,:,::-1])
    cv2.imwrite('test_sample/%04d.jpg'%frame_ct, frame[:,:,::-1])

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
