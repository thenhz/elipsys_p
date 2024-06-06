
import cv2
import numpy as np
import torch

def save_numpy_video(frames, filename, size):
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, size)

    for frame in frames:
        
        #frame = frame.numpy()  # Convert from Torch Tensor back to numpy
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert back to BGR for opencv
        out.write(frame)

    out.release()

def extract_roi(frame, xc, yc, theta, scale, resolution=192):

        # take points on unit square and transform them according to the roi
        points = torch.tensor([[-1, -1, 1, 1],
                            [-1, 1, -1, 1]], device=scale.device).view(1,2,4)
        points = points * scale.view(-1,1,1)/2
        theta = theta.view(-1, 1, 1)
        R = torch.cat((
            torch.cat((torch.cos(theta), -torch.sin(theta)), 2),
            torch.cat((torch.sin(theta), torch.cos(theta)), 2),
            ), 1)
        center = torch.cat((xc.view(-1,1,1), yc.view(-1,1,1)), 1)
        points = R @ points + center

        # use the points to compute the affine transform that maps 
        # these points back to the output square
        res = resolution
        points1 = np.array([[0, 0, res-1],
                            [0, res-1, 0]], dtype=np.float32).T
        affines = []
        imgs = []
        for i in range(points.shape[0]):
            pts = points[i, :, :3].cpu().numpy().T
            M = cv2.getAffineTransform(pts, points1)
            img = cv2.warpAffine(frame, M, (res,res))#, borderValue=127.5)
            img = torch.tensor(img, device=scale.device)
            imgs.append(img)
            affine = cv2.invertAffineTransform(M).astype('float32')
            affine = torch.tensor(affine, device=scale.device)
            affines.append(affine)
        if imgs:
            imgs = torch.stack(imgs).permute(0,3,1,2).float() / 255.#/ 127.5 - 1.0
            affines = torch.stack(affines)
        else:
            imgs = torch.zeros((0, 3, res, res), device=scale.device)
            affines = torch.zeros((0, 2, 3), device=scale.device)

        return imgs, affines, points