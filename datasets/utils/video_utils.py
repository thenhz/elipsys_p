
import cv2

def save_numpy_video(frames, filename, size):
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, size)

    for frame in frames:
        
        #frame = frame.numpy()  # Convert from Torch Tensor back to numpy
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert back to BGR for opencv
        out.write(frame)

    out.release()