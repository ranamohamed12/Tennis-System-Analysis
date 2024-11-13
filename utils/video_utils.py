import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames,fps

def save_video(frames, output_path,fps):
    fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc,fps,(frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)    
    out.release()