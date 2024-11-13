from ultralytics import YOLO

model = YOLO("yolov8x")
results = model.track("Input Videos\input_video.mp4",save = True,conf = 0.2)
# print(results)
# print("boxes: ")

# for box in results[0].boxes:
#     print(box)