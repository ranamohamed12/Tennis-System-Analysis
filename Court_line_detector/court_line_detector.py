import torchvision
import torch
from torchvision import transforms,models
import cv2
import numpy as np


class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights="DEFAULT")
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
            
    
    def predict (self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(img_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
        org_height, org_width = img.shape[:2]

        keypoints[::2] *=org_width/224.0
        keypoints[1::2] *=org_height/224.0

        return keypoints
    
    def draw_keypoints(self, img, keypoints):
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i+1])
            cv2.putText(img, str(i//2), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(img, (x,y), 5, (0, 255, 0), -1)
        return img
    
    def draw_keypoints_video(self, video_frames, keypoints):
        output_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_frames.append(frame)
        return output_frames
          
    

    


