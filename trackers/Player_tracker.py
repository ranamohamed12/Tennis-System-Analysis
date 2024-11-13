from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import sys
sys.path.append('../')
from utils.bbox_utils import get_center_of_bbox, measure_distance



class PlayerTracker:
    # to choose which model to use
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    

    def detect_frames (self, frames, read_from_stub = False,stub_path = None):
        
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open (stub_path, "rb") as f:
                player_detections = pickle.load(f)
            return player_detections
         
        for frame in frames:
            player_detections.append(self.detect_frame(frame))

        if stub_path is not None:
            with open (stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections
    
    # to detect players in a frame
    def detect_frame (self, frame):
        results = self.model.track(frame, persist=True)[0]
        ids_name_dict = results.names
        player_dict = {}

        for box in results.boxes:
            track_id = int (box.id.tolist()[0])
            bbox = box.xyxy.tolist()[0]

            class_id = box.cls.tolist()[0]
            class_name = ids_name_dict[class_id]

            if class_name == "person":
                player_dict[track_id] = bbox

        return player_dict
    
    
    # to draw boxes on frames
    def draw_player_boxes (self,frames, player_detections):
        output_frames = []
        for frame, player_detection in zip(frames, player_detections):
            for track_id, bbox in player_detection.items():
                x1, y1, x2, y2 = map (int,bbox)
                cv2.putText(frame, f"Player ID: {track_id}", (x1, y2 -10 ),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            output_frames.append(frame)

        return output_frames

    def choose_nearset_two_players (self, court_kps, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_centre = get_center_of_bbox(bbox)
            min_distane = float("inf")
            # looping over kps
            for kp in range(0,len(court_kps),2):
                court_points = [court_kps[kp], court_kps[kp+1]]
                distance = measure_distance(player_centre, court_points)
                if distance < min_distane:
                    min_distane = distance
            distances.append([track_id, min_distane])
        # sorting distances according to the second element(y)
        distances.sort(key=lambda x: x[1])
        return distances[0][0], distances[1][0]
    
    def choose_players_in_all_frames (self, court_kps, player_detections):
        
        player_detection_first_frame = player_detections[0]
        chosen_players = self.choose_nearset_two_players(court_kps, player_detection_first_frame)
        filtered_player_detections = []

        for player_detection in player_detections:
            filtered_player_dict = {}
            for track_id, bbox in player_detection.items():
                if track_id in chosen_players:
                    filtered_player_dict[track_id] = bbox
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections
       
    
        
