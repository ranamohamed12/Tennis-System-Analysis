from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import pandas as pd


class BallTracker:
    # to choose which model to use
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    # to predict ball in a frame
    def detect_frame (self, frame):
        results = self.model.predict(frame ,conf = 0.15)[0]
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict
    
    def detect_frames (self, frames, read_from_stub = False,stub_path = None):
        
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open (stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections
         
        for frame in frames:
            ball_detections.append(self.detect_frame(frame))

        if stub_path is not None:
            with open (stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections
    
    # to draw boxes on frames
    def draw_player_boxes (self,frames, ball_detections):
        output_frames = []
        for frame, ball_detection in zip(frames, ball_detections):
            for ball_id, bbox in ball_detection.items():
                x1, y1, x2, y2 = map (int,bbox)
                cv2.putText(frame, f"Ball ID: {ball_id}", (x1, y2 -10 ),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            output_frames.append(frame)

        return output_frames
    
    # to interpolate ball positions
    def interpolate_ball_positions (self, ball_positions):

        ball_positions = [x.get(1,[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns= ["x1","y1","x2","y2"])
        df_ball_positions = df_ball_positions.interpolate()
        #to fill the firsts frames to handle if no ball is detected
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    # to get hit ball positions in the frames
    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions["mid_y"] = (df_ball_positions["y1"] + df_ball_positions["y2"])/2
        df_ball_positions["mid_y_rolling_mean"] = df_ball_positions["mid_y"].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions["delta_mid_y"] = df_ball_positions['mid_y_rolling_mean'].diff()
        df_ball_positions['ball_hit']=0

        minium_change_frame = 25

        for i in range(1,len(df_ball_positions)-int(minium_change_frame * 1.2)):
            change_count = 0 

            neg_change = df_ball_positions['delta_mid_y'].iloc[i] >0 and df_ball_positions['delta_mid_y'].iloc[i+1] <0 
            pos_change = df_ball_positions["delta_mid_y"].iloc[i]<0 and df_ball_positions["delta_mid_y"].iloc[i+1]>0

            if neg_change or pos_change:

                for frame_change in range(i+1, i+int(minium_change_frame*1.2)+1):
                        neg_position_change_following_frame = df_ball_positions['delta_mid_y'].iloc[i] >0 and df_ball_positions['delta_mid_y'].iloc[frame_change] <0
                        pos_position_change_following_frame = df_ball_positions['delta_mid_y'].iloc[i] >0 and df_ball_positions['delta_mid_y'].iloc[frame_change] <0

                        if neg_position_change_following_frame or neg_change:
                            change_count +=1
                        elif pos_position_change_following_frame or pos_change:
                            change_count +=1

                if change_count>(minium_change_frame-1):
                        df_ball_positions['ball_hit'].iloc[i] = 1
        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits







        

       
