from utils import read_video, save_video,measure_distance,convert_pixels_to_meters,draw_player_stats
from trackers import PlayerTracker,BallTracker
from Court_line_detector import CourtLineDetector
from small_court import small_court
import constants
import pandas as pd
from copy import deepcopy
import os
import cv2

def main():

    # Read video
    os.makedirs("Output Videos", exist_ok=True)
    output_path = "Output Videos/output_video.avi"
    video_path = "Input Videos/input_video.mp4"
    frames,fps = read_video(video_path)

    #detect players and balls
    tracker = PlayerTracker(model_path = "yolov8x.pt")
    ball_tracker = BallTracker(model_path = "models/last.pt")

    player_detections = tracker.detect_frames(frames,read_from_stub = True,stub_path = "tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(frames,read_from_stub = True,stub_path = "tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)



    #detect courts (Keypoints) on frames
    court_model_path = "models/kps_model.pth"
    line_detector = CourtLineDetector(court_model_path)
    court_keypoints = line_detector.predict(frames[0])

    #choose players in all frames
    player_detections = tracker.choose_players_in_all_frames(court_keypoints, player_detections)

    #draw mini court on frames
    mini_court = small_court(frames[0])

    #detect mini_shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    #print(ball_shot_frames)


    #convert ball and players position into pixels in small court 
    player_detections_small_court ,ball_detections_small_court = mini_court.convert_bbox_to_small_court_coordinates(player_detections,ball_detections,court_keypoints)
    
    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]
    #loop through ball shots
    for ball_shot in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot]
        end_frame = ball_shot_frames[ball_shot+1]
        ball_shot_time_seconds = (end_frame - start_frame)/24
        
        #calculate the distance covered by the ball in meters then convert it to pixels
        distance_covered_ball_pixels = measure_distance(ball_detections_small_court[start_frame][1],ball_detections_small_court[end_frame][1])
        distance_covered_ball_meters = convert_pixels_to_meters(distance_covered_ball_pixels,constants.DOUBLE_LINE_WIDTH,mini_court.get_width_of_small_court())
        
        #calculate the speed of the ball in km/h
        ball_shot_speed = distance_covered_ball_meters / ball_shot_time_seconds*3.6

        #who shot the ball
        player_position = player_detections_small_court[start_frame]
        player_shot_ball = min(player_position.keys(), key = lambda x: measure_distance(player_position[x],ball_detections_small_court[start_frame][1]))

        # calculate the same calculations for opponent player 
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_opponent_pixels = measure_distance(player_detections_small_court[start_frame][opponent_player_id],player_detections_small_court[end_frame][opponent_player_id])
        distance_covered_opponent_meters = convert_pixels_to_meters(distance_covered_opponent_pixels,constants.DOUBLE_LINE_WIDTH,mini_court.get_width_of_small_court())
        opponent_player_speed = distance_covered_opponent_meters / ball_shot_time_seconds*3.6

        #update the statistics dictionary player  
        #copy the last update of the player stats
        current_player_stats = deepcopy(player_stats_data[-1])

        current_player_stats["frame_num"] = start_frame
        current_player_stats[f"player_{player_shot_ball}_number_of_shots"] += 1
        current_player_stats[f"player_{player_shot_ball}_total_shot_speed"] += ball_shot_speed
        current_player_stats[f"player_{player_shot_ball}_last_shot_speed"] = ball_shot_speed

        current_player_stats[f"player_{opponent_player_id}_total_player_speed"] += opponent_player_speed
        current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = opponent_player_speed

        player_stats_data.append(current_player_stats)
        #convert stats to pandas frame 
        player_stats_data_df = pd.DataFrame(player_stats_data)
        frames_df = pd.DataFrame({'frame_num': list(range(len(frames)))})
        player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
        player_stats_data_df = player_stats_data_df.ffill()

        player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
        player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
        player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
        player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']



    #draw boxes on frames
    output_frames = tracker.draw_player_boxes(frames, player_detections)
    output_frames = ball_tracker.draw_player_boxes(output_frames, ball_detections)
    output_frames = line_detector.draw_keypoints_video(output_frames,court_keypoints)
    
    output_frames = mini_court.draw_small_court(output_frames)

    output_frames = mini_court.draw_points_on_small_court(output_frames,player_detections_small_court)
    output_frames = mini_court.draw_points_on_small_court(output_frames,ball_detections_small_court,color=(0,255,255))

    output_frames = draw_player_stats(output_frames,player_stats_data_df)
  
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #save video
    save_video(output_frames, output_path,fps)

if __name__ == "__main__":
    main()