import cv2
import numpy as np
import os
import sys
sys.path.append('../')
import constants
from utils import (convert_meters_to_pixels, convert_pixels_to_meters,get_center_of_bbox,
                   measure_distance, get_foot_position,get_closest_court_keypoint,
                   get_height_of_bbox,measure_xy_distance)


class small_court:
    def __init__(self,frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        #the distance between the court and the top of the frame
        self.buffer = 50
        #the distance between the black line court and white background
        self.padding_court =20
        self.set_canvas_background_box_position(frame)
        self.set_small_court_position()
        self.set_court_drawing_kps()
        self.set_court_lines()
    
    #set the dimension of the white rectangle
    def set_canvas_background_box_position(self,frame):
        frame = frame.copy()
        self.end_x = frame.shape[1]-self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height


    #set the dimension of the smallcourt
    def set_small_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
    
    
    #set the coordinates of the 28 points of the court in a list
    def set_court_drawing_kps(self):
        drawing_kps = [0]*28
        # point 0
        drawing_kps[0], drawing_kps[1] = int(self.court_start_x),int(self.court_start_y)
        # point 1
        drawing_kps[2], drawing_kps[3] = int(self.court_end_x),int(self.court_start_y)
        # point 2
        drawing_kps[4]=int(self.court_start_x)
        drawing_kps[5]=int(self.court_start_y) + convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2, constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        # point 3
        drawing_kps[6]=drawing_kps[0] + self.court_drawing_width
        drawing_kps[7]=drawing_kps[5]
        # point 4
        drawing_kps[8]=drawing_kps[0]+convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE, constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        drawing_kps[9]=drawing_kps[1]
        # point 5
        drawing_kps[10]=drawing_kps[4]+convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE, constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        drawing_kps[11]=drawing_kps[5]
        #point 6
        drawing_kps[12] = drawing_kps[2]-convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE, constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        drawing_kps[13] = drawing_kps[3] 
        #point 7
        drawing_kps[14] = drawing_kps[6]-convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE, constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        drawing_kps[15] = drawing_kps[7] 
        #point 8
        drawing_kps[16] = drawing_kps[8] 
        drawing_kps[17] = drawing_kps[9]+convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT, constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        #point 9
        drawing_kps[18] = drawing_kps[16]+convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH, constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        drawing_kps[19] = drawing_kps[17] 
        #point 10
        drawing_kps[20] = drawing_kps[10] 
        drawing_kps[21] = drawing_kps[11]-convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT, constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        #point 11
        drawing_kps[22] = drawing_kps[20]+convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH, constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        drawing_kps[23] = drawing_kps[21]
        #point 12
        drawing_kps[24] = int((drawing_kps[16] + drawing_kps[18])/2)
        drawing_kps[25] = drawing_kps[17] 
        #point 13
        drawing_kps[24] = int((drawing_kps[20] + drawing_kps[22])/2)
        drawing_kps[25] = drawing_kps[21] 

        self.drawing_kps = drawing_kps

    #connect the points
    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]
    #drawing the white rectangle
    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes,(self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out
    
    #drawing kps in small court
    def draw_court(self, frame):
        #drawing the points
        for i in range (0,len(self.drawing_kps),2):
            cv2.circle(frame, (int(self.drawing_kps[i]), int(self.drawing_kps[i+1])), 5, (255, 0, 255), -1)

        #drawing the lines
        for line in self.lines:
            cv2.line(frame, (int(self.drawing_kps[line[0]*2]), int(self.drawing_kps[line[0]*2+1])), (int(self.drawing_kps[line[1]*2]), int(self.drawing_kps[line[1]*2+1])), (0, 0, 0), 2)

        #drawing the net
        start_pos = (self.drawing_kps[0],int((self.drawing_kps[1]+self.drawing_kps[5])/2))
        end_pos = (self.drawing_kps[2],int((self.drawing_kps[1]+self.drawing_kps[5])/2))
        cv2.line(frame, start_pos, end_pos, (0, 0, 0), 2)
        return frame

    #looping over frames to draw small court and kps
    def draw_small_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_small_court(self):
        return (self.court_start_x,self.court_start_y)
    
    def get_width_of_small_court(self):
        return self.court_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_kps
    
    def convert_meters_to_pixels_small_court(self, meters):
        return convert_meters_to_pixels(meters,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
    

    def convert_bbox_to_small_court_coordinates(self,player_bboxes, ball_bboxes, orignial_court_kps ):

       output_player_boxes = []
       output_ball_boxes=[]

       player_heights= {
            1:constants.PLAYER_1_HEIGHT_METERS, 
            2:constants.PLAYER_2_HEIGHT_METERS
        }

       for frame, player_bbox in enumerate(player_bboxes):
            ball_box = ball_bboxes[frame][1]
            ball_position = get_center_of_bbox(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance (ball_position, get_center_of_bbox(player_bbox[x])))


            output_player_bboxes_dict = {}
            # looping over players
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)
                #get the closet kps in pixels
                closest_court_kp_index = get_closest_court_keypoint(foot_position,orignial_court_kps, [0,2,12,13])
                closest_kps = (orignial_court_kps[closest_court_kp_index * 2],orignial_court_kps[closest_court_kp_index * 2 + 1])

                #get the height of the player in pixels
                #to ensure that frame_min_indix  is at least 0 so it doesn't go out of bounds (negative) at the start of the video.
                frame_min_indix = max(0,frame-20)
                #to ensure that frame_index_max does not exceed the total number of frames 
                frame_max_indix = min(len(player_bboxes),frame+50)

                bboxes_height_pixels = [get_height_of_bbox(player_bboxes[i][player_id]) for i in range (frame_min_indix,frame_max_indix)]
                max_player_height_pixels = max(bboxes_height_pixels)
                

                small_court_player_position = self.get_small_court_coordinates(foot_position,closest_kps,closest_court_kp_index,max_player_height_pixels,player_heights[player_id])

                # start to draw ball in small court 
                output_player_bboxes_dict[player_id] = small_court_player_position

                if closest_player_id_to_ball == player_id:
                    #get the closet kps in pixels
                    closest_court_kp_index = get_closest_court_keypoint(ball_position,orignial_court_kps, [0,2,12,13])
                    closest_kps = (orignial_court_kps[closest_court_kp_index * 2],orignial_court_kps[closest_court_kp_index * 2 + 1])

                    #get the ball position in pixels
                    small_court_ball_position = self.get_small_court_coordinates(ball_position,closest_kps,closest_court_kp_index,max_player_height_pixels,player_heights[player_id])
                    output_ball_boxes.append({1:small_court_ball_position})

            output_player_boxes.append(output_player_bboxes_dict)

       return output_player_boxes,output_ball_boxes

   

    def get_small_court_coordinates (self,object_positions,closest_kp,closest_kp_index, player_height_pixels, player_height_meters):

        # calculate the distance betweeb the player position and the closest court keypoint in pixels and meters (Actual distance)

        distance_kp_x_pixels , distance_kp_y_pixels = measure_xy_distance(object_positions,closest_kp)

        distance_kp_x_meters = convert_pixels_to_meters(distance_kp_x_pixels,player_height_meters,player_height_pixels)
        distance_kp_y_meters = convert_pixels_to_meters(distance_kp_y_pixels,player_height_meters,player_height_pixels)

        # convert this distance that we calculated before into pxels in small court 
        small_court_x_distance_pixels  = self.convert_meters_to_pixels_small_court(distance_kp_x_meters)
        small_court_y_distance_pixels  = self.convert_meters_to_pixels_small_court(distance_kp_y_meters)

        closest_kp_small_court = (self.drawing_kps[closest_kp_index*2], self.drawing_kps[closest_kp_index*2+1])

        small_court_player_position = (closest_kp_small_court[0]+small_court_x_distance_pixels,
                                      closest_kp_small_court[1]+small_court_y_distance_pixels
                                        )
        return small_court_player_position
    
   

    def draw_points_on_small_court(self,frames,postions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in postions[frame_num].items():
                x,y = position
                cv2.circle(frame, (int(x),int(y)), 5, color, -1)
        return frames