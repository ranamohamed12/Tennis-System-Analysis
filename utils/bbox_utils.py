def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    x_centre = (x1+x2)/2
    y_centre = (y1+y2)/2
    return (x_centre,y_centre)

def measure_distance (p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_foot_position (bbox):
    x1,y1,x2,y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_court_keypoint(point,court_kps,kps_indices):
    min_dist = float('inf')
    closest_index  = kps_indices[0]

    for kps_indix in kps_indices:
        #get the coordinates of the point
        kp = (court_kps[kps_indix * 2], court_kps[kps_indix * 2 + 1])
        #taking the second parmeter because we calculte the difference between the y coordinates
        dist = abs(point[1]-kp[1])
        if dist < min_dist:
            min_dist = dist
            closest_index  = kps_indix

    return closest_index
    
def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])


# def get_center_of_bbox(bbox):
#     return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))