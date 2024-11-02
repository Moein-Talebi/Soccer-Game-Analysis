def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox  # Unpack the bounding box coordinates
    return int((x1 + x2) / 2), int((y1 + y2) / 2)  # Calculate and return the center point of the bounding box

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]  # Calculate and return the width of the bounding box

def measure_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5  # Calculate and return the Euclidean distance between two points

def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]  # Calculate and return the x and y distances between two points

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox  # Unpack the bounding box coordinates
    return int((x1 + x2) / 2), int(y2)  # Calculate and return the foot position (center of the bottom edge) of the bounding box