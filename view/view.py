import numpy as np  
import cv2  

class ViewTransformer():
    def __init__(self):
        court_width = 68  # Define the width of the court
        court_length = 23.32  # Define the length of the court

        self.pixel_vertices = np.array([[110, 1035], 
                               [265, 275], 
                               [910, 260], 
                               [1640, 915]])  # Define the vertices of the court in pixel coordinates
        
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])  # Define the vertices of the court in real-world coordinates

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)  # Convert pixel vertices to float32 type
        self.target_vertices = self.target_vertices.astype(np.float32)  # Convert target vertices to float32 type

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)  # Compute the perspective transformation matrix

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))  # Convert the point to integer coordinates
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0  # Check if the point is inside the polygon defined by pixel vertices
        if not is_inside:
            return None  # Return None if the point is outside the polygon

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)  # Reshape and convert the point to float32 type
        tranform_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)  # Apply the perspective transformation
        return tranform_point.reshape(-1, 2)  # Reshape the transformed point and return it

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():  # Iterate over each object and its tracks
            for frame_num, track in enumerate(object_tracks):  # Iterate over each frame and its track
                for track_id, track_info in track.items():  # Iterate over each track ID and its information
                    position = track_info['position_adjusted']  # Get the adjusted position
                    position = np.array(position)  # Convert the position to a numpy array
                    position_trasnformed = self.transform_point(position)  # Transform the position
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()  # Squeeze and convert the transformed position to a list
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed  # Update the track information with the transformed position