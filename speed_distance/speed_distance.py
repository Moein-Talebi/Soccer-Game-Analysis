import cv2  
import sys  
sys.path.append('../')  # Add the parent directory to the system path
from tools import measure_distance, get_foot_position  # Import specific functions from the utils module

class SpeedAndDistance_Estimator():  # Define a class for estimating speed and distance
    def __init__(self):  # Initialize the class
        self.frame_window = 5  # Set the frame window size
        self.frame_rate = 24  # Set the frame rate
    
    def add_speed_and_distance_to_tracks(self, tracks):  # Define a method to add speed and distance to tracks
        total_distance = {}  # Initialize a dictionary to store total distances

        for object, object_tracks in tracks.items():  # Iterate over each object and its tracks
            if object == "ball" or object == "referees":  # Skip if the object is "ball" or "referees"
                continue 
            number_of_frames = len(object_tracks)  # Get the number of frames for the object
            for frame_num in range(0, number_of_frames, self.frame_window):  # Iterate over frames in steps of frame_window
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)  # Calculate the last frame in the window

                for track_id, _ in object_tracks[frame_num].items():  # Iterate over each track in the current frame
                    if track_id not in object_tracks[last_frame]:  # Skip if the track_id is not in the last frame
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']  # Get the start position
                    end_position = object_tracks[last_frame][track_id]['position_transformed']  # Get the end position

                    if start_position is None or end_position is None:  # Skip if start or end position is None
                        continue
                    
                    distance_covered = measure_distance(start_position, end_position)  # Calculate the distance covered
                    time_elapsed = (last_frame - frame_num) / self.frame_rate  # Calculate the time elapsed
                    speed_meteres_per_second = distance_covered / time_elapsed  # Calculate the speed in meters per second
                    speed_km_per_hour = speed_meteres_per_second * 3.6  # Convert the speed to kilometers per hour

                    if object not in total_distance:  # Check if the object is not in the total_distance dictionary
                        total_distance[object] = {}  # Initialize the dictionary for the object
                    
                    if track_id not in total_distance[object]:  # Check if the track_id is not in the object's dictionary
                        total_distance[object][track_id] = 0  # Initialize the distance for the track_id
                    
                    total_distance[object][track_id] += distance_covered  # Add the distance covered to the total distance for the track_id

                    for frame_num_batch in range(frame_num, last_frame):  # Iterate over each frame in the batch
                        if track_id not in tracks[object][frame_num_batch]:  # Check if the track_id is not in the tracks for the current frame
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour  # Update the speed for the track_id in the current frame
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]  # Update the distance for the track_id in the current frame
    
    def draw_speed_and_distance(self, frames, tracks):  # Define a method to draw speed and distance on frames
        output_frames = []  # Initialize a list to store output frames
        for frame_num, frame in enumerate(frames):  # Iterate over each frame
            for object, object_tracks in tracks.items():  # Iterate over each object and its tracks
                if object == "ball" or object == "referees":  # Skip if the object is "ball" or "referees"
                    continue 
                for _, track_info in object_tracks[frame_num].items():  # Iterate over each track info in the current frame
                   if "speed" in track_info:  # Check if speed is in the track info
                       speed = track_info.get('speed', None)  # Get the speed from the track info
                       distance = track_info.get('distance', None)  # Get the distance from the track info
                       if speed is None or distance is None:  # Skip if speed or distance is None
                           continue
                       
                       bbox = track_info['bbox']  # Get the bounding box from the track info
                       position = get_foot_position(bbox)  # Get the foot position from the bounding box
                       position = list(position)  # Convert the position to a list
                       position[1] += 40  # Adjust the y-coordinate of the position

                       position = tuple(map(int, position))  # Convert the position back to a tuple of integers
                       cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Draw the speed on the frame
                       cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Draw the distance on the frame
            output_frames.append(frame)  # Add the frame to the output frames list
        
        return output_frames  # Return the output frames