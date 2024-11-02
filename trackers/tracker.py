from ultralytics import YOLO  
import supervision as sv       
import pickle                  
import os                      
import numpy as np             
import pandas as pd            
import cv2                     
import sys                    
import gc        

sys.path.append('../')         # Add the parent directory to the system path
from tools import get_center_of_bbox, get_bbox_width, get_foot_position  # Import utility functions

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # Initialize the YOLO model with the given path
        self.tracker = sv.ByteTrack()  # Initialize the ByteTrack tracker

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']  # Get the bounding box
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)  # Get center position for the ball
                    else:
                        position = get_foot_position(bbox)  # Get foot position for players and referees
                    tracks[object][frame_num][track_id]['position'] = position  # Add position to track info

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]  # Extract ball bounding boxes
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])  # Create DataFrame

        df_ball_positions = df_ball_positions.interpolate()  # Interpolate missing values
        df_ball_positions = df_ball_positions.bfill()        # Backfill any remaining NaNs

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]  # Convert back to list

        return ball_positions  # Return the interpolated ball positions

    def detect_frames(self, frames):
        batch_size = 10  # Set batch size for processing frames
        detections = []  # Initialize list to store detections
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)  # Predict on batch
            detections += detections_batch  # Add batch detections to the list
        return detections  # Return all detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)  # Load tracks from stub file if available
            return tracks  # Return loaded tracks

        detections = self.detect_frames(frames)  # Detect objects in frames

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }  # Initialize tracks dictionary

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # Get class names
            cls_names_inv = {v: k for k, v in cls_names.items()}  # Invert class names dictionary

            detection_supervision = sv.Detections.from_ultralytics(detection)  # Convert to supervision format

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]  # Convert goalkeeper to player

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)  # Update tracker

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()  # Get bounding box
                cls_id = frame_detection[3]         # Get class ID
                track_id = frame_detection[4]       # Get track ID

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}  # Add player track
                    
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}  # Add referee track

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()  # Get bounding box
                cls_id = frame_detection[3]         # Get class ID

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}  # Add ball track

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)  # Save tracks to stub file

        return tracks  # Return tracks dictionary

    def draw_ellipse(self, frame, bbox, color=(0,0,255), track_id=None):
        y2 = int(bbox[3])  # Get bottom y-coordinate of bounding box
        x_center, _ = get_center_of_bbox(bbox)  # Get center x-coordinate
        width = get_bbox_width(bbox)  # Get width of bounding box

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )  # Draw an ellipse on the frame

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2  # Calculate rectangle coordinates
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )  # Draw a filled rectangle for the track ID label

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10  # Adjust text position for larger IDs

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )  # Put the track ID text on the frame

        return frame  # Return the modified frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])  # Get top y-coordinate of bounding box
        x, _ = get_center_of_bbox(bbox)  # Get center x-coordinate

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])  # Define triangle points

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)  # Draw filled triangle
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)       # Outline the triangle

        return frame  # Return the modified frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()  # Create a copy of the frame for overlay
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)  # Draw a white rectangle

        alpha = 0.4  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Blend the overlay with the frame

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]  # Get ball control data up to current frame

        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]  # Frames controlled by Team 1
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]  # Frames controlled by Team 2

        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)  # Calculate control percentage
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1 * 100:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3
        )  # Display Team 1 control percentage

        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2 * 100:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3
        )  # Display Team 2 control percentage

        return frame  # Return the modified frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []  # Initialize list for output frames
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # Copy the frame to avoid modifying the original

            player_dict = tracks["players"][frame_num]    # Get player tracks for the frame
            ball_dict = tracks["ball"][frame_num]         # Get ball tracks for the frame
            referee_dict = tracks["referees"][frame_num]  # Get referee tracks for the frame

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))  # Get team color or default to red
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)  # Draw ellipse around player

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 255, 0))  # Draw triangle if player has ball

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))  # Draw ellipse around referee

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 0, 255))  # Draw triangle for the ball

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)  # Draw ball control stats

            output_video_frames.append(frame)  # Add the annotated frame to the output list

        return output_video_frames  # Return the list of annotated frames