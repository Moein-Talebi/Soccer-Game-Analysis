from tools import read_video, save_video  # Import functions to read and save video files
from trackers import Tracker  # Import the Tracker class for object tracking
import cv2  
import numpy as np  
from team_assigner import TeamAssigner  # Import the TeamAssigner class
from player_ball_assigner import PlayerBallAssigner  # Import the PlayerBallAssigner class
from camera_movement import CameraMovementEstimator  # Import the CameraMovementEstimator class
from view import ViewTransformer  # Import the ViewTransformer class
from speed_distance import SpeedAndDistance_Estimator  # Import the SpeedAndDistance_Estimator class


def main():
    # Read video frames from the input video file
    video_frames = read_video('input_videos/NWANERI_WITH_A_WORLDIE!_Preston_vS_Arsenal_0_3_Carabao_Cup - Trim.mp4')

    # Initialize the Tracker with a pre-trained model
    tracker = Tracker('models/best.pt')

    # Get object tracks from the video frames (using stub data if available)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )

    # Add position data to the object tracks
    tracker.add_position_to_tracks(tracks)

    # Initialize the CameraMovementEstimator with the first frame
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])

    # Estimate camera movement for each frame (using stub data if available)
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )

    # Adjust positions in the tracks based on camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks,
        camera_movement_per_frame
    )

    # Initialize the ViewTransformer
    view_transformer = ViewTransformer()

    # Add transformed positions to the tracks
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate missing ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Initialize the SpeedAndDistance_Estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()

    # Add speed and distance data to the tracks
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Initialize the TeamAssigner
    team_assigner = TeamAssigner()

    # Assign team colors using the first frame and initial player tracks
    team_assigner.assign_team_color(
        video_frames[0],
        tracks['players'][0]
    )

    # Assign team information to each player's track data
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Initialize the PlayerBallAssigner
    player_assigner = PlayerBallAssigner()
    team_ball_control = []  # List to track which team controls the ball

    # Assign the ball to players in each frame
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            # Mark that the player has the ball
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            # Append the team controlling the ball
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # If no player is assigned, keep the last known team control
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw annotations on the video frames
    output_video_frames = tracker.draw_annotations(
        video_frames,
        tracks,
        team_ball_control
    )

    # Draw camera movement on the video frames
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames,
        camera_movement_per_frame
    )

    # Draw speed and distance information on the video frames
    speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames,
        tracks
    )

    # Save the annotated video frames to an output video file
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()