import cv2  
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)  # Create a VideoCapture object to read the video file
    frames = []  # Initialize an empty list to store video frames
    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break  # If no frame is returned, end of video is reached
        frames.append(frame)  # Add the frame to the list of frames
    return frames  # Return the list of frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec using 'XVID' fourcc code
    out = cv2.VideoWriter(
        output_video_path,                  # Output file path
        fourcc,                             # Codec
        24,                                 # Frames per second
        (
            output_video_frames[0].shape[1],  # Frame width
            output_video_frames[0].shape[0]   # Frame height
        )
    )  # Create a VideoWriter object to write the video
    for frame in output_video_frames:
        out.write(frame)  # Write each frame to the output video
    out.release()  # Release the VideoWriter object
