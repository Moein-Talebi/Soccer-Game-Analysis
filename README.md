##  Soccer Game Analysis

This project is designed to detect and track players, referees, and footballs in video footage using YOLO, one of the most effective AI object detection models available. The objective is to train the model to enhance its performance. I also apply K-means clustering based on t-shirt colors to categorize players into their respective teams, allowing us to calculate each team's possession percentage during a match. Additionally, optical flow techniques will be employed to track camera movement between frames, enabling precise measurement of player movement. To accurately represent the scene's depth and scale, I will implement perspective transformation, converting measurements from pixels to meters, which will facilitate the calculation of player speed and the distance covered.

This project serves as a practical application of the concepts I've learned, inspired by various YouTube videos. It encompasses a range of techniques and addresses real-world challenges, making it a valuable experience for both novice and experienced machine learning engineers.                                      

## Image of output
![Screenshot](output_videos/screenshot.png)


## Modules Used in This Project

- **YOLO**: AI model for object detection.
- **Optical Flow**: Tracks camera movement.
- **Perspective Transformation**: Adds depth and perspective to scenes.
- **K-means**: Clustering for t-shirt color detection through pixel segmentation.
- **Speed and Distance Calculation**: Computes each player's speed and distance covered.

## Requirements
