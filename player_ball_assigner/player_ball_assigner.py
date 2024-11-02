import sys 
sys.path.append('../')  # Add the parent directory to the system path
from tools import get_center_of_bbox, measure_distance  # Import utility functions

class PlayerBallAssigner():  # Define the PlayerBallAssigner class
    def __init__(self):  # Initialize the class
        self.max_player_ball_distance = 70  # Set the maximum distance to assign the ball to a player

    def assign_ball_to_player(self, players, ball_bbox):  # Define method to assign ball to a player
        ball_position = get_center_of_bbox(ball_bbox)  # Get the center position of the ball

        miniumum_distance = 99999  # Initialize minimum distance with a large number
        assigned_player = -1  # Initialize assigned player as -1 (no player)

        for player_id, player in players.items():  # Iterate over each player
            player_bbox = player['bbox']  # Get player's bounding box

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)  # Calculate distance from player's left foot to ball
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)  # Calculate distance from player's right foot to ball
            distance = min(distance_left, distance_right)  # Choose the smaller distance

            if distance < self.max_player_ball_distance:  # Check if distance is within maximum limit
                if distance < miniumum_distance:  # Check if this is the minimum distance found so far
                    miniumum_distance = distance  # Update minimum distance
                    assigned_player = player_id  # Assign player ID

        return assigned_player  # Return the assigned player's ID