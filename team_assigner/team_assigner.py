from sklearn.cluster import KMeans  

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}  # Dictionary to store team colors
        self.player_team_dict = {}  # Dictionary to store player-team assignments
    
    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)  # Reshape the image to a 2D array with 3 color channels

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)  # Initialize KMeans with 2 clusters
        kmeans.fit(image_2d)  # Fit the KMeans model to the image data

        return kmeans  # Return the fitted KMeans model

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  # Extract the bounding box region from the frame

        top_half_image = image[0:int(image.shape[0] / 2), :]  # Get the top half of the bounding box image

        kmeans = self.get_clustering_model(top_half_image)  # Get the clustering model for the top half image

        labels = kmeans.labels_  # Get the cluster labels for each pixel

        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])  # Reshape the labels to the image shape

        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]  # Get the clusters of the corners
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)  # Determine the non-player cluster
        player_cluster = 1 - non_player_cluster  # Determine the player cluster

        player_color = kmeans.cluster_centers_[player_cluster]  # Get the color of the player cluster

        return player_color  # Return the player color

    def assign_team_color(self, frame, player_detections):
        player_colors = []  # List to store player colors
        for _, player_detection in player_detections.items():  # Iterate over player detections
            bbox = player_detection["bbox"]  # Get the bounding box of the player
            player_color = self.get_player_color(frame, bbox)  # Get the player color
            player_colors.append(player_color)  # Append the player color to the list
        
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)  # Initialize KMeans with 2 clusters
        kmeans.fit(player_colors)  # Fit the KMeans model to the player colors

        self.kmeans = kmeans  # Store the fitted KMeans model

        self.team_colors[1] = kmeans.cluster_centers_[0]  # Assign the first cluster center to team 1
        self.team_colors[2] = kmeans.cluster_centers_[1]  # Assign the second cluster center to team 2

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:  # Check if the player is already assigned to a team
            return self.player_team_dict[player_id]  # Return the assigned team

        player_color = self.get_player_color(frame, player_bbox)  # Get the player color

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]  # Predict the team ID using the KMeans model
        team_id += 1  # Increment the team ID to match the team_colors dictionary

        if player_id == 91:  # Special case for player ID 91
            team_id = 1  # Assign team 1 to player ID 91

        self.player_team_dict[player_id] = team_id  # Store the player-team assignment

        return team_id  # Return the team ID
