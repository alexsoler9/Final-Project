import pandas as pd
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_csv(filename):
    FILE_PATH = Path("../Project_v2/data/"+filename)
    if not FILE_PATH.exists():
        raise Exception(f"The path '{FILE_PATH.as_posix()}' is not recognized.")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(FILE_PATH)
    return df

def unique_ID_dict(dataframe, class_ids=None):
    """
    Computes unique tracker IDs, their appearances and corresponding class IDs in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        clas_ids (list, optional): List of class IDs to filter by (default is None)

    Returns:
        pd.DataFrame: DataFrame with unique tracker IDs, appearances and class IDs.
    """
    if "tracker_id" not in dataframe.columns or "class_id" not in dataframe.columns:
        raise ValueError("Dataframe must contain 'tracker_id' and 'class_id' columns.")
    
    if class_ids:
        filtered_df = dataframe[dataframe["class_id"].isin(class_ids)]
    else:
        filtered_df = dataframe

    unique_tracker_ids, counts = np.unique(filtered_df["tracker_id"], return_counts=True)

    # Create a DataFrame
    tracker_id_counts = pd.DataFrame({
            "Unique IDs": unique_tracker_ids,
            "Appearances": counts,
            "Class IDs": filtered_df.groupby("tracker_id")["class_id"].first().values
            })
    return tracker_id_counts

def get_ids_by_appearances(df, num_appearances):
    # Filter rows where Appearances are less than or equal to the specified value
    filtered_df = df[df["Appearances"] >= num_appearances]

    # Get the unique IDs from the filtered DataFrame
    unique_ids = filtered_df["Unique IDs"].tolist()
    
    return unique_ids

def filter_data_by_ids(dataframe, id_list):
    """
    Filter the DataFrame to include only rows with specified IDs.
    
    Args:
        dataframe (pd.Dataframe): The input DataFrame.
        id_list (list): List of IDs to filter by.
        
    Returns:
        pd.Dataframe: Filtered DataFrame containing rows with specified IDs.
    """
    filtered_df = dataframe[dataframe["tracker_id"].isin(id_list)]
    return filtered_df

def combine_with_person(dataframe):
    """
    Combines person and bike/motorbike detections based on intersection of bounding boxes.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing bounding box information.

    Returns:
        pd.DataFrame: New DataFrame with combined detections.

    """

    combined_detections = []

    for frame_number, frame_group in dataframe.groupby("Frame"):
        # Get bounding boxes for persons and bikes/motorbikes
        person_boxes = frame_group[frame_group["class_id"] == 0][["x_min", "y_min", "x_max", "y_max","class_id","confidence","tracker_id","Frame","class_name"]]
        bike_boxes = frame_group[frame_group["class_id"].isin([1,3])][["x_min", "y_min", "x_max", "y_max", "class_id","confidence","tracker_id","Frame","class_name"]]
        
        # Compute intersection between person and bike/motorbike bounding boxes
        for _, person_box in person_boxes.iterrows():
            for _, bike_box in bike_boxes.iterrows():
                x_min = max(person_box["x_min"], bike_box["x_min"])
                y_min = max(person_box["y_min"], bike_box["y_min"])
                x_max = max(person_box["x_max"], bike_box["x_max"])
                y_max = max(person_box["y_max"], bike_box["y_max"])
                person_center = person_box["x_min"] + (person_box["x_max"] - person_box["x_min"])/2
                bike_area = (bike_box["x_max"]-bike_box["x_min"])* (bike_box["y_max"]-bike_box["y_min"])
                person_area = (person_box["x_max"]-person_box["x_min"])* (person_box["y_max"]-person_box["y_min"])
                if (bike_box["x_min"] < person_center < bike_box["x_max"] and person_box["y_max"] > bike_box["y_min"] 
                    and 0.5 < person_area/bike_area < 3 ):
                    # Intersection exists, create a new combined detection
                    combined_detections.append({
                        "x_min": min(person_box["x_min"], bike_box["x_min"]),
                        "y_min": min(person_box["y_min"], bike_box["y_min"]),
                        "x_max": max(person_box["x_max"], bike_box["x_max"]),
                        "y_max": max(person_box["y_max"], bike_box["y_max"]),
                        "class_id": 81,
                        "class_name": "rider",
                        "confidence": person_box["confidence"],
                        "tracker_id": [person_box["tracker_id"], bike_box["tracker_id"]],
                        "Frame": frame_number
                    })
        
    combined_df = pd.DataFrame(combined_detections)
        
    return combined_df

def combine_person_and_bike(dataframe):
    """
    Combines person and bike/motorbike detections based on intersection of bounding boxes.
    Retruns an updated dataframe with all detections.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing bounding box information.

    Returns:
        pd.DataFrame: New DataFrame with combined detections.
    """

    combined_detections = []

    for frame_number, frame_group in dataframe.groupby("Frame"):
        # Get bounding boxes for persons and bikes/motorbikes
        person_boxes = frame_group[frame_group["class_id"] == 0][["x_min", "y_min", "x_max", "y_max", "class_id", "confidence", "tracker_id", "Frame", "class_name"]]
        bike_boxes = frame_group[frame_group["class_id"].isin([1, 3])][["x_min", "y_min", "x_max", "y_max", "class_id", "confidence", "tracker_id", "Frame", "class_name"]]

        person_added = set()  # Keep track of person boxes already added in the current frame
        bike_added = set()    # Keep track of bike boxes already added in the current frame
        # Compute intersection between person and bike/motorbike bounding boxes
        for _, person_box in person_boxes.iterrows():
            
            for _, bike_box in bike_boxes.iterrows():

                x_min = max(person_box["x_min"], bike_box["x_min"])
                y_min = max(person_box["y_min"], bike_box["y_min"])
                x_max = max(person_box["x_max"], bike_box["x_max"])
                y_max = max(person_box["y_max"], bike_box["y_max"])
                person_center = person_box["x_min"] + (person_box["x_max"] - person_box["x_min"]) / 2
                bike_area = (bike_box["x_max"] - bike_box["x_min"]) * (bike_box["y_max"] - bike_box["y_min"])
                person_area = (person_box["x_max"] - person_box["x_min"]) * (person_box["y_max"] - person_box["y_min"])

                if (bike_box["x_min"] < person_center < bike_box["x_max"] and person_box["y_max"] > bike_box["y_min"]
                        and 0.5 < person_area / bike_area < 3):
                    # Intersection exists, create a new combined detection
                    combined_detections.append({
                        "x_min": min(person_box["x_min"], bike_box["x_min"]),
                        "y_min": min(person_box["y_min"], bike_box["y_min"]),
                        "x_max": max(person_box["x_max"], bike_box["x_max"]),
                        "y_max": max(person_box["y_max"], bike_box["y_max"]),
                        "class_id": 81,
                        "class_name": "rider",
                        "confidence": person_box["confidence"],
                        "tracker_id": [person_box["tracker_id"], bike_box["tracker_id"]],
                        "Frame": frame_number
                    })
                    person_added.add(person_box["tracker_id"])
                    bike_added.add(bike_box["tracker_id"])
                    break
        
        # Add the boxes for the current frame that have not been added        
        for _, person_box in person_boxes.iterrows():
            if person_box["tracker_id"] not in person_added:
                combined_detections.append({
                    "x_min": person_box["x_min"],
                    "y_min": person_box["y_min"],
                    "x_max": person_box["x_max"],
                    "y_max": person_box["y_max"],
                    "class_id": person_box["class_id"],
                    "class_name": person_box["class_name"],
                    "confidence": person_box["confidence"],
                    "tracker_id": person_box["tracker_id"],
                    "Frame": frame_number
                })
        for _, bike_box in bike_boxes.iterrows():
            if bike_box["tracker_id"] not in bike_added:
                combined_detections.append({
                    "x_min": bike_box["x_min"],
                    "y_min": bike_box["y_min"],
                    "x_max": bike_box["x_max"],
                    "y_max": bike_box["y_max"],
                    "class_id": bike_box["class_id"],
                    "class_name": bike_box["class_name"],
                    "confidence": bike_box["confidence"],
                    "tracker_id": bike_box["tracker_id"],
                    "Frame": frame_number
                })
        
    combined_df = pd.DataFrame(combined_detections)

    return combined_df

def update_dataframe(dataframe):
    """
    Combines person and bike/motorbike detections based on intersection of bounding boxes.
    Returns an updated dataframe with all detections.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing bounding box information.

    Returns:
        pd.DataFrame: New DataFrame with combined detections.
    """

    updated_detections = []

    for frame_number, frame_group in dataframe.groupby("Frame"):
        # Get bounding boxes for persons and bikes/motorbikes
        person_boxes = frame_group[frame_group["class_id"] == 0][["x_min", "y_min", "x_max", "y_max", "class_id", "confidence", "tracker_id", "Frame", "class_name"]]
        bike_boxes = frame_group[frame_group["class_id"].isin([1, 3])][["x_min", "y_min", "x_max", "y_max", "class_id", "confidence", "tracker_id", "Frame", "class_name"]]
        other_boxes = frame_group[~frame_group["class_id"].isin([0,1,3])]

        person_added = set()  # Keep track of person boxes already added in the current frame
        bike_added = set()    # Keep track of bike boxes already added in the current frame

        # Compute intersection between person and bike/motorbike bounding boxes
        for _, person_box in person_boxes.iterrows():
            for _, bike_box in bike_boxes.iterrows():

                # x_min = max(person_box["x_min"], bike_box["x_min"])
                # y_min = max(person_box["y_min"], bike_box["y_min"])
                # x_max = max(person_box["x_max"], bike_box["x_max"])
                # y_max = max(person_box["y_max"], bike_box["y_max"])
                person_center = person_box["x_min"] + (person_box["x_max"] - person_box["x_min"]) / 2
                bike_area = (bike_box["x_max"] - bike_box["x_min"]) * (bike_box["y_max"] - bike_box["y_min"])
                person_area = (person_box["x_max"] - person_box["x_min"]) * (person_box["y_max"] - person_box["y_min"])

                if (bike_box["x_min"] < person_center < bike_box["x_max"] and person_box["y_max"] > bike_box["y_min"]
                        and 0.5 < person_area / bike_area < 3):
                    # Intersection exists, create a new combined detection
                    updated_detections.append({
                        "x_min": min(person_box["x_min"], bike_box["x_min"]),
                        "y_min": min(person_box["y_min"], bike_box["y_min"]),
                        "x_max": max(person_box["x_max"], bike_box["x_max"]),
                        "y_max": max(person_box["y_max"], bike_box["y_max"]),
                        "class_id": 81,
                        "class_name": "rider",
                        "confidence": person_box["confidence"],
                        "tracker_id": [person_box["tracker_id"], bike_box["tracker_id"]],
                        "Frame": frame_number
                    })
                    person_added.add(person_box["tracker_id"])
                    bike_added.add(bike_box["tracker_id"])
                    break
        
        # Add the boxes for the current frame that have not been added        
        for _, person_box in person_boxes.iterrows():
            if person_box["tracker_id"] not in person_added:
                updated_detections.append({
                    "x_min": person_box["x_min"],
                    "y_min": person_box["y_min"],
                    "x_max": person_box["x_max"],
                    "y_max": person_box["y_max"],
                    "class_id": person_box["class_id"],
                    "class_name": person_box["class_name"],
                    "confidence": person_box["confidence"],
                    "tracker_id": person_box["tracker_id"],
                    "Frame": frame_number
                })
        for _, bike_box in bike_boxes.iterrows():
            if bike_box["tracker_id"] not in bike_added:
                updated_detections.append({
                    "x_min": bike_box["x_min"],
                    "y_min": bike_box["y_min"],
                    "x_max": bike_box["x_max"],
                    "y_max": bike_box["y_max"],
                    "class_id": bike_box["class_id"],
                    "class_name": bike_box["class_name"],
                    "confidence": bike_box["confidence"],
                    "tracker_id": bike_box["tracker_id"],
                    "Frame": frame_number
                })
        # Add the other boxes that are neither person nor bike/motorbike
        for _, other_box in other_boxes.iterrows():
            updated_detections.append(other_box.to_dict())

    updated_df = pd.DataFrame(updated_detections)

    return updated_df

def update_dataframe_improved(dataframe):
    """
    Combines person and bike/motorbike detections based on intersection of bounding boxes.
    Returns an updated dataframe with all detections.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame containing bounding box information.
        
    Returns:
        pd.DataFrame: New DataFrame with combined detections.
    """

    updated_detections = []

    for frame_number, frame_group in dataframe.groupby("Frame"):
        # Get the bounding boxes for persons and bikes/motorbikes
        person_boxes = frame_group[frame_group["class_id"] == 0][["x_min", "y_min", "x_max", "y_max", "class_id", "confidence", "tracker_id", "Frame", "class_name"]]
        bike_boxes = frame_group[frame_group["class_id"].isin([1, 3])][["x_min", "y_min", "x_max", "y_max", "class_id", "confidence", "tracker_id", "Frame", "class_name"]]
        # Get the other bounding boxes
        other_boxes = frame_group[~frame_group["class_id"].isin([0,1,3])]
    
        person_added = set() # Keep track of person boxes already added in the current frame
        bike_added = set() # Keep track of bike boxes already added in the current frame

        # Store the best match for each bike
        bike_person_matches = {}

        # Compute intersection and distances
        for _, bike_box in bike_boxes.iterrows():
            bike_center_x = bike_box["x_min"] + (bike_box["x_max"]-bike_box["x_min"])/2
            bike_center_y = bike_box["y_min"] + (bike_box["y_max"]-bike_box["y_min"])/2
            bike_area = (bike_box["x_max"] - bike_box["x_min"]) * (bike_box["y_max"] -  bike_box["y_min"])

            for _, person_box in person_boxes.iterrows():
                person_center_x = person_box["x_min"] + (person_box["x_max"] - person_box["x_min"])/2
                person_center_y = person_box["y_min"] + (person_box["y_max"] - person_box["y_min"])/2
                person_area = (person_box["x_max"] - person_box["x_min"]) * (person_box["y_max"] -  person_box["y_min"])

                if (bike_box["x_min"] < person_center_x < bike_box["x_max"] and person_box["y_max"] > bike_box["y_min"]
                        and 0.5 < person_area / bike_area < 3):
                    # Calculate distance between centers
                    distance = ((bike_center_x - person_center_x)**2 + (bike_center_y - person_center_y)**2) ** 0.5

                    if (bike_box["tracker_id"] not in bike_person_matches or 
                        bike_person_matches[bike_box["tracker_id"]][1] > distance):
                        bike_person_matches[bike_box["tracker_id"]] = (person_box, distance)

        # Create combined detections based on the best matches
        for bike_tracker_id, (person_box, _) in bike_person_matches.items():
            bike_box = bike_boxes[bike_boxes["tracker_id"] == bike_tracker_id].iloc[0]
            updated_detections.append({
                "x_min": min(person_box["x_min"], bike_box["x_min"]),
                "y_min": min(person_box["y_min"], bike_box["y_min"]),
                "x_max": max(person_box["x_max"], bike_box["x_max"]),
                "y_max": max(person_box["y_max"], bike_box["y_max"]),
                "class_id": 81,
                "class_name": "rider",
                "confidence": person_box["confidence"],
                "tracker_id": [person_box["tracker_id"], bike_box["tracker_id"]],
                "Frame": frame_number
            })
            person_added.add(person_box["tracker_id"])
            bike_added.add(bike_tracker_id)
        
        # Add the boxes for the current frame that have not been added        
        for _, person_box in person_boxes.iterrows():
            if person_box["tracker_id"] not in person_added:
                updated_detections.append({
                    "x_min": person_box["x_min"],
                    "y_min": person_box["y_min"],
                    "x_max": person_box["x_max"],
                    "y_max": person_box["y_max"],
                    "class_id": person_box["class_id"],
                    "class_name": person_box["class_name"],
                    "confidence": person_box["confidence"],
                    "tracker_id": person_box["tracker_id"],
                    "Frame": frame_number
                })
        for _, bike_box in bike_boxes.iterrows():
            if bike_box["tracker_id"] not in bike_added:
                updated_detections.append({
                    "x_min": bike_box["x_min"],
                    "y_min": bike_box["y_min"],
                    "x_max": bike_box["x_max"],
                    "y_max": bike_box["y_max"],
                    "class_id": bike_box["class_id"],
                    "class_name": bike_box["class_name"],
                    "confidence": bike_box["confidence"],
                    "tracker_id": bike_box["tracker_id"],
                    "Frame": frame_number
                })
        # Add the other boxes that are neither person nor bike/motorbike
        for _, other_box in other_boxes.iterrows():
            updated_detections.append(other_box.to_dict())

    updated_df = pd.DataFrame(updated_detections)

    return updated_df

def group_riders_trackers_id(dataframe):
    """
    Groups rider tracker IDs.

    Args: 
        dataframe (pd.DataFrame)

    Returns:
        list[list[int]]: List of lists, where each inner list represents a group of rider trackrs IDs
    """

    # Initialize an empty dictionary to store rider trackr IDs and corresponding group
    rider_groups = {}

    for _, row in dataframe.iterrows():
        if row["class_name"] == "rider":
            tracker_ids = row["tracker_id"]

            # Use the first tracker ID as the group identifier
            group_identifier = tracker_ids[0]

            # Check if the group identifier is already in any group
            in_group = False
            for group in rider_groups.values():
                if group_identifier in group:
                    in_group = True
                    break

            if not in_group:
                # Create a new group for the tracker ID
                rider_groups[group_identifier] = set(tracker_ids)

                # Search for other rider detections with overlapping tracker IDs
                for _, other_row in dataframe.iterrows():
                    if other_row["class_name"] == "rider":
                        other_tracker_ids = other_row["tracker_id"]
                        for track_id in other_tracker_ids:
                            if track_id  in rider_groups[group_identifier]:
                                # Add the other tracker IDs to the same group
                                rider_groups[group_identifier].update(other_tracker_ids)
                                break
    return rider_groups

def propagate_detection(dataframe):
    rider_groups = group_riders_trackers_id(dataframe)

    new_rows = []

    for _, row in dataframe.iterrows():
        if row["class_name"] == "rider":
            tracker_ids = row["tracker_id"]
            for group_id, group_trackers in rider_groups.items():
                if any(track_id in group_trackers for track_id in tracker_ids):
                    # Update tracker IDs to the group identifier
                    row["tracker_id"] = group_id
                    row["class_id"] = 81
                    row["class_name"] = "rider"
                    break
        elif row["class_name"] in [ "person", "bycicle", "motorcycle"]:
            track_id = row["tracker_id"]
            for group_id, group_trackers in rider_groups.items():
                if track_id in group_trackers:
                    row["tracker_id"] = group_id
                    row["class_id"] = 81
                    row["class_name"] = "rider"
                    break
        new_rows.append(row)

    updated_detections = pd.DataFrame(new_rows)
    return updated_detections

def combine_detections(dataframe):
    """
    Combines detections with the same tracker ID within each frame.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing bounding box information.

    Returns:
        pd.DataFrame: New DataFrame with combined detections.
    """
    combined_detections = []

    for  frame_number, frame_group in dataframe.groupby("Frame"):
        # Group detections by tracker ID
        grouped_by_tracker = frame_group.groupby("tracker_id")

        for _, group in grouped_by_tracker:
            if len(group) > 1:
                # Compute the outer bounding box for the group
                x_min = group["x_min"].min()
                y_min = group["y_min"].min()
                x_max = group["x_max"].max()
                y_max = group["y_max"].max()

                # Combine detections with the same tracker ID
                combined_detection = group.iloc[0].copy()
                combined_detection["x_min"] = x_min
                combined_detection["y_min"] = y_min
                combined_detection["x_max"] = x_max
                combined_detection["y_max"] = y_max
                combined_detection["confidence"] = group["confidence"].max()
                combined_detections.append(combined_detection)
            else:
                # Keep single detection as is
                combined_detections.append(group.iloc[0])
                
    # Create a new DataFrame with combined detections
    combined_df = pd.DataFrame(combined_detections)
    return combined_df

def display_bounding_boxes(dataframe, video_file_path):
    """
    Generator function to display bounding boxes and labels for each frame.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing bounding box coordinates.
        video_file_path (str): Path to the video file.
        
    Yields:
    """
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: '{video_file_path}'.")
    
    # Dictionary for unique colors for each track ID
    track_colors = {}

    frame = None
    last_frame_number = -1
    for _, row in dataframe.iterrows():
        frame_number = row["Frame"]
        
        if last_frame_number != frame_number or last_frame_number==-1:
            if frame is not None:
                yield frame, last_frame_number
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            last_frame_number = frame_number

        if not ret:
            break

        x_min, y_min, x_max, y_max = int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])
        class_id = row["class_id"]
        class_name = row["class_name"]
        track_id = row["tracker_id"]

        # Assign a unique color to each track ID
        if track_id not in track_colors:
            track_colors[track_id] = np.random.randint(0, 256, size=3).tolist()

        # Draw bounding box
        color = tuple(track_colors[track_id])
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Add label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"c:{class_id} id:{track_id}", (x_min, y_min - 20), font, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"n:{class_name}", (x_min, y_min - 5), font, 0.6, color, 2, cv2.LINE_AA)
        
    cap.release()

def display_bounding_boxes_new(dataframe, video_file_path):
    """
    Generator function to display bounding boxes and labels for each frame.

    Args:
        dataframe (pd.DataFrame): Dataframe containing bounding box coordinates.
        video_file_path (str): Path to the video file

    Yields:
        frame (np.array: Frame with bounding boxes drwan.
        frame_number (int): Frame number.
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: '{video_file_path}'.")
    
    # Dictionary for unique colors for each track ID
    track_colors = {}

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Iterate over the dataframe rows
    for frame_number in range(total_frames):
        # Get all bounding boxes for the current frame
        frame_rows = dataframe[dataframe["Frame"] == frame_number]

        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw bounding boxes and labels
        for _, row in frame_rows.iterrows():
            x_min, y_min, x_max, y_max = int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])
            class_id = row["class_id"]
            class_name = row["class_name"]
            track_id = row["tracker_id"]

            # Assign a unque color to each track ID
            if track_id not in track_colors:
                np.random.seed(track_id) # Seed to ensure color consistency for track_id
                track_colors[track_id] = np.random.randint(0, 256, size=3).tolist()

            # Draw bounding box
            color = tuple(track_colors[track_id])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

            # Add label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"c:{class_id} id:{track_id}", (x_min, y_min - 20), font, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"n:{class_name}", (x_min, y_min - 5), font, 0.6, color, 2, cv2.LINE_AA)

        yield frame, frame_number

# def save_dataframe_to_csv(dataframe, video_path, name=None):
#     """
#     Saves the given dataframe as a CSV file in a specified folder given a video_path.

#     """

#     # Check if the folder to stroe data exists
#     DATA_PATH = Path(f"../Project_V2/data")
#     if not DATA_PATH.exists():
#         raise Exception(f"The path '{DATA_PATH.as_posix()}' is not recognized.")
#     video_name = Path(video_path).stem
#     print(video_name)
#     DATA_PATH = Path(DATA_PATH / video_name)
#     if not DATA_PATH.exists():
#         print(f"Creating data folder in {DATA_PATH.as_posix()}")
#         DATA_PATH.mkdir()
#     if name is None:
#         raise ValueError("Provide a name")
#     file_path = DATA_PATH.as_posix() + "/" + name
#     # Save the DataFrame as a CSV file to the specific folder

#     dataframe.to_csv(file_path, index=False)
#     print(f"Data frame successfully saved at '{DATA_PATH.as_posix()}'.")

def test():
    csvfile = "VID_20220426_161600\objects_detections_unprocessed.csv"
    videofile = "X:\Life\TFG\Coding\Testing\Videos\BikeBi\VID_20220426_161600.mp4"

    # Read CSV file
    df = load_csv(csvfile)

    # Obtain the unique IDs
    tracker_ID_df = unique_ID_dict(df)

    #df = assign_class_to_tracker(df)

    # Maintain only the track IDs that appear at least
    appearances = 3
    result_ids = get_ids_by_appearances(tracker_ID_df, appearances)
    filter_data = filter_data_by_ids(df, result_ids)

    # Combine persons with bikes and motorcycles
    combined_df = combine_person_and_bike(filter_data)

    # Obtain a dictionary of groups of id
    grouped_list = group_riders_trackers_id(combined_df)
    print(grouped_list)
    updated_detections = propagate_detection(combined_df)
    updated_detections = combine_detections(updated_detections)
    
    frame_iterator = iter(display_bounding_boxes(updated_detections, videofile))