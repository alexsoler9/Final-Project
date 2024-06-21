import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from pathlib import Path

from tqdm import tqdm

#####################################################
#       For Ultra Fast Lane Detector V2             # 
#####################################################

###############  Metric Calculation  ################

def process_coordinate_string(coord_str):
    # Initialize an empty list to store parsed coordianates
    coordinates = []

    # Check if coord_str is NaN
    if isinstance(coord_str, float) and np.isnan(coord_str):
        success = False
        return coordinates, success
    # Split the coordinates string by semicolon to get individual coordinate pairs
    coordinate_pairs = coord_str.split(";")

    # Process each coordinate pair
    for pair in coordinate_pairs:
        # Split each pair into x and y components
        x_str, y_str = pair.split(",")

        # Convert x and y components to integers
        x = int(x_str)
        y = int(y_str)

        # Append the coordinates tuple to the list
        coordinates.append((x,y))
    success = True 

    return coordinates, success

def compute_coordinate_variance(coords1, coords2):
    """
    Compute the variance (mean squared error) between two sets of coordinates.

    Parameters:
        coords1 (list of tuples): Coordinats from frame t-1.
        coords2 (list of tuples): Coordiantes from frame t.

    Returns:
        variance (float): The mean squared error between two sets of coordinates.
    """
    if not coords1 or not coords2:
        return None # Not enough points to compute variance
    
    coords1 = np.array(coords1) 
    coords2 = np.array(coords2)

    # Both sets must have the same length
    if len(coords1) != len(coords2):
        min_len = min(len(coords1), len(coords2))
        coords1 = coords1[:min_len]
        coords2 = coords2[:min_len]

    # Compute the mean squared error (variance)

    variance = mean_squared_error(coords1, coords2)

    return variance

def line_fit(coordinates, y_vals=[1300, 1000, 600], compute_r2=True):
    """
    Evaluates how well the point form a line

    Parameters:
        coordinates (list of tuples): List of (x,y) coordinate tuples.
        y_vals_to_predict (list of int): List of y_coordinates to predict x_coordinates
    
    Returns:
        r2 (float): Coefficient of determination indicating goodness of fit.
        predicted_coords (list of tuples): Predicted (x, y) coordinates for the given y-coordinates
    """
    if len(coordinates) < 2:
        return 0, np.nan # Not enough points to form a line
    
    # Separate the coordinates into X and y
    coordinates_array = np.array(coordinates, dtype=np.int32)
    X = coordinates_array[:,0].reshape(-1, 1)
    y = coordinates_array[:,1]

    # Perform linear regression
    model = LinearRegression().fit(X,y)

    if compute_r2:
        # Predict y values based on the model
        y_pred = model.predict(X)

        # Calculate the coefficient of determination (R^2)
        r2 = r2_score(y, y_pred)
    else:
        r2 = None

    # Perform linear regression (y -> X) for prediction
    min_value = np.min(y)
    y_vals_to_predict = y_vals.copy()
    #y_vals_to_predict.append(min_value)
    model_predict = LinearRegression().fit(y.reshape(-1, 1), X)
    x_vals_predicted = model_predict.predict(np.array(y_vals_to_predict).reshape(-1,1)).reshape(-1)

    predicted_coords = [(int(x), int(y)) for x, y in zip(x_vals_predicted, y_vals_to_predict)]

    return r2, predicted_coords

def mean_absolute_error(actual_coords, predicted_coords):
    """
    Compute the Mean Absolute Error (MAE) between actual and predicted coordinates.

    Parameters:
        actual_coords (list of tuples): List of actual (x, y) coordinates.
        predicted_coords (list of tuples): List of predicted (x, y) coordinates.

    Returns:
        float: MAE value.
    """
    actual_coords = np.array(actual_coords)
    predicted_coords = np.array(predicted_coords)
    mae = np.mean(np.abs(actual_coords - predicted_coords))
    return mae

def root_mean_squared_error(actual_coords, predicted_coords):
    """
    Compute the Root Mean Squared Error (RMSE) between actual and predicted coordinates.

    Parameters:
        actual_coords (list of tuples): List of actual (x, y) coordinates.
        predicted_coords (list of tuples): List of predicted (x, y) coordinates.

    Returns:
        float: RMSE value.
    """
    actual_coords = np.array(actual_coords)
    predicted_coords = np.array(predicted_coords)
    rmse = np.sqrt(np.mean((actual_coords - predicted_coords) ** 2))
    return rmse

def compute_lane_metrics_across_frames(dataframe):
    """
    Computes the metrics for lane coordinates between consecutive frames.

    Parameters:
        dataframe (pd.Dataframe): Dataframe containing lane coordinates for each frame

    Returns:
        r2_df (pd.DataFrame): DataFrame with computed r2 for each lane.
    """
    rows = []
    previous_predicted_coords = {}

    for frame_number in sorted(dataframe["Frame Number"].unique()):
        frame_data = dataframe[dataframe["Frame Number"] == frame_number]
        frame_metrics = {"Frame Number": frame_number}

        for lane in range(0, 8): # Adjust the range based on the number of lanes
            column_name = f"Lane_{lane}_Coordinates"
            if column_name in frame_data.columns:
                coord_str = frame_data[column_name].values[0]
                current_coords, success = process_coordinate_string(coord_str)
            
                if success:
                    r2, predicted_coords = line_fit(current_coords)
                    
                    
                    frame_metrics[f"Lane_{lane}_Coords"] = predicted_coords
                    #frame_variance[f"Lane_{lane}_Variance"] = variance
                    frame_metrics[f"Lane_{lane}_R^2"] = r2
                    
                    previous_predicted_coords[lane] = predicted_coords

        rows.append(frame_metrics)

    updated_dataframe = pd.DataFrame(rows)
    return updated_dataframe

############  Determine Lane Use  #############

def compute_moving_average_r2(dataframe, window_size):
    """
    Compute the moving average of R^2 values for each lane in each frame.

    """
    lanes = [col for col in dataframe.columns if "R^2" in col]
    avg_r2_df = dataframe.copy()

    for lane in lanes:
        rolling_window_size = 2 * window_size + 1
        original_values = avg_r2_df[lane]
        # Mask zero values
        masked_values = original_values.mask(original_values == 0)
        # Compute rolling mean ensuring all values in the window are valid
        rolling_mean = masked_values.rolling(window=rolling_window_size, center=True, min_periods=rolling_window_size).mean()
        avg_r2_df[f"Avg_{lane}"] = rolling_mean
    
    return avg_r2_df

def determine_lane_use(dataframe, window_size, threshold):
    """
    Determine if a lane should be used based on the moving average R^2 values.
    """
    avg_r2_df = compute_moving_average_r2(dataframe, window_size)
    lane_use_df = dataframe[["Frame Number"]].copy()

    lanes = [col for col in dataframe.columns if "R^2" in col]
    for lane in lanes:
        avg_col = f"Avg_{lane}"
        lane_use_df[f"Use_{lane.split('_')[1]}"] = avg_r2_df[avg_col] > threshold

    return lane_use_df, avg_r2_df

############  Determine Ego Frames  #############
def find_ego_frames(lane_use_df, avg_r2_df, only_ego=True):
    """
    Determine frames where both ego_left and ego_right are present and compute the width of the ego lane.

    Parameters:
        lane_use_df (pd.DataFrame): DataFrame indicating lane usage based on R^2 values.
        avg_r2_df (pd.DataFrame): Dataframe with the moving average R^2 values.
        ony_ego (bool): [To Be Implemented] Flag to indicate wheter to check only ego lanes. Default: True.

    
    """
    ego_frames = []
    all_distances = []

    for frame_number in lane_use_df["Frame Number"]:
        ego_left = lane_use_df.loc[lane_use_df["Frame Number"] == frame_number, ["Use_1", "Use_5"]].any(axis=1).values[0]
        ego_right = lane_use_df.loc[lane_use_df["Frame Number"] == frame_number, ["Use_2", "Use_6"]].any(axis=1).values[0]

        if ego_left and ego_right:
            #ego_frames.append(frame_number)
            # Find the lane with the highest R^2 for ego left and ego right
            r2_left = avg_r2_df.loc[avg_r2_df["Frame Number"] == frame_number, ["Lane_1_R^2", "Lane_5_R^2"]]
            r2_right = avg_r2_df.loc[avg_r2_df["Frame Number"] == frame_number, ["Lane_2_R^2", "Lane_6_R^2"]]

            best_left_lane  = r2_left.idxmax(axis=1).values[0]
            best_right_lane = r2_right.idxmax(axis=1).values[0]
            
            best_left_lane_coords = avg_r2_df.loc[avg_r2_df["Frame Number"]==frame_number, f"{best_left_lane.split('_')[0]}_{best_left_lane.split('_')[1]}_Coords"].values[0]
            best_right_lane_coords = avg_r2_df.loc[avg_r2_df["Frame Number"]==frame_number, f"{best_right_lane.split('_')[0]}_{best_right_lane.split('_')[1]}_Coords"].values[0]


            if len(best_left_lane_coords) == len(best_right_lane_coords):
                distances = [np.linalg.norm(np.array(c1) - np.array(c2)) for c1, c2 in zip(best_left_lane_coords[:-1], best_right_lane_coords[:-1])]
                all_distances.append(distances)
                
                ego_frames.append({
                    "Frame Number": frame_number,
                    "Ego Lane Width": distances,
                    "Ego Left Lane": best_left_lane_coords,
                    "Ego Right Lane": best_right_lane_coords,
                })
        else:
            ego_frames.append({
                    "Frame Number": frame_number,
                    "Ego Lane Width": np.nan,
                    "Ego Left Lane": np.nan,
                    "Ego Right Lane": np.nan,
                })
    avg_width = np.mean(all_distances, axis=0) if all_distances else 0
    deviation = np.std(all_distances, axis=0) if all_distances else 0

    # Plotting
    if all_distances :
        num_y_coords = len(all_distances[0])  # Number of y-coordinates
        fig, axes = plt.subplots(num_y_coords, 1, figsize=(10, 2 * num_y_coords), sharex=True)

        if num_y_coords == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            widths_at_y = [dist[i] for dist in all_distances]
            ax.plot(range(len(widths_at_y)), widths_at_y, label=f'Width at Y-Coord {i}')
            ax.axhline(avg_width[i], color='green', linestyle='-', linewidth=2, label='Average Width')
            ax.axhline(avg_width[i] + deviation[i], color='green', linestyle='--', linewidth=1, label='Average + Std Dev')
            ax.axhline(avg_width[i] - deviation[i], color='green', linestyle='--', linewidth=1, label='Average - Std Dev')
            ax.set_ylabel('Width (pixels)')
            ax.legend()

        plt.xlabel('Frame Number')
        plt.suptitle('Ego Lane Width Across Frames at Different Y Coordinates')
        plt.show()

    return ego_frames, avg_width, deviation

def find_ego_frames_updated(lane_use_df, avg_r2_df, only_ego=True, y_min=0, y_max=1080):
    """
    Determine frames where both ego_left and ego_right are present and compute the width of the ego lane.

    Parameters:
        lane_use_df (pd.DataFrame): DataFrame indicating lane usage based on R^2 values.
        avg_r2_df (pd.DataFrame): DataFrame with the moving average R^2 values.
        only_ego (bool): [To Be Implemented] Flag to indicate whether to check only ego lanes. Default: True.
    """
    ego_frames = []
    all_distances = []
    all_frame_numbers = lane_use_df["Frame Number"].unique()
    distances_dict = {frame: [np.nan] * 10 for frame in all_frame_numbers}  # Assuming up to 10 y-coordinates

    for frame_number in lane_use_df["Frame Number"]:
        ego_left = lane_use_df.loc[lane_use_df["Frame Number"] == frame_number, ["Use_1", "Use_5"]].any(axis=1).values[0]
        ego_right = lane_use_df.loc[lane_use_df["Frame Number"] == frame_number, ["Use_2", "Use_6"]].any(axis=1).values[0]

        if ego_left and ego_right:
            # Find the lane with the highest R^2 for ego left and ego right
            r2_left = avg_r2_df.loc[avg_r2_df["Frame Number"] == frame_number, ["Lane_1_R^2", "Lane_5_R^2"]]
            r2_right = avg_r2_df.loc[avg_r2_df["Frame Number"] == frame_number, ["Lane_2_R^2", "Lane_6_R^2"]]

            best_left_lane = r2_left.idxmax(axis=1).values[0]
            best_right_lane = r2_right.idxmax(axis=1).values[0]
            
            best_left_lane_coords = avg_r2_df.loc[avg_r2_df["Frame Number"] == frame_number, f"{best_left_lane.split('_')[0]}_{best_left_lane.split('_')[1]}_Coords"].values[0]
            best_right_lane_coords = avg_r2_df.loc[avg_r2_df["Frame Number"] == frame_number, f"{best_right_lane.split('_')[0]}_{best_right_lane.split('_')[1]}_Coords"].values[0]

            if len(best_left_lane_coords) == len(best_right_lane_coords):
                distances = [np.linalg.norm(np.array(c1) - np.array(c2)) for c1, c2 in zip(best_left_lane_coords[:-1], best_right_lane_coords[:-1])]
                all_distances.append(distances)
                distances_dict[frame_number] = distances
                
                ego_frames.append({
                    "Frame Number": frame_number,
                    "Ego Lane Width": distances,
                    "Ego Left Lane": best_left_lane_coords,
                    "Ego Right Lane": best_right_lane_coords,
                })
        else:
            ego_frames.append({
                "Frame Number": frame_number,
                "Ego Lane Width": np.nan,
                "Ego Left Lane": np.nan,
                "Ego Right Lane": np.nan,
            })

    avg_width = np.mean(all_distances, axis=0) if all_distances else 0
    deviation = np.std(all_distances, axis=0) if all_distances else 0

    # Plotting
    if all_distances:
        num_y_coords = len(all_distances[0])  # Number of y-coordinates
        fig, axes = plt.subplots(num_y_coords, 1, figsize=(10, 2 * num_y_coords), sharex=True)

        if num_y_coords == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            widths_at_y = [distances_dict[frame][i] for frame in all_frame_numbers]
            ax.scatter(all_frame_numbers, widths_at_y, label=f'Width at Y-Coord {i}')
            ax.axhline(avg_width[i], color='green', linestyle='-', linewidth=2, label='Average Width')
            ax.axhline(avg_width[i] + deviation[i], color='green', linestyle='--', linewidth=1, label='Average + Std Dev')
            ax.axhline(avg_width[i] - deviation[i], color='green', linestyle='--', linewidth=1, label='Average - Std Dev')
            ax.set_ylabel('Width (pixels)')
            #ax.set_ylim(y_min, y_max)
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        plt.xlabel('Frame Number')
        plt.xlim(all_frame_numbers.min()), all_frame_numbers.max()
        plt.suptitle('Ego Lane Width Across Frames at Different Y Coordinates')
        plt.show()

    return ego_frames, avg_width, deviation

############  Determine Valid Lane  #############

def gaussian_filter(width, mean, std_dev, n_std_l=0, n_std_u=2):
    """ 
    Determine if the width falls within n standard deviations from the mean.

    """
    lower_bound = mean - n_std_l * std_dev
    upper_bound = mean + n_std_u * std_dev
    return np.all((width >= lower_bound) & (width <= upper_bound))

def determine_valid_lanes(ego_frames, avg_width, deviation, n_std_l=0, n_std_u=2):
    """ 
    Determine if a lane should be considered based on the width information.
    """
    valid_frames = []

    for frame in ego_frames:
        frame_number = frame["Frame Number"]
        ego_lane_width = np.array(frame["Ego Lane Width"])

        if gaussian_filter(ego_lane_width, avg_width, deviation, n_std_l, n_std_u):
            valid_frames.append(frame_number)

    return valid_frames

############  Smooth coordinates across frames  #############
# NOTE NOT USED
def expand_coordinates(dataframe, lane_col):
    """
    Expand the coordinate tuples into seprate columns for x and y.
    """
    coord_data = dataframe[lane_col].apply(lambda coords: coords if isinstance(coords, list) else [(np.nan, np.nan)]*3) # Where N is the number of points that form the lane
    expanded_coords = pd.DataFrame(coord_data.tolist(), columns=[f"{lane_col}_{i}" for i in range(len(coord_data.iloc[0]))])

    for i in range(len(expanded_coords.columns)):
        expanded_coords[[f"{lane_col}_x_{i}", f"{lane_col}_y_{i}"]] = expanded_coords[f"{lane_col}_{i}"].apply(pd.Series)
        expanded_coords.drop(columns=[f"{lane_col}_{i}"], inplace=True)

    return expanded_coords

def reconstruct_coordinates(expanded_df, lane_col):
    """
    Reconstruct the coordinate tuples from separate x and y columns.
    """
    coords = expanded_df[[col for col in expanded_df.columns if col.startswith(lane_col)]].values
    reconstructed_coords = [list(zip(coords[i][0::2], coords[i][1::2])) for i in range(coords.shape[0])]
    return reconstructed_coords

def apply_rolling_mean(dataframe, window_size):

    expanded_left = expand_coordinates(dataframe, "Ego Left Lane")
    expanded_right = expand_coordinates(dataframe, "Ego Right Lane")

    # Apply rolling mean
    smoothed_left = expanded_left.rolling(window=2*window_size+1, center=True, min_periods=int(window_size*1.5)).mean()
    smoothed_right = expanded_right.rolling(window=2*window_size+1, center=True, min_periods=int(window_size*1.5)).mean()

    # Reconstruct coordinates
    dataframe["Ego Left Lane Smoothed"] = reconstruct_coordinates(smoothed_left, "Ego Left Lane")
    dataframe["Ego Right Lane Smoothed"] = reconstruct_coordinates(smoothed_right, "Ego Right Lane") 
    
    return dataframe

############  Smooth coordinates across frames  ############# 

def plot_ego_lane(frame, left_coords, right_coords, color_left=(0, 255, 0), color_right=(0, 0, 255), thickness=2):
    
    left_coords = np.array([left_coords], dtype=np.int32)
    right_coords = np.array([right_coords], dtype=np.int32)
    frame = cv2.polylines(frame, left_coords, isClosed=False, color=color_left, thickness=thickness)
    frame = cv2.polylines(frame, right_coords, isClosed=False, color=color_right, thickness=thickness)

    return frame    

def view_valid_frames(video_path, valid_frames, ego_frames):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    frame_number=0
    ego_frames = pd.DataFrame(ego_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number in valid_frames:
            # Extract the coordinates for the current frame
            ego_row = ego_frames[ego_frames["Frame Number"] == frame_number]
            left_lane = ego_row["Ego Left Lane"].values[0]
            right_lane = ego_row["Ego Right Lane"].values[0]

            frame = plot_ego_lane(frame, left_lane, right_lane)

            yield frame_number, frame
        
        frame_number += 1
    
    cap.release()

def is_valid_coords(coords):
    return all(not any(np.isnan(coord)) for coord in coords)

def view_smoothed_frames(video_path, dataframe):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    frame_number=0
    ego_frames = dataframe
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if True:
            # Extract the coordinates for the current frame
            ego_row = ego_frames[ego_frames["Frame Number"] == frame_number]
            left_lane = ego_row["Ego Left Lane Smoothed"].values[0]
            right_lane = ego_row["Ego Right Lane Smoothed"].values[0]
            if is_valid_coords(left_lane) and is_valid_coords(right_lane):# check if left_lane and right_lane contain NaN
                frame = plot_ego_lane(frame, left_lane, right_lane)

        yield frame_number, frame
        
        frame_number += 1
    
    cap.release()

def plot_valid_frames(valid_frames, dataframe):
    # Get the total number of frames from the DataFrame
    total_frames = dataframe["Frame Number"].max() + 1
    print(f"\tNumber of Valid Frames: {len(valid_frames)}")
    print(f"\tNumber of Frames:{total_frames}")
    # Create a list of all frame numbers
    all_frames = list(range(total_frames))

    # Create a boolean list indicating wheter each frame is valid
    is_valid = [frame in valid_frames for frame in all_frames]

    plt.figure(figsize=(15,5))

    plt.scatter(all_frames, is_valid, linestyle='-', marker='o', color='b', label='Valid Frames')
    plt.xlabel("Frame Number")
    plt.ylabel("Is Valid")
    plt.title('Valid Frames Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    


########### Propagate valid lanes #############

def set_invalid_rows_to_nan(dataframe, valid_frames):
    dataframe_copy = dataframe.copy()
    for index, row in dataframe_copy.iterrows():
        if row["Frame Number"] not in valid_frames:
            for col in dataframe_copy.columns:
                if col != "Frame Number":
                    dataframe_copy.at[index, col] = np.nan
    
    return dataframe_copy

def fit_points_to_line(points):
    if len(points) < 2:
        return points # Not enough points to fit a line
    points = np.array(points)
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    fitted_points = list(zip(X.flatten(), y_pred))
    return fitted_points

def interpolate_points(p1, p2, num_points):
    points = []
    for i in range(num_points):
        alpha = i / (num_points - 1)
        x = (1 - alpha) * p1[0] + alpha*p2[0]
        y = (1 - alpha) * p1[1] + alpha*p2[1]
        points.append((x,y))
    return points

def propagate_lanes_with_optical_flow(dataframe, valid_frames, valid_ranges, window_size, video_path):
    dataframe = set_invalid_rows_to_nan(dataframe, valid_frames)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot opon video file.")
        return
    print(len(valid_ranges))
    progress_bar = tqdm(total=len(valid_ranges))

    for start, end in valid_ranges:
        range_df = dataframe[(dataframe["Frame Number"] >= start) & (dataframe["Frame Number"] <= end)]
        valid_range_frames = range_df["Frame Number"].tolist()

        prev_frame_number = None
        prev_left_lane = None
        prev_right_lane = None

        last_frame = dataframe["Frame Number"].max()
        progress_bar.update()
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start-1)
        for frame_idx in valid_range_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            #print(frame_idx)
            ego_row = range_df[range_df["Frame Number"] == frame_idx]
            left_lane = ego_row['Ego Left Lane'].values[0]
            right_lane = ego_row['Ego Right Lane'].values[0]

            if frame_idx == 1177:
                print()

            if isinstance(right_lane, list) and isinstance(left_lane, list):
                left_lane = interpolate_points(left_lane[0], left_lane[-1], 50)
                right_lane = interpolate_points(right_lane[0], right_lane[-1], 50)
            if frame_idx == last_frame:
                break
            
            if frame_idx in range(start, end+1): #prev_frame_number is not None and frame_idx == prev_frame_number + 1:
                if frame_idx not in valid_frames:
                    # Use optical flow to predict the next lane points
                    #left_lane = np.array(left_lane, dtype=np.float32)
                    #right_lane = np.array(right_lane, dtype=np.float32)
                    
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    left_lane_flow, st_l, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_left_lane, None, winSize=(window_size, window_size), maxLevel=9)
                    right_lane_flow, st_r, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_right_lane, None, winSize=(window_size, window_size), maxLevel=9)
                    
                    if left_lane_flow  is not None and st_l is not None and right_lane_flow  is not None and st_r is not None:
                        left_lane = left_lane_flow[st_l == 1].reshape(-1, 2)
                        right_lane = right_lane_flow[st_r == 1].reshape(-1, 2)
                    
                        # Fit the points to a line
                        r_2_left, left_lane_fit = line_fit(left_lane)
                        r_2_right, right_lane_fit = line_fit(right_lane)
                    
                    dataframe.at[frame_idx, 'Ego Left Lane'] = left_lane_fit
                    dataframe.at[frame_idx, 'Ego Right Lane'] = right_lane_fit
            if len(left_lane) > 1 and len(right_lane):
                prev_left_lane = np.array(left_lane, dtype=np.float32).reshape(-1, 1, 2)
                prev_right_lane = np.array(right_lane, dtype=np.float32).reshape(-1, 1, 2)
                if len(prev_left_lane) < 2 or len(prev_right_lane) < 2:
                    break
            prev_frame = frame.copy()

        

    cap.release()
    return dataframe

def propagate_lanes_with_optical_flow_updated(dataframe, valid_frames, window_size, video_path, n_frames):
    dataframe = set_invalid_rows_to_nan(dataframe, valid_frames)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    total_valid_frames = len(valid_frames)
    progress_bar = tqdm(total=total_valid_frames)
    
    last_frame = dataframe["Frame Number"].max()
    
    for valid_frame_idx, start in enumerate(valid_frames):
        progress_bar.update()
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start-1)
        ret, prev_frame = cap.read()
        if not ret:
            continue
        
        ego_row = dataframe[dataframe["Frame Number"] == start]
        left_lane = ego_row['Ego Left Lane'].values[0]
        right_lane = ego_row['Ego Right Lane'].values[0]
        
        if isinstance(left_lane, list) and isinstance(right_lane, list):
            left_lane = interpolate_points(left_lane[0], left_lane[-1], 50)
            right_lane = interpolate_points(right_lane[0], right_lane[-1], 50)
        
        prev_left_lane = np.array(left_lane, dtype=np.float32).reshape(-1, 1, 2)
        prev_right_lane = np.array(right_lane, dtype=np.float32).reshape(-1, 1, 2)
        
        for offset in range(1, n_frames + 1):
            frame_idx = start + offset
            if frame_idx > last_frame:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            left_lane_flow, st_l, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_left_lane, None, winSize=(window_size, window_size), maxLevel=9)
            right_lane_flow, st_r, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_right_lane, None, winSize=(window_size, window_size), maxLevel=9)
            
            if left_lane_flow is not None and st_l is not None and right_lane_flow is not None and st_r is not None:
                left_lane = left_lane_flow[st_l == 1].reshape(-1, 2)
                right_lane = right_lane_flow[st_r == 1].reshape(-1, 2)
                
                # Fit the points to a line
                r_2_left, left_lane_fit = line_fit(left_lane)
                r_2_right, right_lane_fit = line_fit(right_lane)
                
                dataframe.at[frame_idx, 'Ego Left Lane'] = left_lane_fit
                dataframe.at[frame_idx, 'Ego Right Lane'] = right_lane_fit
                
                prev_left_lane = np.array(left_lane, dtype=np.float32).reshape(-1, 1, 2)
                prev_right_lane = np.array(right_lane, dtype=np.float32).reshape(-1, 1, 2)
            
            prev_frame = frame.copy()
    
    cap.release()
    progress_bar.close()
    return dataframe

def smooth_lane_coordinates(dataframe, columns_to_smooth=['Ego Left Lane', 'Ego Right Lane'], window_size=2, y_vals=[1920, 600]):
    dataframe_copy = dataframe.copy()

    for column in columns_to_smooth:
        if column in dataframe_copy.columns:
            lane = expand_coordinates(dataframe, column)
            smoothed_lane = lane.rolling(window=2*window_size+1, center=True, min_periods=1).mean()
            
            smoothed_lane = smoothed_lane.apply(
                lambda row: [(x, y) for x, y in zip(row[::2], row[1::2]) if not np.isnan(x) and not np.isnan(y)], 
                axis=1
            )

            fitted_lanes = smoothed_lane.apply(
                lambda coords: line_fit(coords, y_vals=y_vals, compute_r2=False)[1]
            )

            dataframe_copy[column] = fitted_lanes

    return dataframe_copy


# Final Visualization

def view_frames(video_path, dataframe):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    frame_number=0
    ego_frames = dataframe
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if True:
            # Extract the coordinates for the current frame
            ego_row = ego_frames[ego_frames["Frame Number"] == frame_number]
            left_lane = ego_row["Ego Left Lane"].values[0]
            right_lane = ego_row["Ego Right Lane"].values[0]
            if isinstance(right_lane, list) and isinstance(left_lane, list):
                if is_valid_coords(left_lane) and is_valid_coords(right_lane):
                    frame = plot_ego_lane(frame, left_lane, right_lane)
                    
        yield frame_number, frame
        
        frame_number += 1
    
    cap.release()

def create_video_from_frames(folder_name, output_string, frame_iterator):
    # Check if the folder exists
    DATA_PATH = Path("../../Project_V2/data/"+folder_name)
    if not DATA_PATH.exists():
        raise Exception(f"The path '{DATA_PATH.as_posix()}' is not recognized.")
    
    # Set frame size (width, height)
    frame_size = (1080, 1920)

    # Create a VideoWriter
    output_filename = f"{folder_name}_{output_string}.mp4"
    out = cv2.VideoWriter(DATA_PATH.as_posix()+"/"+output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, frame_size)

    for frame_number, frame in frame_iterator:
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(frame, f"Frame Number: {frame_number}", (50, 50 + 2*20), font, 1.5, (0, 255, 0), 2)
        out.write(frame)
    
    out.release()
    print(f"Video saved as '{output_filename}' at '{DATA_PATH.as_posix()}'")

def view_video_from_iterator( frame_iterator):

    for frame_number, frame in frame_iterator:
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(frame, f"Frame Number: {frame_number}", (50, 50 + 2*20), font, 1.5, (0, 255, 0), 2)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation= cv2.INTER_LINEAR)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(15)
        if key == ord("q"):
            break

    
    

############  Example  #############

def test():
    # TODO: Store in a file path to the video, either when lane_detector is used or when object detector is used.
    # Load initial csv 
    csv_file_path = "X:\Life\TFG\Coding\Project_v2\data\VID_20220426_161600\lane_detection_unprocessed.csv"
    videofile = "X:\Life\TFG\Coding\Testing\Videos\BikeBi\VID_20220426_161600.mp4"
    df = pd.read_csv(csv_file_path)

    # Uses the information from the original dataframe to compute: predicted coordiantes at certain y_values, compute R^2, and variance
    variance_df = compute_lane_metrics_across_frames(df)

    # Determine the lane use
    window_size = 5
    threshold = 0.9
    lane_use_df, avg_r2_df = determine_lane_use(variance_df, window_size, threshold)

    # Returns a list of dictionaries containining, it also returns the average mean across all rows and the standard deviation across all rows.
    ego_frames, avg_width, deviation = find_ego_frames(lane_use_df, avg_r2_df)

    # Determines if a lane is valid by using the standard deviation.
    valid_frames = determine_valid_lanes(ego_frames, avg_width, deviation, n_std=1)

    dataframe = pd.DataFrame(ego_frames)
    dataframe = apply_rolling_mean(dataframe, window_size=5)

    ego_lane_generator = iter(view_smoothed_frames(video_path=videofile, dataframe=dataframe))

def compute_lane(csv_file_path = "X:\Life\TFG\Coding\Project_v2\data\VID_20220426_161600\lane_detection_unprocessed.csv"):
    df = pd.read_csv(csv_file_path)

    # Uses the information from the original dataframe to compute: predicted coordiantes at certain y_values, compute R^2, and variance
    variance_df = compute_lane_metrics_across_frames(df)

    # Determine the lane use
    window_size = 5
    threshold = 0.9
    lane_use_df, avg_r2_df = determine_lane_use(variance_df, window_size, threshold)

    # Returns a list of dictionaries containining, it also returns the average mean across all rows and the standard deviation across all rows.
    ego_frames, avg_width, deviation = find_ego_frames(lane_use_df, avg_r2_df)
    return ego_frames, avg_width, deviation





def determine_valid_data(csv_file_path = "X:\Life\TFG\Coding\Project_v2\data\VID_20220426_161600\lane_detection_unprocessed.csv",lane_thr=(5, 0.8), width_std_l=0, width_std_h=1, optical_window=60):
    df = pd.read_csv(csv_file_path)

    # Uses the information from the original dataframe to compute: predicted coordiantes at certain y_values, compute R^2, and variance
    metric_df = compute_lane_metrics_across_frames(df)

    # Determine the lane use
    window_size, threshold = lane_thr
    
    lane_use_df, avg_r2_df = determine_lane_use(metric_df, window_size, threshold)

    # Returns a list of dictionaries containining, it also returns the average mean across all rows and the standard deviation across all rows.
    ego_frames, avg_width, deviation = find_ego_frames_updated(lane_use_df, avg_r2_df)
    print(f"Average width: {avg_width}")
    # Determines if a lane is valid by using the standard deviation.
    valid_frames = determine_valid_lanes(ego_frames, avg_width, deviation, n_std_l=width_std_l, n_std_u=width_std_h)


    dataframe = pd.DataFrame(ego_frames)
    valid_ranges = get_valid_ranges(valid_frames, dataframe, window_size=optical_window)

    return  valid_frames, valid_ranges, dataframe



def get_valid_ranges(valid_frames, dataframe, window_size=60):
    # Get the total number of frames from the DataFrame
    total_frames = dataframe["Frame Number"].max() + 1
    print(f"\tNumber of Valid Frames: {len(valid_frames)}")
    print(f"\tNumber of Frames:{total_frames}")
    # Create a list of all frame numbers
    all_frames = list(range(total_frames))

    # Create a boolean list indicating wheter each frame is valid
    is_valid = [frame in valid_frames for frame in all_frames]

    # Group the valid frames into ranges based on window_size
    valid_ranges = []
    current_range = []

    for i, frame in enumerate(valid_frames):
        if not current_range:
            current_range.append(frame)
        elif frame- valid_frames[i-1] <= window_size:
            current_range.append(frame)
        else:
            valid_ranges.append((current_range[0], current_range[-1]))
            current_range = [frame]

    if current_range:
        valid_ranges.append((current_range[0], current_range[-1]))

    plt.figure(figsize=(15,5))
    plt.scatter(all_frames, is_valid, linestyle='-', marker='o', color='b', label='Valid Frames')
    for start, end in valid_ranges:
        plt.plot(range(start, end+1), [True]*(end-start+1), color="g", linewidth=10, label="Propagated Region")
    plt.xlabel("Frame Number")
    plt.ylabel("Is Valid")
    plt.title('Valid Frames Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    return valid_ranges 

# def save_dataframe_to_csv(dataframe, video_path, name):
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

#     file_path = DATA_PATH.as_posix() + "/" + "lane_detection_processed.csv"
#     # Save the DataFrame as a CSV file to the specific folder

#     dataframe.to_csv(file_path, index=False)
#     print(f"Data frame successfully saved at '{DATA_PATH.as_posix()}'.")


#########################################
#               TEST                    #
#########################################

def refine_lane_coordinates(csv_file, threshold=20):
    import pandas as pd
    import numpy as np
    from pykalman import KalmanFilter
    import cv2
    from scipy.interpolate import interp1d

    # Load and parse the CSV file
    df = pd.read_csv(csv_file)

    def parse_coordinates(coord_str):
        if pd.isna(coord_str):
            return []
        return [tuple(map(float, point.split(','))) for point in coord_str.split(';')]

    frame_numbers = df['Frame Number'].values
    lane_coordinates = [[parse_coordinates(row[i]) for i in range(1, df.shape[1])] for row in df.values]

    # Smoothing the coordinates
    def kalman_smoothing(points):
        if len(points) < 2:
            return points
        kf = KalmanFilter(initial_state_mean=points[0], n_dim_obs=2)
        smoothed_data, _ = kf.smooth(points)
        return smoothed_data

    def smooth_coordinates(lane_coords):
        smoothed_coords = []
        for lane in lane_coords:
            smoothed_lane = []
            for points in lane:
                smoothed_lane.append(kalman_smoothing(points))
            smoothed_coords.append(smoothed_lane)
        return smoothed_coords

    smoothed_coordinates = smooth_coordinates(lane_coordinates)

    # Lane consistency check
    def consistency_check(lane_coords, threshold):
        def check_points(points):
            for i in range(1, len(points)):
                if np.linalg.norm(np.array(points[i]) - np.array(points[i-1])) > threshold:
                    points[i] = points[i-1]
            return points

        checked_coords = []
        for lane in lane_coords:
            checked_lane = []
            for points in lane:
                checked_lane.append(check_points(points))
            checked_coords.append(checked_lane)
        return checked_coords

    checked_coordinates = consistency_check(smoothed_coordinates, threshold)

    # Interpolation and extrapolation
    def interpolate_coordinates(lane_coords):
        def interpolate_points(points):
            if len(points) < 2:
                return points
            x = np.arange(len(points))
            points = np.array(points)
            valid_points = ~np.isnan(points).any(axis=1)
            if valid_points.sum() < 2:
                return points
            interp_func = interp1d(x[valid_points], points[valid_points], axis=0, kind='linear', fill_value='extrapolate')
            return interp_func(x)

        interpolated_coords = []
        for lane in lane_coords:
            interpolated_lane = []
            for points in lane:
                interpolated_lane.append(interpolate_points(points))
            interpolated_coords.append(interpolated_lane)
        return interpolated_coords

    final_coordinates = interpolate_coordinates(checked_coordinates)

    return frame_numbers, final_coordinates

def line_fit_with_dist(coordinates, y_vals=[1300, 1000, 600], compute_r2=True):
    if len(coordinates) < 2:
        return 0, np.nan
    
    coordinates_array = np.array(coordinates, dtype=np.int32)
    X = coordinates_array[:, 0].reshape(-1, 1)
    y = coordinates_array[:, 1]

    y_min,y_max = y.min(), y.max()
    distance = np.linalg.norm(y_max-y_min)
    if distance < 100:
        return 0, np.nan

    model = LinearRegression().fit(X, y)

    if compute_r2:
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
    else:
        r2 = None

    min_value = np.min(y)
    y_vals_to_predict = y_vals.copy()
    model_predict = LinearRegression().fit(y.reshape(-1, 1), X)
    x_vals_predicted = model_predict.predict(np.array(y_vals_to_predict).reshape(-1, 1)).reshape(-1)

    predicted_coords = [(int(x), int(y)) for x, y in zip(x_vals_predicted, y_vals_to_predict)]

    return r2, predicted_coords

def combine_lanes(refined_coords):
    combined_coords = []
    for frame_coords in refined_coords:
        # Initialize placeholders for empty arrays
        empty_array = np.zeros_like(frame_coords[0])  # Replace with the desired shape

        combined_frame_coords = [
            frame_coords[0],  # Combine Lane 1 and Lane 5
            np.concatenate((frame_coords[1], frame_coords[5])) if (len(frame_coords[1]) > 0 and len(frame_coords[5]) > 0) else frame_coords[1],
            np.concatenate((frame_coords[2], frame_coords[6])) if (len(frame_coords[2]) > 0 and len(frame_coords[6]) > 0) else frame_coords[2],
            frame_coords[3]
        ]
        combined_coords.append(combined_frame_coords)
    return combined_coords

def find_ego_frames_kalman(frame_numbers, coordinates):
    avg_r2_df = pd.DataFrame()
    all_r2 = []
    all_pred_coords = []
    all_frame_numbers = frame_numbers
    distances_dict = {frame: [np.nan] * 10 for frame in all_frame_numbers}  # Assuming up to 10 y-coordinates
    
    for i, frame_number in enumerate(frame_numbers):
        frame_r2 = {"Frame Number": frame_number}
        pred_coords = {"Frame Number": frame_number}
        
        for j in range(4):
            coords = coordinates[i][j]

            r2, pred = line_fit_with_dist(coords)
            
            frame_r2[f"Lane_{j}_R^2"] = r2
            pred_coords[f"Lane_{j}_Coords"] = pred
        
        all_r2.append(frame_r2)
        all_pred_coords.append(pred_coords)
    
    avg_r2_df = pd.DataFrame(all_r2)
    pred_coords_df = pd.DataFrame(all_pred_coords)
    
    ego_frames = []
    all_distances = []

    for i, frame_number in enumerate(frame_numbers):
        ego_left = not pred_coords_df.loc[pred_coords_df["Frame Number"] == frame_number, "Lane_1_Coords"].isna().values[0]
        ego_right = not pred_coords_df.loc[pred_coords_df["Frame Number"] == frame_number, "Lane_2_Coords"].isna().values[0]

        if ego_left and ego_right:
            r2_left = avg_r2_df.loc[avg_r2_df["Frame Number"] == frame_number, ["Lane_1_R^2"]]
            r2_right = avg_r2_df.loc[avg_r2_df["Frame Number"] == frame_number, ["Lane_2_R^2"]]

            best_left_lane = r2_left.idxmax(axis=1).values[0]
            best_right_lane = r2_right.idxmax(axis=1).values[0]
            
            best_left_lane_coords = pred_coords_df.loc[pred_coords_df["Frame Number"] == frame_number, f"{best_left_lane.split('_')[0]}_{best_left_lane.split('_')[1]}_Coords"].values[0]
            best_right_lane_coords = pred_coords_df.loc[pred_coords_df["Frame Number"] == frame_number, f"{best_right_lane.split('_')[0]}_{best_right_lane.split('_')[1]}_Coords"].values[0]

            if len(best_left_lane_coords) == len(best_right_lane_coords):
                distances = [np.linalg.norm(np.array(c1) - np.array(c2)) for c1, c2 in zip(best_left_lane_coords[:-1], best_right_lane_coords[:-1])]
                all_distances.append(distances)
                distances_dict[frame_number] = distances
                
                ego_frames.append({
                    "Frame Number": frame_number,
                    "Ego Lane Width": distances,
                    "Ego Left Lane": best_left_lane_coords,
                    "Ego Right Lane": best_right_lane_coords,
                })
        else:
            ego_frames.append({
                "Frame Number": frame_number,
                "Ego Lane Width": np.nan,
                "Ego Left Lane": np.nan,
                "Ego Right Lane": np.nan,
            })

    avg_width = np.mean(all_distances, axis=0) if all_distances else 0
    deviation = np.std(all_distances, axis=0) if all_distances else 0
    # Plotting
    if all_distances:
        num_y_coords = len(all_distances[0])  # Number of y-coordinates
        fig, axes = plt.subplots(num_y_coords, 1, figsize=(10, 2 * num_y_coords), sharex=True)

        if num_y_coords == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            widths_at_y = [distances_dict[frame][i] for frame in all_frame_numbers]
            ax.scatter(all_frame_numbers, widths_at_y, label=f'Width at Y-Coord {i}')
            ax.axhline(avg_width[i], color='green', linestyle='-', linewidth=2, label='Average Width')
            ax.axhline(avg_width[i] + deviation[i], color='green', linestyle='--', linewidth=1, label='Average + Std Dev')
            ax.axhline(avg_width[i] - deviation[i], color='green', linestyle='--', linewidth=1, label='Average - Std Dev')
            ax.set_ylabel('Width (pixels)')
            #ax.set_ylim(y_min, y_max)
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        plt.xlabel('Frame Number')
        plt.xlim(all_frame_numbers.min()), all_frame_numbers.max()
        plt.suptitle('Ego Lane Width Across Frames at Different Y Coordinates')
        plt.show()

    return ego_frames, avg_width, deviation

def determine_valid_data_kalman(csv_file_path = "X:\Life\TFG\Coding\Project_v2\data\VID_20220426_161600\lane_detection_unprocessed.csv",width_std_l=0, width_std_h=1, optical_window=60):
    
    threshold = 30
    frame_numbers, refined_coordinates = refine_lane_coordinates(csv_file_path, threshold=threshold, homography_matrix=None)
    refined_coordinates = combine_lanes(refined_coordinates)
    # Returns a list of dictionaries containining, it also returns the average mean across all rows and the standard deviation across all rows.
    ego_frames, avg_width, deviation = find_ego_frames_kalman(frame_numbers, refined_coordinates)
    print(f"Average width: {avg_width}")
    # Determines if a lane is valid by using the standard deviation.
    valid_frames = determine_valid_lanes(ego_frames, avg_width, deviation, n_std_l=10*width_std_l, n_std_u=10*width_std_h)


    dataframe = pd.DataFrame(ego_frames)
    valid_ranges = get_valid_ranges(valid_frames, dataframe, window_size=optical_window)

    return  valid_frames, valid_ranges, dataframe

if __name__ == "__main__":
    video_file = "X:\Life\TFG\Coding\Testing\Videos\BikeBi\VID_20220426_161600.mp4"
    valid_frames, valid_ranges, dataframe = determine_valid_data()
    # dataframe_optical = propagate_lanes_with_optical_flow(dataframe, valid_frames, valid_ranges, 150, video_file)
    # dataframe_optical = smooth_lane_coordinates(dataframe_optical)
    # dataframe = smooth_lane_coordinates(dataframe_optical)
    
    #ego_lane_generator = iter(view_frames(video_path=video_file, dataframe=dataframe_optical))
    #view_video_from_iterator(ego_lane_generator)