from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import supervision as sv
import pandas as pd
import matplotlib.pyplot as plt
from general.utils import objectdetection_csv_to_sv_detections, path_to_files_in_folder

# # Add the "ObjectDetector" directory to sys.path
# main_path = Path(__file__).parent
# depthEstimator_path =  main_path / "Metric3D"
# sys.path.insert(0, str(depthEstimator_path))

from Metric_3D.DepthEstimator import DepthEstimator

def save_objects_detections_list_to_csv_sink(list_to_save, video_path, name):
    """
    Saves the given object detection list as a CSV file in a specified folder given a video_path and a name.

    """

    # Check if the folder to stroe data exists
    DATA_PATH = Path(f"../Project_V2/data")
    if not DATA_PATH.exists():
        raise Exception(f"The path '{DATA_PATH.as_posix()}' is not recognized.")
    video_name = Path(video_path).stem
    DATA_PATH = Path(DATA_PATH / video_name)
    if not DATA_PATH.exists():
        print(f"There is no datapath: '{DATA_PATH.as_posix()}'")

    file_path = DATA_PATH.as_posix() + "/" + f"objects_detections_{name}.csv"
    # Save the DataFrame as a CSV file to the specific folder
    csv_sink = sv.CSVSink(file_path)

    with csv_sink as sink:
        for detections, frame_number in list_to_save:
            sink.append(detections, custom_data={'Frame': frame_number})

    print(f"Successfully saved at '{DATA_PATH.as_posix()}'.")

def select_less_blurry_Laplacian(input_video_path, number_of_frames):
    """
    Loads frames from a video, computes their blurriness, and yields chunks of frames.

    Args:
        input_video_path (str): Path to the input video file.
        number_of_frames (int): Number of frames to process in each chunk.
        debug (bool, optional): If True, displays all frames and the selected least blurry frame. Defaults to False.

    Yields:
        least_blurry_frame
        frame_blur_index
        first_frame_index
        last_frame_index
    """
    cap = cv2.VideoCapture(input_video_path)
    total_frame_index = 0
    while True:
        frames = []
        for _ in range(number_of_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            total_frame_index += 1
        

        if not frames:
            cap.release()
            cv2.destroyAllWindows()
            break
        
        frame_blur_index = total_frame_index-number_of_frames
        least_blurry_frame = frames[0]
        first_frame_index = total_frame_index-len(frames)
        last_frame_index = total_frame_index - 1

        if number_of_frames > 1:
            # Compute blurriness for each frame
            blur_scores = []
            for frame in frames:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
                blur_scores.append(blur_score)

            # Find the index of the least blurry frame
            least_blurry_index = blur_scores.index(max(blur_scores))
            least_blurry_frame = frames[least_blurry_index]
            frame_blur_index = total_frame_index-number_of_frames+least_blurry_index


        yield least_blurry_frame, frame_blur_index, first_frame_index, last_frame_index

def test1():
    folder_path = "X:\Life\TFG\Coding\Project_v2\data\VID_20220426_161600"
                        
    video_path, object_detections_csv = path_to_files_in_folder(folder_path, file_names_to_check=["video_path.csv", "objects_detections_processed.csv"])

    #Create a function that reads from csv creates the detections in sv format
    detections_gen = objectdetection_csv_to_sv_detections(object_detections_csv)
    depth_estimator = DepthEstimator(model_selection="vit_small")
    cap = cv2.VideoCapture(video_path)
    # Check if it opened succesfully
    if not cap.isOpened():
        raise Exception("Error opening video file")

    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    ## Annotators
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(frame_width, frame_height))
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(frame_width, frame_height))
    # Initialize bbox annotator for drawing bounding boxes.
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    # Initialize label annotator for adding labels to objects
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detection_frame = next(detections_gen,(None, -1))
    detections_list = []
    frame_number=0
    with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detections = None
                annotated_frame = frame.copy()

                if detection_frame and detection_frame[1] == frame_number:
                    detections, _ = detection_frame
                    detection_frame = next(detections_gen, (None, -1))

                if detections is not None:
                    
                    pred_depth, pred_color = depth_estimator.prediction(annotated_frame, fx=1000.0, fy=1000.0)
                    detections = depth_estimator.draw_depth(annotated_frame, pred_depth, detections)

                    detections_list.append([detections, frame_number])

                    # Generate the labels
                    labels = [f"#{trck_id} {data['class_name']}  {data['depth']:0.2f}m" 
                            for _, _, confidence, class_id, trck_id, data in detections]
                    
                    # Annotate the labels
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame, detections=detections, labels=labels)
                    # Annotate the bounding box
                    annotated_frame = bounding_box_annotator.annotate(
                        scene=annotated_frame, detections=detections)

                show_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.imshow("Depth estimation", show_frame)
                if cv2.waitKey(1) == ord("q"):
                    break

                frame_number += 1

    cap.release()
    if len(detections_list) == total_frames:
        # Save the list of detections frome zones to CSV
        save_objects_detections_list_to_csv_sink(detections_list, video_path, name="processed")

def test_laplacian():
    folder_path = "X:\Life\TFG\Coding\Project_v2\data\VID_20220426_161600"
                        
    video_path, object_detections_csv = path_to_files_in_folder(folder_path, file_names_to_check=["video_path.csv", "objects_detections_processed.csv"])

    #Create a function that reads from csv creates the detections in sv format
    detections_gen = objectdetection_csv_to_sv_detections(object_detections_csv)
    depth_estimator = DepthEstimator(model_selection="vit_small")
    cap = cv2.VideoCapture(video_path)
    # Check if it opened succesfully
    if not cap.isOpened():
        raise Exception("Error opening video file")

    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    ## Annotators
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(frame_width, frame_height))
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(frame_width, frame_height))
    # Initialize bbox annotator for drawing bounding boxes.
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    # Initialize label annotator for adding labels to objects
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detection_frame = next(detections_gen,(None, -1))
    detections_list = []
    frame_number=0
    with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detections = None
                annotated_frame = frame.copy()

                if detection_frame and detection_frame[1] == frame_number:
                    detections, _ = detection_frame
                    detection_frame = next(detections_gen, (None, -1))

                if detections is not None:
                    
                    pred_depth, pred_color = depth_estimator.prediction(annotated_frame, fx=1000.0, fy=1000.0)
                    detections = depth_estimator.draw_depth(annotated_frame, pred_depth, detections)

                    detections_list.append([detections, frame_number])

                    # Generate the labels
                    labels = [f"#{trck_id} {data['class_name']}  {data['depth']:0.2f}m" 
                            for _, _, confidence, class_id, trck_id, data in detections]
                    
                    # Annotate the labels
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame, detections=detections, labels=labels)
                    # Annotate the bounding box
                    annotated_frame = bounding_box_annotator.annotate(
                        scene=annotated_frame, detections=detections)

                show_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.imshow("Depth estimation", show_frame)
                if cv2.waitKey(1) == ord("q"):
                    break

                frame_number += 1

    cap.release()
    if len(detections_list) == total_frames:
        # Save the list of detections frome zones to CSV
        save_objects_detections_list_to_csv_sink(detections_list, video_path, name="processed")

def test_overlap():
    folder_path = "X:\Life\TFG\Coding\Project_v2\data\VID_20220426_161600"
                        
    video_path, object_detections_csv = path_to_files_in_folder(folder_path, file_names_to_check=["video_path.csv", "objects_detections_processed.csv"])
    frame_generator = select_less_blurry_Laplacian(video_path, number_of_frames=5)
    depth_estimator = DepthEstimator(model_selection="vit_small")

    #Create a function that reads from csv creates the detections in sv format
    detections_gen = objectdetection_csv_to_sv_detections(object_detections_csv)
    
    cap = cv2.VideoCapture(video_path)
    # Check if it opened succesfully
    if not cap.isOpened():
        raise Exception("Error opening video file")

    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    ## Annotators
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(frame_width, frame_height))
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(frame_width, frame_height))
    # Initialize bbox annotator for drawing bounding boxes.
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    # Initialize label annotator for adding labels to objects
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    detection_frame = next(detections_gen, (None, -1))
    detections_list = []

    with torch.no_grad():
        for frame, frame_blur_index, first_frame_index, last_frame_index in frame_generator:
            annotated_frame = frame.copy()

            pred_depth, pred_color = depth_estimator.prediction(annotated_frame, fx=1000.0, fy=1000.0)

            # Loop trhough all detections within the range
            while detection_frame and first_frame_index <= detection_frame[1] <= last_frame_index:
                detections, frame_number = detection_frame
                detection_frame = next(detections_gen, (None, -1))

                if detections is not None:
                    
                    detections = depth_estimator.draw_depth(annotated_frame, pred_depth, detections)

                    detections_list.append([detections, frame_number])

                    # Generate the labels
                    labels = [f"#{trck_id} {data['class_name']}  {data['depth']:0.2f}m" 
                            for _, _, confidence, class_id, trck_id, data in detections]
                    
                    # Annotate the labels
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame, detections=detections, labels=labels)
                    # Annotate the bounding box
                    annotated_frame = bounding_box_annotator.annotate(
                        scene=annotated_frame, detections=detections)
                    
                show_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.imshow("Depth estimation", show_frame)
                if cv2.waitKey(1) == ord("q"):
                    break
    cap.release()
    save_objects_detections_list_to_csv_sink(detections_list, video_path, name="depth")

def depth_estimation_with_Laplacian_generator(video_path, object_detections_csv, generator_set_size = 5):
                        
    #video_path, object_detections_csv = path_to_files_in_folder(folder_path, file_names_to_check=["video_path.csv", "objects_detections_processed.csv"])
    frame_generator = select_less_blurry_Laplacian(video_path, number_of_frames=generator_set_size)
    depth_estimator = DepthEstimator(model_selection="vit_small")

    #Create a function that reads from csv creates the detections in sv format
    detections_gen = objectdetection_csv_to_sv_detections(object_detections_csv)
    
    cap = cv2.VideoCapture(video_path)
    # Check if it opened succesfully
    if not cap.isOpened():
        raise Exception("Error opening video file")

    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    ## Annotators
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(frame_width, frame_height))
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(frame_width, frame_height))
    # Initialize bbox annotator for drawing bounding boxes.
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    # Initialize label annotator for adding labels to objects
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    detection_frame = next(detections_gen, (None, -1))
    detections_list = []
    last_frame_blur_index = -1
    key=0

    with torch.no_grad():
        for frame, frame_blur_index, first_frame_index, last_frame_index in frame_generator:
            annotated_frame = frame.copy()


            # Loop trhough all detections within the range
            while detection_frame and first_frame_index <= detection_frame[1] <= last_frame_index:
                detections, frame_number = detection_frame
                detection_frame = next(detections_gen, (None, -1))
                if detections is not None and last_frame_blur_index != frame_blur_index:
                    pred_depth, pred_color = depth_estimator.prediction(annotated_frame, fx=1000.0, fy=1000.0)
                    last_frame_blur_index = frame_blur_index
                
                if detections is not None and frame_number == frame_blur_index:
                    
                    detections = depth_estimator.draw_depth(annotated_frame, pred_depth, detections)

                    detections_list.append([detections, frame_number])

                    # Generate the labels
                    labels = [f"#{trck_id} {data['class_name']}  {data['depth']:0.2f}m" 
                            for _, _, confidence, class_id, trck_id, data in detections]
                    
                    # Annotate the labels
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame, detections=detections, labels=labels)
                    # Annotate the bounding box
                    annotated_frame = bounding_box_annotator.annotate(
                        scene=annotated_frame, detections=detections)
                elif detections is not None:
                    # Create an array of NaNs for depths
                    detections.data["depth"] = np.full(len(detections), np.nan, dtype=np.float32)
                    detections_list.append([detections, frame_number])

                    # Generate the labels
                    labels = [f"#{trck_id} {data['class_name']}  {data['depth']:0.2f}m" 
                            for _, _, confidence, class_id, trck_id, data in detections]
                    
                    
                show_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.imshow("Depth estimation", show_frame)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
            if key == ord("q"):
                break
    cap.release()
    save_objects_detections_list_to_csv_sink(detections_list, video_path, name="processed")
    postprocess(video_path, object_detections_csv,  generator_set_size)

def apply_rolling_average_to_detph(csv_path, frame_distance):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Sort the DataFrame by tracker_id and Frame
    df.sort_values(by=['tracker_id', 'Frame'], inplace=True)
    
    # Define a custom rolling average function
    def custom_rolling_avg(group, frame_distance):
        depths = group['depth'].copy()
        frames = group['Frame']
        
        # Create a new column for the rolling average depth
        rolling_avg_depth = depths.copy()
        
        for i in range(len(depths)):
            # If the depth is not NaN, assign it directly to the rolling average column
            if not pd.isna(depths.iloc[i]):
                rolling_avg_depth.iloc[i] = depths.iloc[i]
            else:
                # Get the frame number of the current detection
                current_frame = frames.iloc[i]
                
                # Determine the window of frames within the frame distance
                window_mask = (frames >= current_frame - frame_distance) & (frames <= current_frame + frame_distance)
                
                # Calculate the rolling average for the depths within this window, ignoring NaNs
                rolling_avg_depth.iloc[i] = depths[window_mask].mean()
        
        return rolling_avg_depth
    
    # Apply the custom rolling average function to each group of tracker_id
    df['depth'] = df.groupby('tracker_id').apply(lambda group: custom_rolling_avg(group, frame_distance)).reset_index(level=0, drop=True)
    df.sort_values(by=['Frame'], inplace=True)
    return df

def get_frames_with_nan_depth_avg(result_df):
    # Filter the DataFrame for rows where depth_rolling_avg is NaN
    rows_without_depth_avg = result_df[result_df['depth'].isna()]

    # Obtain the list of unique frames from the filtered DataFrame
    unique_frames_with_nan = rows_without_depth_avg['Frame'].unique().tolist()

    return unique_frames_with_nan

def compute_depth_from_list(video_path, object_detections_csv, frames_with_nan):
   
    depth_estimator = DepthEstimator(model_selection="vit_small")

    # Read object detections from CSV and convert to detections generator
    detections_gen = objectdetection_csv_to_sv_detections(object_detections_csv)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    #thickness = sv.calculate_optimal_line_thickness(resolution_wh=(frame_width, frame_height))
    #text_scale = sv.calculate_optimal_text_scale(resolution_wh=(frame_width, frame_height))
    #bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    #label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    #detection_frame = next(detections_gen, (None, -1))
    detections_list = []
    
    with torch.no_grad():
        for  detections, frame_number in detections_gen:
            
            
            if frame_number in frames_with_nan:
                # Set the frame position directly to the desired frame number
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame = frame.copy()

                pred_depth, pred_color = depth_estimator.prediction(annotated_frame, fx=1000.0, fy=1000)
                detections = depth_estimator.draw_depth(annotated_frame, pred_depth, detections)

            if detections is not None:
                detections_list.append([detections, frame_number])
                
    cap.release()
    save_objects_detections_list_to_csv_sink(detections_list, video_path, name="processed")

def corrected_depth(df, tracker_id, plot=False):
    # Filter the DataFrame for the specified tracker_id
    tracker_df = df[df['tracker_id'] == tracker_id]
    
    if tracker_df.empty:
        print(f"No data found for tracker_id {tracker_id}")
        return
    
    # Get the Frame and depth columns
    frames = tracker_df['Frame'].values
    depths = tracker_df['depth'].values
    
    # Remove NaN values
    valid_mask = ~np.isnan(depths)
    frames_valid = frames[valid_mask]
    depths_valid = depths[valid_mask]
    
    # Fit a second-order polynomial to the data
    p = np.polyfit(frames_valid, depths_valid, 2)
    poly_fit = np.poly1d(p)
    
    # Generate fitted values for all frames
    estimated_depths = poly_fit(frames)
    
    # Add the estimated_depth to the DataFrame
    df.loc[df['tracker_id'] == tracker_id, 'estimated_depth'] = estimated_depths
    
    if plot:
        # Plot the original rolling average depth and the fitted depth
        plt.figure(figsize=(10, 6))
        plt.scatter(frames_valid, depths_valid, marker='o', color='b', label='Depth')
        plt.plot(frames, estimated_depths, linestyle='--', color='r', label='Estimated Depth (2nd Order)')
        plt.xlabel('Frame')
        plt.ylabel('Depth')
        plt.title(f'Depth Estimation for Tracker ID {tracker_id}')
        plt.grid(True)
        plt.legend()
        plt.show()        

def compute_corrected_depth_for_all_trackers(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure the DataFrame has an 'estimated_depth' column initialized with NaNs
    df['estimated_depth'] = np.nan
    
    # Get unique tracker IDs
    unique_tracker_ids = df['tracker_id'].unique()
    
    # Compute corrected depth for each tracker
    for tracker_id in unique_tracker_ids:
        corrected_depth(df, tracker_id)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(csv_path, index=False)

def postprocess(video_path, object_detections_csv,  generator_set_size):
    #video_path, object_detections_csv = path_to_files_in_folder(folder_path, file_names_to_check=["video_path.csv", "objects_detections_processed.csv"])
    result_df = apply_rolling_average_to_detph(object_detections_csv, frame_distance=generator_set_size*3)
    result_df.to_csv(object_detections_csv, index=False)
    frames_with_nan = get_frames_with_nan_depth_avg(result_df)
    compute_depth_from_list(video_path, object_detections_csv, frames_with_nan)
    compute_corrected_depth_for_all_trackers(object_detections_csv)

if __name__ == "__main__":
    less_blurry_window = 5
    folder_path = "X:\Life\TFG\Coding\Project_v2\data\VID_20220428_171813"
    video_path, object_detections_csv = path_to_files_in_folder(folder_path, file_names_to_check=["video_path.csv", "objects_detections_processed.csv"])
    depth_estimation_with_Laplacian_generator(video_path, object_detections_csv, generator_set_size=less_blurry_window)
    #postprocess(folder_path, generator_set_size=less_blurry_window)