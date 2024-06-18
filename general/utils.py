from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import cv2

import supervision as sv
import ast

PROJECT_PATH = Path("../Project_v2")

def objectdetection_csv_to_sv_detections(csv_file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Group the DataFrame by the "Frame" column
    grouped = df.groupby("Frame")

    for frame_number, group in grouped:
        # Extract bounding box coordinates
        xyxy = group[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
        # Extract confidence score
        confidence = group["confidence"].to_numpy()
        # Extract class IDs
        class_id = group["class_id"].to_numpy()
        # Extract tracker IDs
        tracker_id = group["tracker_id"].to_numpy()

        data = {}
        # Check if "class_name" and "depth" columns exist
        if "class_name" in group.columns:
            data["class_name"] = group["class_name"].to_numpy()
        if "depth" in group.columns:
            data["depth"] = group["depth"].to_numpy()
        if "estimated_depth" in group.columns:
            data["estimated_depth"] = group["estimated_depth"].to_numpy()
        

        # Create the Detections object
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
            data=data
        )

        # Yield the detecctions object and the frame number
        yield detections, frame_number

def lanedetection_csv_to_polygons(csv_file_path, column1="Ego Left Lane", column2="Ego Right Lane"):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        frame_number = row["Frame Number"]

        # Extract points from te specified columns and convert from string to a list of tuples
        points1 = row[column1]
        points2 = row[column2]

        if pd.notna(points1) and pd.notna(points2):
            points1 = ast.literal_eval(points1)
            points2 = ast.literal_eval(points2)

            # Form the polygon by combining the points from both columns
            polygon = points1 + points2[::-1] # Reverse the order of the second list to form a closed polygon

            yield polygon, frame_number
        else:
            yield [], frame_number

def select_folder():
    # Derive the initial directory
    initial_directory = Path(__file__).resolve().parents[0]
    initial_directory = initial_directory / "data"

    folder_path = filedialog.askdirectory(initialdir=initial_directory)
    return folder_path

def check_data_folder(file_name):
	
	# Check if folder to store data exists
	DATA_PATH = PROJECT_PATH / "data"
	if not DATA_PATH.exists():
		raise Exception(f"The path '{DATA_PATH.as_posix()}' is not recognized.")
	video_name = Path(file_name).stem
	
	DATA_PATH = Path(DATA_PATH / video_name)
	if not DATA_PATH.exists():
		print(f"Creating data folder in {DATA_PATH.as_posix()}")
		DATA_PATH.mkdir()

	return DATA_PATH

def path_to_files_in_folder(folder_path, file_names_to_check=["video_path.csv", "objects_detections_processed.csv"]):
    folder_path = Path(folder_path)
    if not folder_path.exists() or str(folder_path)==".":
        raise Exception(f"Could not find folder path at: '{folder_path.as_posix()}'")
    file_paths = []
    for file_name in file_names_to_check:
        file_path = folder_path / file_name
        if not file_path.exists():
            raise Exception(f"Could not find file path: '{file_path.as_posix()}'")
        if file_name == "video_path.csv":
            try:
                df = pd.read_csv(file_path)
                video_paths = df["video_path"].tolist()
                print(f"Video paths from 'video_path.csv': {video_paths}")
            except pd.errors.EmptyDataError:
                print(f"'{file_name}' is empty.")
            except KeyError:
                print(f"Column 'video_path' not found in '{file_name}'.")
            except Exception as e:
                print(f"Error reading '{file_name}': {e}")
            file_path = video_paths[0]
        file_paths.append(file_path)

    return file_paths


def save_dataframe_to_csv(dataframe, video_path, name=None):
    """
    Saves the given dataframe as a CSV file in a specified folder given a video_path.

    """

    # Check if the folder to store data exists
    DATA_PATH = Path(f"../Project_V2/data")
    if not DATA_PATH.exists():
        raise Exception(f"The path '{DATA_PATH.as_posix()}' is not recognized.")
    video_name = Path(video_path).stem
    print(video_name)
    DATA_PATH = Path(DATA_PATH / video_name)
    if not DATA_PATH.exists():
        print(f"Creating data folder in {DATA_PATH.as_posix()}")
        DATA_PATH.mkdir()
    if name is None:
        raise ValueError("Provide a name")
    file_path = DATA_PATH.as_posix() + "/" + name
    # Save the DataFrame as a CSV file to the specific folder

    dataframe.to_csv(file_path, index=False)
    print(f"Data frame successfully saved at '{DATA_PATH.as_posix()}'.")

def save_video_path(data_path, video_path):
    # Check if the folder to store data exists
    DATA_PATH = Path(f"../Project_V2/data")
    if not DATA_PATH.exists():
        raise Exception(f"The path '{DATA_PATH.as_posix()}' is not recognized.")
    video_name = Path(video_path).stem
    DATA_PATH = Path(DATA_PATH / video_name)
    if not DATA_PATH.exists():
        print(f"Creating data folder in {DATA_PATH.as_posix()}")
        DATA_PATH.mkdir()
    file_path = DATA_PATH.as_posix() + "/" + "video_path.csv"
    df = pd.DataFrame({"video_path": video_path}, index=[0])
    df.to_csv(file_path, index=False)
        
        
