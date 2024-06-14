import tkinter as tk
from tkinter import filedialog, messagebox

from pathlib import Path
import cv2

from lane_detection import lane_detection

PROJECT_PATH = Path("../Project_v2")

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
def select_folder():
    # Derive the initial directory
    initial_directory = Path(__file__).resolve().parents[0]
    initial_directory = initial_directory / "data"

    folder_path = filedialog.askdirectory(initialdir=initial_directory)
    return folder_path

def check_files(folder_path, files_to_check=["lane_detection_unprocessed.csv", "lane_detection_processed.csv", "objects_detections_unprocessed.csv" ]):
    # List of files to check
    files_to_check = files_to_check

    # Convert folder pat to a Path object
    folder = Path(folder_path)

    # Check if each file exists in the folder
    missing_files = []
    for file_name in files_to_check:
        if not (folder / file_name).is_file():
            missing_files.append(file_name)

    # Display the result
    if missing_files:
        messagebox.showwarning("Missing Files", f"The following files are missing: {', '.join(missing_files)}")
    else:
        messagebox.showinfo("Success", "All files are present.")