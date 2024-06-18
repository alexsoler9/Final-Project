import cv2
import numpy as np
import os
from tkinter.filedialog import askopenfilename
from pathlib import Path
import pandas as pd

from LaneDetector.ufldDetector.ultrafastLaneDetectorV2 import UltrafastLaneDetectorV2
from LaneDetector.ufldDetector.utils.type import LaneModelType

from typing import List, Tuple
from shapely.geometry import Polygon
import ast
import matplotlib.pyplot as plt

def check_lane_detection_OLD(file_name):
	global SAVE_COORDS
	global VIEW 
	video_name = os.path.splitext(file_name)[0]
	file_path = f"data/lane_detection_csv/ld_{video_name}.csv"
	if os.path.exists(file_path):
		user_input = input("Do you want to recompute lane detections? (Y/n)").strip().lower()

		if user_input == "y" or "":
			print("Recomputing lane detection...")
			
		elif user_input == "n":
			print("Using existing values.")
			SAVE_COORDS = False
			VIEW = False
	else:
		SAVE_COORDS = True
		VIEW = True

def check_data_folder(file_name):
	global SAVE_COORDS
	global VIEW
	# Check if folder to store data exists
	DATA_PATH = Path(f"../Project_V2/data")
	if not DATA_PATH.exists():
		raise Exception(f"The path '{DATA_PATH.as_posix()}' is not recognized.")
	video_name = Path(file_name).stem
	print(video_name)
	DATA_PATH = Path(DATA_PATH / video_name)
	if not DATA_PATH.exists():
		print(f"Creating data folder in {DATA_PATH.as_posix()}")
		DATA_PATH.mkdir()

	SAVE_COORDS = False
	VIEW = True
	return DATA_PATH


def update_dataframe(dataframe, frame_number, coords):
	"""
	Updates the Dataframe with frame number and coordinates list for a given frame.

	Paramaters:
		dataframe (pd.DataFrame): Existing Dataframe to update.
		frame_number (int): Frame number.
		coords (list): List of coordinates (e.g., [(x1, y1),(x2, y2), ..., (xn, yn)]).

	Returns: 
		pd.Dataframe: Updated Dataframe.
	"""	
	# Create a new row for the current frame
	new_row = {}
	new_row["Frame Number"] = int(frame_number)
	
	# Convert each coordinate list to a Numpy array of integers
	for index, coordinates in enumerate(coords):
		column_name = f"Lane_{index}_Coordinates"
		# Clean coordinates data by removing newline characters and convert to Numpy array
		
		coordinates_str = ';'.join([f'{x},{y}' for x, y in coordinates])
		new_row[column_name] = coordinates_str

	extended_df = pd.DataFrame([new_row])
	
	# Append the new row to the existing Dataframe
	dataframe = pd.concat([dataframe, extended_df], ignore_index=True)
	return dataframe

def save_video_path(data_path, video_path):
	file_path = data_path.as_posix() + "/" + "video_path.csv"
	df = pd.DataFrame({"video_path": video_path}, index=[0])
	df.to_csv(file_path, index=False)

# lane_config = {
# 	"model_path" : "./LaneDetector/models/tusimple_res18.pth",
# 	"model_config" : "./LaneDetector/configs/tusimple_res18.py",
# 	"model_type" : LaneModelType.UFLDV2_TUSIMPLE
# }
# SAVE_COORDS = True
# VIEW = True

# video_path = askopenfilename()
# video_name = os.path.basename(video_path)
# print(f"Selected file path: {video_path}")
# print(f"Selected file: {video_name}")
# data_path = check_data_folder(video_name)
# save_video_path(data_path, video_path)

# cap = cv2.VideoCapture(video_path)


# # Initialize lane detection model
# #lane_detector = UltrafastLaneDetectorv2(arguments, use_gpu, save_coords = False, video_name=video_name)
# LINE_POINT = 1200

# #cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	

# RECORD = False
# if RECORD:
# 	# Get the video frame properties
# 	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
# 	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
# 	fps = int(cap.get(cv2.CAP_PROP_FPS))
# 	# Define the codec and create VideoWriter object
# 	output_path = "UFLDv2_crop_" + video_name
# 	fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for mp4 format
# 	out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# key = ord("q")

# df = pd.DataFrame()

# # Lane Detection Model Initializer
# if ("UFLDV2" in lane_config["model_type"].name):
# 	UltrafastLaneDetectorV2.set_defaults(lane_config)
# 	laneDetector = UltrafastLaneDetectorV2(save_coords=SAVE_COORDS, video_name=video_name)
# elif ("UFLD" in lane_config["model_type"].name):
# 	NotImplemented

# while cap.isOpened():
	
# 	# Read frame from the video
# 	frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
# 	print(frame_number)
# 	ret, frame = cap.read()
# 	if not ret:
# 		break
	
# 	frame_cpy = frame.copy()
# 	cropped_frame = frame_cpy[:LINE_POINT, :]
# 	# Detect the lanes
# 	output_img, coords = laneDetector.detect_lanes(cropped_frame)
# 	df = update_dataframe(df, frame_number, coords)

# 	if VIEW:
# 		frame_cpy[:LINE_POINT, :] = output_img[:LINE_POINT, :] 
# 		output_img = cv2.resize(frame_cpy, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
# 		cv2.imshow("Detected lanes", output_img)
# 		key = cv2.waitKey(1)

# 	#output_img = cv2.resize(output_img, (1080,LINE_POINT), interpolation=cv2.INTER_LINEAR)
# 	if RECORD:
# 		if ~VIEW:
# 			frame_cpy[:LINE_POINT, :] = output_img[:LINE_POINT, :] 
# 			output_img = cv2.resize(frame_cpy, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
# 		out.write(output_img)

# 	# Press key q to stop
# 	if key == ord('q'):
# 		break
# 	elif key == ord("p"):
# 		cv2.waitKey(0)
# 	elif key == ord('v'):
# 		VIEW = False
# 		cv2.destroyAllWindows()

# file_path = data_path.as_posix() + "/" + "lane_detection_unprocessed.csv"
# # Save the DataFrame as a CSV file to the specific folder
# df.to_csv(file_path, index=False)
# cap.release()
# cv2.destroyAllWindows()


### FINAL
def lane_detection(video_path, LINE_POINT=1200):
	video_name = os.path.basename(video_path)
	print(f"Selected file path: {video_path}")
	print(f"Selected file: {video_name}")
	data_path = check_data_folder(video_name)

	# Lane detection model selection
	lane_config = {
		"model_path" : "./LaneDetector/models/tusimple_res18.pth",
		"model_config" : "./LaneDetector/configs/tusimple_res18.py",
		"model_type" : LaneModelType.UFLDV2_TUSIMPLE
	}

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise Exception(f"Error opening video file. No such file at '{video_path}'.")	
	save_video_path(data_path, video_path)

	# Lane Detection Model Initializer
	if ("UFLDV2" in lane_config["model_type"].name):
		UltrafastLaneDetectorV2.set_defaults(lane_config)
		laneDetector = UltrafastLaneDetectorV2(save_coords=SAVE_COORDS, video_name=video_name)
	elif ("UFLD" in lane_config["model_type"].name):
		NotImplemented
	
	df = pd.DataFrame()

	frame_number = 0
	while cap.isOpened():
	
		# Read frame from the video
		#frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
		
		ret, frame = cap.read()
		if not ret:
			break
		
		frame_cpy = frame.copy()
		cropped_frame = frame_cpy[:LINE_POINT, :]
		# Detect the lanes
		output_img, coords = laneDetector.detect_lanes(cropped_frame)
		df = update_dataframe(df, frame_number, coords)

		frame_cpy[:LINE_POINT, :] = output_img[:LINE_POINT, :] 
		output_img = cv2.resize(frame_cpy, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
		cv2.imshow("Detected lanes", output_img)
		key = cv2.waitKey(1)

		# Press key q to stop
		if key == ord('q'):
			break
		elif key == ord("p"):
			cv2.waitKey(0)
		elif key == ord('v'):
			VIEW = False
			cv2.destroyAllWindows()
		frame_number += 1

	file_path = data_path.as_posix() + "/" + "lane_detection_unprocessed.csv"
	# Save the DataFrame as a CSV file to the specific folder
	df.to_csv(file_path, index=False)
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	# video_path = askopenfilename()
	# LINE_POINT = 1220

	# lane_detection(video_path, LINE_POINT=LINE_POINT)

	gt_file = "X:\Life\TFG\Coding\Testing\Videos\BikeBi\GT_LD_VID_20220426_161600.csv"
	dt_file = "X:\Life\TFG\Coding\Project_v2\data\VID_20220426_161600\lane_detection_processed.csv"
	gt_df = load_data(gt_file)
	dt_df = load_data(dt_file)
	results = evaluate_lanes(gt_df, dt_df)
	plot_iou(results)

	f1_score = compute_f1_score(gt_df, dt_df)
	print(f"F1 Score: {f1_score:.4f}")