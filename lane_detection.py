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


#################################
#			EVALUATION			#
#################################

def load_data(file_path: str) -> pd.DataFrame:
	return pd.read_csv(file_path)

def parse_lane(lane_str: str) -> List[Tuple[int, int]]:
	if isinstance(lane_str, float) and np.isnan(lane_str):
		return []
	if lane_str == "" or lane_str == "[]":
		return []
	return ast.literal_eval(lane_str)

def create_polygon(left_lane: List[Tuple[int, int]], right_lane: List[Tuple[int, int]]) -> Polygon:
	if not left_lane or not right_lane:
		return Polygon()
	# Combine left_lane and reversed right lane to form a closed polygon
	polygon_points = left_lane + right_lane[::-1]
	return Polygon(polygon_points)

def compute_iou(poly1: Polygon, poly2: Polygon) -> float:
	if poly1.is_empty or poly2.is_empty:
		return 0.0
	intersection = poly1.intersection(poly2).area
	union = poly1.union(poly2).area
	if union == 0:
		return 0.0
	
	return intersection / union

def evaluate_lanes(gt_df: pd.DataFrame, dt_df: pd.DataFrame) -> pd.DataFrame:
	print(gt_df.shape, dt_df.shape)
	assert gt_df.shape[0] == dt_df.shape[0], "Ground truth and detection files should have the same number of rows"
	results = []

	for idx, gt_row in gt_df.iterrows():
		dt_row = dt_df.iloc[idx]

		gt_left_lane = parse_lane(gt_row["Ego Left Lane"])
		gt_right_lane = parse_lane(gt_row["Ego Right Lane"])
		dt_left_lane = parse_lane(dt_row["Ego Left Lane"])
		dt_right_lane = parse_lane(dt_row["Ego Right Lane"])

		gt_polygon = create_polygon(gt_left_lane, gt_right_lane)
		dt_polygon = create_polygon(dt_left_lane, dt_right_lane)

		lane_iou = compute_iou(gt_polygon, dt_polygon)

		results.append({
			"Frame Number": gt_row["Frame Number"],
			"Lane IoU": lane_iou,
		})

	return pd.DataFrame(results)

def plot_iou(results: pd.DataFrame):
	plt.figure(figsize=(10, 6))
	plt.plot(results["Frame Number"], results["Lane IoU"], label="Lane IoU")
	avg_iou = results["Lane IoU"].mean()
	plt.axhline(y=avg_iou, linestyle='--', color='r', label=f'Average IoU: {avg_iou:.2f}')
	plt.xlabel("Frame Number")
	plt.ylabel("IoU")
	plt.title("Lane IoU for VID_20220426_161600")
	plt.legend()
	plt.grid(True)
	plt.show()

def compute_f1_score(gt_df: pd.DataFrame, dt_df: pd.DataFrame) -> float:
	assert gt_df.shape[0] == dt_df.shape[0], "Ground truth and detection files should have the same number of rows"

	total_TP = 0
	total_FP = 0
	total_FN = 0

	for idx, gt_row in gt_df.iterrows():
		dt_row = dt_df.iloc[idx]

		gt_left_lane = parse_lane(gt_row["Ego Left Lane"])
		gt_right_lane = parse_lane(gt_row["Ego Right Lane"])
		dt_left_lane = parse_lane(dt_row["Ego Left Lane"])
		dt_right_lane = parse_lane(dt_row["Ego Right Lane"])

		gt_polygon = create_polygon(gt_left_lane, gt_right_lane)
		dt_polygon = create_polygon(dt_left_lane, dt_right_lane)

		intersection = gt_polygon.intersection(dt_polygon)

		# Calculate True Positives (TP)
		TP = intersection.area

		# Calculate False Positives (FP)
		FP = dt_polygon.area - TP

		# Calculate  False Negatives (FN)
		FN = gt_polygon.area - TP

		total_TP += TP
		total_FP += FP
		total_FN += FN
	
	precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
	recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0

	if precision + recall == 0:
		return 0.0
	
	f1_score = 2 * (precision * recall) / (precision + recall)

	return f1_score

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