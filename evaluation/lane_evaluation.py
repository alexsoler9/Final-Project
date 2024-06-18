
import numpy as np
import os
from tkinter.filedialog import askopenfilename
from pathlib import Path
import pandas as pd


from typing import List, Tuple
from shapely.geometry import Polygon
import ast
import matplotlib.pyplot as plt



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
	polygon = Polygon(polygon_points)
	if not polygon.is_valid:
		polygon = polygon.buffer(0)  # Fix invalid geometry
	return polygon

def compute_iou(poly1: Polygon, poly2: Polygon) -> float:
	if poly1.is_empty and poly2.is_empty:
		return 0.0
	
	if not poly1.is_valid:
		poly1 = poly1.buffer(0)  # Fix invalid geometry
	if not poly2.is_valid:
		poly2 = poly2.buffer(0)  # Fix invalid geometry
	intersection = poly1.intersection(poly2).area
	union = poly1.union(poly2).area
	if union == 0:
		return 0.0
	
	return intersection / union

def compute_metrics(gt_polygon: Polygon, dt_polygon: Polygon) -> dict:
    # If both ground truth and detected polygons are empty
    if gt_polygon.is_empty and dt_polygon.is_empty:
        return {"TP": 0, "FP": 0, "FN": 0, "TN": 1}

    # If only the ground truth polygon is empty
    if gt_polygon.is_empty and not dt_polygon.is_empty:
        return {"TP": 0, "FP": dt_polygon.area, "FN": 0, "TN": 0}

    # If only the detected polygon is empty
    if not gt_polygon.is_empty and dt_polygon.is_empty:
        return {"TP": 0, "FP": 0, "FN": gt_polygon.area, "TN": 0}

    # Calculate areas
    intersection_area = gt_polygon.intersection(dt_polygon).area
    gt_area = gt_polygon.area
    dt_area = dt_polygon.area

    # Ensure non-negative areas
    TP = max(0, intersection_area)
    FP = max(0, dt_area - intersection_area)
    FN = max(0, gt_area - intersection_area)
    TN = 0  # True negatives are not applicable in this context

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}


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
		metrics = compute_metrics(gt_polygon, dt_polygon)

		results.append({
			"Frame Number": gt_row["Frame Number"],
			"Lane IoU": lane_iou,
			"TP": metrics["TP"],
			"FP": metrics["FP"],
			"FN": metrics["FN"],
			"TN": metrics["TN"],
		})

	results_df = pd.DataFrame(results)

	# Calculate Precision, Recall, and F1-score
	TP_sum = results_df["TP"].sum()
	FP_sum = results_df["FP"].sum()
	FN_sum = results_df["FN"].sum()
	TN_sum = results_df["TN"].sum()

	precision = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) > 0 else 0
	recall = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else 0
	f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

	summary_metrics = {
		"Precision": precision,
		"Recall": recall,
		"F1-score": f1_score
	}

	return results_df, summary_metrics

def plot_metrics(results_df: pd.DataFrame, summary_metrics: dict):
	frames = results_df["Frame Number"]
	ious = results_df["Lane IoU"]
	precisions = results_df["TP"] / (results_df["TP"] + results_df["FP"])
	recalls = results_df["TP"] / (results_df["TP"] + results_df["FN"])
	f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

	fig, ax = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

	ax[0].plot(frames, ious, label="IoU")
	ax[0].axhline(y=ious.mean(), color="r", linestyle="--", label="Average IoU")
	ax[0].set_ylabel("IoU")
	ax[0].legend()

	ax[1].plot(frames, precisions, label="Precision")
	ax[1].axhline(y=summary_metrics["Precision"], color="r", linestyle="--", label="Average Precision")
	ax[1].set_ylabel("Precision")
	ax[1].legend()

	ax[2].plot(frames, recalls, label="Recall")
	ax[2].axhline(y=summary_metrics["Recall"], color="r", linestyle="--", label="Average Recall")
	ax[2].set_ylabel("Recall")
	ax[2].legend()

	ax[3].plot(frames, f1_scores, label="F1 Score")
	ax[3].axhline(y=summary_metrics["F1-score"], color="r", linestyle="--", label="Average F1 Score")
	ax[3].set_xlabel("Frame Number")
	ax[3].set_ylabel("F1 Score")
	ax[3].legend()
	fig.suptitle("Lane Metrics for VID_20220427_144342")
	plt.show()


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

	gt_file = "X:\Life\TFG\Coding\Project_v2\data\VID_20220427_144342\GT_LD_VID_20220427_144342.csv"
	dt_file = "X:\Life\TFG\Coding\Project_v2\data\VID_20220427_144342\lane_detection_processed.csv"
	gt_df = load_data(gt_file)
	dt_df = load_data(dt_file)
	results, summary_metrics = evaluate_lanes(gt_df, dt_df)
	#plot_iou(results)

	f1_score = compute_f1_score(gt_df, dt_df)
	print(f"F1 Score: {f1_score:.4f}")
	plot_metrics(results, summary_metrics)