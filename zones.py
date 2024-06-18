import pandas as pd
import numpy as np
import cv2
from typing import Iterable, Optional, Tuple
from supervision.geometry.core import Position

import supervision as sv
from supervision.detection.utils import clip_boxes, polygon_to_mask
from supervision.geometry.utils import get_polygon_center
from supervision.draw.utils import draw_polygon, draw_text
import ast
from ObjectDetector.ZoneMonitor import ZoneMonitor as zn
from pathlib import Path

from general.utils import objectdetection_csv_to_sv_detections, lanedetection_csv_to_polygons

class PolygonZoneExtended(sv.PolygonZone):
    def update_polygon(self, polygon):
        self.triggering_anchors = (Position.BOTTOM_CENTER,)
        self.polygon = polygon
        x_max, y_max = np.max(polygon, axis=0)
        self.frame_resolution_wh = (x_max + 1, y_max + 1)
        self.mask = polygon_to_mask(
            polygon=polygon, resolution_wh=(x_max + 2, y_max + 2)
            )

class PolygonZoneAnnotatorExtended(sv.PolygonZoneAnnotator):
    def annotate(self, scene: np.ndarray, label: Optional[str] = None) -> np.ndarray:
        """
        Annotates the polygon zone within a frame with a count of detected objects.

        Parameters:
            scene (np.ndarray): The image on which the polygon zone will be annotated
            label (Optional[str]): An optional label for the count of detected objects
                within the polygon zone (default: None)

        Returns:
            np.ndarray: The image with the polygon zone and count of detected objects
        """
        annotated_frame = draw_polygon(
            scene=scene,
            polygon=self.zone.polygon,
            color=self.color,
            thickness=self.thickness,
        )
        self.center = get_polygon_center(polygon=self.zone.polygon)
        if self.display_in_zone_count:
            annotated_frame = draw_text(
                scene=annotated_frame,
                text=str(self.zone.current_count) if label is None else label,
                text_anchor=self.center,
                background_color=self.color,
                text_color=self.text_color,
                text_scale=self.text_scale,
                text_thickness=self.text_thickness,
                text_padding=self.text_padding,
                text_font=self.font,
            )

        return annotated_frame


def save_list_to_csv(list_to_save, video_path, name):
    """
    Saves the given list as a CSV file in a specified folder given a video_path and a name.

    """

    # Check if the folder to stroe data exists
    DATA_PATH = Path(f"./data")
    if not DATA_PATH.exists():
        raise Exception(f"The path '{DATA_PATH.as_posix()}' is not recognized.")
    video_name = Path(video_path).stem
    print(video_name)
    DATA_PATH = Path(DATA_PATH / video_name)
    if not DATA_PATH.exists():
        print(f"There is no datapath: '{DATA_PATH.as_posix()}'")
    DATA_PATH = Path(DATA_PATH / "Zones")
    if not DATA_PATH.exists():
        print(f"Creating Zones folder at: '{DATA_PATH.as_posix()}'")
        DATA_PATH.mkdir()

    file_path = DATA_PATH.as_posix() + "/" + f"detections_{name}.csv"
    # Save the DataFrame as a CSV file to the specific folder
    csv_sink = sv.CSVSink(file_path)

    with csv_sink as sink:
        for detections, frame_number in list_to_save:
            sink.append(detections, custom_data={'Frame': frame_number})

    print(f"Successfully saved at '{DATA_PATH.as_posix()}'.")

def overlay_detections_and_polygons(video_path, detections_csv, lane_csv, output_path = None):
    detections_gen = objectdetection_csv_to_sv_detections(detections_csv)
    lanes_gen = lanedetection_csv_to_polygons(lane_csv)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    detection_frame, lane_frame = next(detections_gen,(None, -1)), next(lanes_gen, (None, -1))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if detection_frame and detection_frame[1] == frame_number:
            detections, _ = detection_frame
            for i, bbox in enumerate(detections.xyxy):
                x_min, y_min, x_max, y_max = bbox
                class_name = detections.data["class_name"][i]
                confidence = detections.confidence[i]
                label = f"c: {class_name}, conf:{confidence:.2f}"
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255,0, 0), 2)
                cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            detection_frame = next(detections_gen, (None, -1))
        
        if lane_frame and lane_frame[1] == frame_number:
            polygon, _ = lane_frame
            if polygon:
                polygon = np.array(polygon, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)
            lane_frame = next(lanes_gen, (None, -1))

        annotated_frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow("Object Detection", annotated_frame)
        if cv2.waitKey(int(fps)) == ord("q"):
            break

        if output_path: out.write(frame)
        frame_number += 1

    cap.release()
    if output_path: out.release()

def overlay_detections_and_polygons_with_zones(video_path, detections_csv, lane_csv, output_path):
    detections_gen = objectdetection_csv_to_sv_detections(detections_csv)
    lanes_gen = lanedetection_csv_to_polygons(lane_csv)
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # POLYGON
    bottom_right = np.array([frame_width, frame_height], dtype=np.int64)
    bottom_left = np.array([0, frame_height], dtype=np.int64)
    no_zone_polygon = np.vstack([bottom_left, bottom_right])

    colors = sv.ColorPalette.DEFAULT
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(frame_width, frame_height))
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(frame_width, frame_height))

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    polygon_zone_lane = PolygonZoneExtended(no_zone_polygon, frame_resolution_wh=(frame_width, frame_height))
    polygon_zone_out = PolygonZoneExtended(no_zone_polygon, frame_resolution_wh=(frame_width, frame_height))

    TRACK_BUFFER_S = 5
    zone_monitor_lane = zn.ZoneMonitor(in_threshold=int(fps)//2, out_timeout=int(fps)*TRACK_BUFFER_S)
    zone_monitor_out = zn.ZoneMonitor(in_threshold=int(fps)//2, out_timeout=int(fps)*TRACK_BUFFER_S)
    
    zone_annotator_lane = PolygonZoneAnnotatorExtended(zone=polygon_zone_lane, color=colors.by_idx(0), thickness=6, text_thickness=3, text_scale=1)
    zone_annotator_out = PolygonZoneAnnotatorExtended(zone=polygon_zone_out, color=colors.by_idx(0), thickness=6, text_thickness=3, text_scale=1)

    detection_frame, lane_frame = next(detections_gen,(None, -1)), next(lanes_gen, (None, -1))
    count_entered_lane, count_out = 0, 0
    frame_number = 0

    # Lists to store the detections data
    inside_detections_list = []
    outside_detections_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = None
        annotated_frame = frame.copy()

        if detection_frame and detection_frame[1] == frame_number:
            detections, _ = detection_frame
            detection_frame = next(detections_gen, (None, -1))

        if lane_frame and lane_frame[1] == frame_number:
            lane_polygon, _ = lane_frame
            lane_polygon = np.array(lane_polygon, dtype=np.int32)
            if len(lane_polygon) == 0: lane_polygon = no_zone_polygon
            lane_frame = next(lanes_gen, (None, -1))

            frame_width, frame_height= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            upper_right_point = np.array([frame_width, 0], dtype=np.int32)
            upper_left_points = np.array([0, 0], dtype=np.int32)
            out_polygon = np.vstack([bottom_left, lane_polygon, bottom_right, upper_right_point, upper_left_points])
            polygon_zone_lane.update_polygon(lane_polygon)
            polygon_zone_out.update_polygon(out_polygon)
        
        if detections is not None:
            try:
                lane_mask = polygon_zone_lane.trigger(detections)
                out_mask = polygon_zone_out.trigger(detections)
            except Exception as e:
                print(frame_number)
                print(e)

            detections_lane = detections[lane_mask]
            detections_out = detections[out_mask]

            entered_detections, _ = zone_monitor_lane.update(detections_lane)
            count_entered_lane += len(entered_detections)
            out_detections, _ = zone_monitor_out.update(detections_out)
            count_out += len(out_detections)

            inside_detections_list.append([detections_lane, frame_number])
            outside_detections_list.append([detections_out, frame_number])

            labels = [f"c: {data['class_name']} id: {track_id}" 
                    for (_, _, confidence, class_id, track_id, data) in detections_lane]
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections_lane, labels=labels)
            annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections_lane)

            labels = [f"c: {data['class_name']} id: {track_id}" for (_, _, confidence, class_id, track_id, data) in detections_out]
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections_out, labels=labels)
            annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections_out)

        annotated_frame = zone_annotator_lane.annotate(annotated_frame, f"Total Lane: {count_entered_lane}")
        annotated_frame = zone_annotator_out.annotate(annotated_frame, f"Total Out Lane: {count_out}")

        view_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow("Object Detection", view_frame)
        if cv2.waitKey(int(fps)) == ord("q"):
            break

        if output_path:
            out.write(annotated_frame)
        frame_number += 1
        

    cap.release()
    cv2.destroyAllWindows()
    if output_path: out.release()

    # Save the list of detections frome zones to CSV
    save_list_to_csv(inside_detections_list, video_path, name="inside")
    save_list_to_csv(outside_detections_list, video_path, name="outside")
        

if __name__ == "__main__":

    video_path = "X:\Life\TFG\Coding\Testing\Videos\BikeU\VID_20220428_171813.mp4"
    detections_csv = "X:\Life\TFG\Coding\Project_v2\data\VID_20220428_171813\objects_detections_processed.csv"
    lane_csv = "X:\Life\TFG\Coding\Project_v2\data\VID_20220428_171813\lane_detection_processed.csv"
    output_path = None
    overlay_detections_and_polygons_with_zones(video_path, detections_csv, lane_csv, output_path)