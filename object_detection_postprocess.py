from ObjectDetector.utils.visualization import *
import cv2

from general.utils import save_dataframe_to_csv

def postprocess_object_detection(df_objects_detection_unprocessed: pd.DataFrame):
        
    # Obtain the unique IDs
    tracker_ID_df = unique_ID_dict(df_objects_detection_unprocessed)

    #df = assign_class_to_tracker(df)

    # Maintain only the track IDs that appear at least
    appearances = 0
    result_ids = get_ids_by_appearances(tracker_ID_df, appearances)
    filter_data = filter_data_by_ids(df_objects_detection_unprocessed, result_ids)

    # Combine persons with bikes and motorcycles
    combined_df = update_dataframe_improved(filter_data)

    # Obtain a dictionary of groups of id
    grouped_list = group_riders_trackers_id(combined_df)
    #print(grouped_list)
    updated_detections = propagate_detection(combined_df)

    return updated_detections


if __name__ == "__main__":

    csvfile = "VID_20220428_171813\objects_detections_unprocessed.csv"
    videofile = "X:\Life\TFG\Coding\Testing\Videos\BikeU\VID_20220428_171813.mp4"

    # Read CSV file
    df = load_csv(csvfile)

    # Obtain the unique IDs
    tracker_ID_df = unique_ID_dict(df)

    #df = assign_class_to_tracker(df)

    # Maintain only the track IDs that appear at least
    appearances = 0
    result_ids = get_ids_by_appearances(tracker_ID_df, appearances)
    filter_data = filter_data_by_ids(df, result_ids)

    # Combine persons with bikes and motorcycles
    combined_df = update_dataframe_improved(filter_data)

    # Obtain a dictionary of groups of id
    grouped_list = group_riders_trackers_id(combined_df)
    print(grouped_list)
    updated_detections = propagate_detection(combined_df)
    #updated_detections = combine_detections(updated_detections)

    frame_iterator = iter(display_bounding_boxes_new(updated_detections, videofile))

    for frame, frame_number in frame_iterator:
        annotated_frame = frame.copy()
        annotated_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow("Object Detection", annotated_frame)
        if cv2.waitKey(33) == ord("q"):
            break
    cv2.destroyAllWindows()

    save_dataframe_to_csv(updated_detections, videofile, "objects_detections_processed.csv")