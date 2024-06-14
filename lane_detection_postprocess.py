from LaneDetector.ufldDetector.utils.visualization import *
from general.utils import select_folder, path_to_files_in_folder, save_dataframe_to_csv

def determine_valid_data_from_df(metric_df, window_size = 5, r2_threshold=0.8, std_l_width=1, std_h_width=1,optical_window=30):
    
    lane_use_df, avg_r2_df = determine_lane_use(metric_df, window_size, r2_threshold)

    # Returns a list of dictionaries containining, it also returns the average mean across all rows and the standard deviation across all rows.
    ego_frames, avg_width, deviation = find_ego_frames_updated(lane_use_df, avg_r2_df)
    # Determines if a lane is valid by using the standard deviation.
    valid_frames = determine_valid_lanes(ego_frames, avg_width, deviation, n_std_l=std_l_width, n_std_u=std_h_width)


    dataframe = pd.DataFrame(ego_frames)
    plot_valid_frames(valid_frames, dataframe)
    valid_ranges = get_valid_ranges(valid_frames, dataframe, window_size=optical_window)

    return  valid_frames, valid_ranges, dataframe
    

if __name__ == "__main__":
    folder_path = select_folder()
    video_file, lane_detections_csv = path_to_files_in_folder(folder_path, file_names_to_check=["video_path.csv", "lane_detection_unprocessed.csv"])

    valid_frames, valid_ranges, dataframe = determine_valid_data(csv_file_path=lane_detections_csv, lane_thr = (5, 0.9) ,width_std_l=0.7, width_std_h=0.7, optical_window=90)
    #valid_frames, valid_ranges, dataframe = determine_valid_data_kalman(csv_file_path=lane_detections_csv, width_std_l=1, width_std_h=0.5, optical_window=30)
    dataframe_optical = propagate_lanes_with_optical_flow(dataframe, valid_frames, valid_ranges, 150, video_file)
    #dataframe_optical = propagate_lanes_with_optical_flow_updated(dataframe, valid_frames, 150, video_file, 15)
    #dataframe_optical = smooth_lane_coordinates(dataframe_optical)
    dataframe = smooth_lane_coordinates(dataframe_optical, window_size=2)

    frame_iterator = iter(view_frames(video_path=video_file, dataframe=dataframe))
    for frame_number, frame in frame_iterator:
        annotated_frame = frame.copy()
        annotated_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow("Lane Detection Processed", annotated_frame)
        if cv2.waitKey(33) == ord("q"):
            break
    cv2.destroyAllWindows()

    save_dataframe_to_csv(dataframe, video_file, "lane_detection_processed.csv")