from ObjectDetector.utils.visualization import *

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk

#from utils_app import *
from general.utils import select_folder, path_to_files_in_folder, check_data_folder, save_dataframe_to_csv
from lane_detection import lane_detection
from LaneDetector.ufldDetector.utils.visualization import compute_lane_metrics_across_frames, propagate_lanes_with_optical_flow, smooth_lane_coordinates, view_frames
from lane_detection_postprocess import determine_valid_data_from_df
from object_detection import object_detection_in_video
from object_detection_postprocess import postprocess_object_detection
from depth_estimation import depth_estimation_with_Laplacian_generator
from zones import overlay_detections_and_polygons_with_zones

# Initialize the video_path variable
video_path = None

def select_video():
    global video_path
    video_path = None
    # Open a file dialog to select a video file
    file_path = filedialog.askopenfilename(
        title="Select Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
    # Check if a file was selected
    if file_path:
        video_path = file_path
        messagebox.showinfo("Video Selected", f"Selected Video: {video_path}")
    else:
        messagebox.showwarning("No Selection", "No video selected")

#############################################
#               LANE DETECTION              #
#############################################
def perform_lane_detection():
    global video_path

    # Check if video_path is set
    if video_path:
        # Perform lane detection
        try:
            # Open the video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mid_frame = total_frames//2

            # Set the frame position to the middle frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to read the frame from the video")
            
            cap.release()

            # Resize the frame to fit the screen
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()//1.5
            frame_height, frame_width = frame.shape[:2]

            # Calculate the new size while maintaining the aspect ratio
            scaling_factor = min(screen_width / frame_width, screen_height / frame_height)
            new_size = (int(frame_width * scaling_factor), int(frame_height * scaling_factor))
            frame = cv2.resize(frame, new_size)

            # Create a new window for displaying the frame
            display_window = tk.Toplevel(root)
            display_window.title("Lane Detection - Middle Frame")

            # Convert the frame to a ImageTk object
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=image)

            # Function to update the canvas witht the slider position
            def update_line(val):
                y_pos = int(val)*scaling_factor
                canvas.delete("line")
                canvas.create_line(0, y_pos, new_size[0], y_pos, fill="red", tags="line")

            # Function to handle the accept button click
            def accept():
                y_pos = int(slider.get())
                display_window.destroy()
                lane_detection(video_path, LINE_POINT=y_pos)
                messagebox.showinfo("Lane Detection", f"Lane detection completed succesfully.")

            # Create a slider
            slider = tk.Scale(display_window, from_=0, to=frame_height, orient=tk.VERTICAL, command=update_line)
            slider.pack(side=tk.LEFT)

            # Create a canvas to dsiplay the image
            canvas = tk.Canvas(display_window, width=new_size[0], height=new_size[1])
            canvas.pack()
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            # Create an accept button
            accept_button = tk.Button(display_window, text="Accept", command=accept)
            accept_button.pack(pady=20)
            slider.pack(fill=tk.Y)
            
            # Keep a reference to the image object
            canvas_image = imgtk
            display_window.mainloop()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("No Video", "Please select a video first.")

def lane_postprocess():
    global video_path

    # Check if video_path is set
    if video_path:
        # Obtain the video name
        video_name = Path(video_path).stem
        data_path = check_data_folder(video_name)
        video_file, lane_detections_csv = path_to_files_in_folder(data_path, file_names_to_check=["video_path.csv", "lane_detection_unprocessed.csv"])
        df = pd.read_csv(lane_detections_csv)
        # Uses the information from the original dataframe to compute: predicted coordiantes at certain y_values, compute R^2, and variance
        metric_df = compute_lane_metrics_across_frames(df)
        #Open a new window for furhter processing
        postprocess_window = tk.Toplevel(root)
        postprocess_window.title("Lane Postprocess")

        # Define default values
        default_window_size     = tk.StringVar(value="5")
        default_r2_threshold    = tk.StringVar(value="0.8")
        default_std_l_width     = tk.StringVar(value="1")
        default_std_h_width     = tk.StringVar(value="1")
        default_optical_window  = tk.StringVar(value="30")

        # Create entry widgets for parameters
        tk.Label(postprocess_window, text="Window Size:").pack()
        window_size_entry = tk.Entry(postprocess_window, textvariable=default_window_size)
        window_size_entry.pack()

        tk.Label(postprocess_window, text="R2 Threshold:").pack()
        r2_threshold_entry = tk.Entry(postprocess_window, textvariable=default_r2_threshold)
        r2_threshold_entry.pack()

        tk.Label(postprocess_window, text="Std Lower Width:").pack()
        std_l_width_entry = tk.Entry(postprocess_window, textvariable=default_std_l_width)
        std_l_width_entry.pack()

        tk.Label(postprocess_window, text="Std Higher Width:").pack()
        std_h_width_entry = tk.Entry(postprocess_window, textvariable=default_std_h_width)
        std_h_width_entry.pack()

        tk.Label(postprocess_window, text="Optical Window:").pack()
        optical_window_entry = tk.Entry(postprocess_window, textvariable=default_optical_window)
        optical_window_entry.pack()

        # Store the results in a global dictionary
        global postprocess_results
        postprocess_results = {}

        # Function to handle compute button
        def compute():
            window_size = int(window_size_entry.get())
            r2_threshold = float(r2_threshold_entry.get())
            std_l_width = float(std_l_width_entry.get())
            std_h_width = float(std_h_width_entry.get())
            optical_window = int(optical_window_entry.get())

            valid_frames, valid_ranges, processed_df = determine_valid_data_from_df(
                metric_df, window_size, r2_threshold, std_l_width, std_h_width, optical_window)
            
            # Store the results
            postprocess_results["valid_frames"] = valid_frames
            postprocess_results["valid_ranges"] = valid_ranges
            postprocess_results["dataframe"] = processed_df
        
        def postprocess():
            global postprocess_results, video_path

            if postprocess_results and video_path:
                dataframe = postprocess_results["dataframe"]
                valid_frames = postprocess_results["valid_frames"]
                valid_ranges = postprocess_results["valid_ranges"]

                # Use the results to propagate lines using optical flow
                dataframe_optical = propagate_lanes_with_optical_flow(dataframe, valid_frames, valid_ranges, 150, video_path)
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
            else:
                messagebox.showwarning("No Data", "Please use compute first.")
        
        # Create the compute button
        compute_button = tk.Button(postprocess_window, text="Compute", command=compute)
        compute_button.pack(pady=20)
        # Create the Postprocess button
        postprocess_button = tk.Button(postprocess_window, text="Process Data", command=postprocess)
        postprocess_button.pack(pady=20)

        postprocess_window.mainloop()
        #messagebox.showinfo("Lane Postprocess", f"Video Name: {lane_detections_csv}")
    else:
        messagebox.showwarning("No Video", "Please select a video first.")

#############################################
#               OBJECT DETECTION            #
#############################################
def perform_object_detection():
    global video_path

    # Check if video_path is set
    if video_path:
        # Perform lane detection
        try:

            # Create a new window for displaying the frame
            display_window = tk.Toplevel(root)
            display_window.title("Object Detection")


            # Function to handle the accept button click
            def accept():
                display_window.destroy()
                object_detection_in_video(video_path)
                messagebox.showinfo("Object Detection", f"Object detection completed succesfully.")

            
            # Create an accept button
            accept_button = tk.Button(display_window, text="Accept", command=accept)
            accept_button.pack(pady=20)
            
            display_window.mainloop()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("No Video", "Please select a video first.")

def object_postprocess():
    global video_path

    # Check if video_path is set
    if video_path:
        try:
            key = None
            # Obtain the video name
            video_name = Path(video_path).stem
            data_path = check_data_folder(video_name)
            video_file, object_detections_csv = path_to_files_in_folder(data_path, file_names_to_check=["video_path.csv", "objects_detections_unprocessed.csv"])
            df = pd.read_csv(object_detections_csv)
            updated_df = postprocess_object_detection(df)
            frame_iterator = iter(display_bounding_boxes_new(updated_df, video_file))
            for frame, frame_number in frame_iterator:
                annotated_frame = frame.copy()
                annotated_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.imshow("Object Detection", annotated_frame)
                key = cv2.waitKey(33)
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_dataframe_to_csv(updated_df, video_file, "objects_detections_processed.csv")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("No Video", "Please select a video first.")

#############################################
#               DEPTH ESTIMATION            #
#############################################
def depth_estimation():
    global video_path

    # Check if video_path is set
    if video_path:
        try:
            # Obtain the video name
            video_name = Path(video_path).stem
            data_path = check_data_folder(video_name)
            video_file, object_detections_csv = path_to_files_in_folder(data_path, file_names_to_check=["video_path.csv", "objects_detections_processed.csv"])
            
            #Open a new window for furhter processing
            postprocess_window = tk.Toplevel(root)
            postprocess_window.title("Depth Estimation")

            # Define default values
            default_window_size = tk.StringVar(value="5")

            # Create entry widgets for parameters
            tk.Label(postprocess_window, text="Window Size:").pack()
            window_size_entry = tk.Entry(postprocess_window, textvariable=default_window_size)
            window_size_entry.pack()

            # Function to handle compute button
            def compute():
                window_size = int(window_size_entry.get())

                depth_estimation_with_Laplacian_generator(video_file, object_detections_csv, generator_set_size=window_size)

            # Create the compute button
            compute_button = tk.Button(postprocess_window, text="Compute", command=compute)
            compute_button.pack(pady=20)

            postprocess_window.mainloop()


        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("No Video", "Please select a video first.")  

#############################################
#               ZONE SEPARATION             #
#############################################
def zone_separation():
    global video_path

    # Check if video_path is set
    if video_path:
        # Perform lane detection
        try:
            # Obtain the video name
            video_name = Path(video_path).stem
            data_path = check_data_folder(video_name)
            video_file, object_detections_csv, lane_detection_csv = path_to_files_in_folder(data_path, file_names_to_check=["video_path.csv", "objects_detections_processed.csv", "lane_detection_processed.csv"])
            output_path = None
            overlay_detections_and_polygons_with_zones(video_file, object_detections_csv, lane_detection_csv, output_path)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("No Video", "Please select a video first.")

#############################################
#               MAIN APP                    #
#############################################

# Create the main window
root = tk.Tk()
root.title("System")

# Create and place the "Select Video" button
select_button = tk.Button(root, text="Select Video", command = select_video)
select_button.pack(pady=20)

# Create and place the "Perform Lane Detection" button
lane_detect_button = tk.Button(root, text="Perform Lane Detection", command = perform_lane_detection)
lane_detect_button.pack(pady=20)

# Create and place the "Lane Postprocess" button
lane_postprocess_button = tk.Button(root, text="Lane Postprocess", command = lane_postprocess)
lane_postprocess_button.pack(pady=20)

# Create and place the "Perform Object Detection" button
object_detect_button = tk.Button(root, text="Perform Object Detection", command = perform_object_detection)
object_detect_button.pack(pady=20)

# Create and place the "Perform Object Postprocess" button
object_postprocess_button = tk.Button(root, text="Object Postprocess", command = object_postprocess)
object_postprocess_button.pack(pady=20)

# Create and place the "Perform Depth Estimation" button
depth_estimation_button = tk.Button(root, text="Depth Estimation", command = depth_estimation)
depth_estimation_button.pack(pady=20)

# Create and place the "Perform Zone Separation" button
zone_separation_button = tk.Button(root, text="Zone Separation", command = zone_separation)
zone_separation_button.pack(pady=20)

# Run the application
root.mainloop()
