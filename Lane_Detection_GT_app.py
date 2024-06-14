import cv2
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from PIL import Image, ImageTk
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import ast

class LaneDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lane Detection Ground Truth Annotation")

        self.video_path = None
        self.cap = None 
        self.frame = None
        self.scaling_factor = 0.5

        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()
        
        self.open_button = tk.Button(self.button_frame, text="Open Video", command=self.open_video)
        self.open_button.pack(side=tk.LEFT)

        self.open_csv_button = tk.Button(self.button_frame, text="Open CSV", command=self.open_csv)
        self.open_csv_button.pack(side=tk.LEFT)

        self.left_line_button = tk.Button(self.button_frame, text="Left Line", command=self.select_left_line)
        self.left_line_button.pack(side=tk.LEFT)

        self.right_line_button = tk.Button(self.button_frame, text="Right Line", command=self.select_right_line)
        self.right_line_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(self.button_frame, text="Next Frame", command=self.next_frame)
        self.next_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(self.button_frame, text="Save Annotations", command=self.save_annotations)
        self.save_button.pack(side=tk.LEFT)

        self.annotations = []
        self.loaded_annotations = None

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Key>", self.next_frame_on_key)
        self.current_frame_number = 0
        self.lane_points = {"left": [], "right": []}
        self.selected_line = None  # Track the selected line ("left" or "right")

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not self.video_path:
            return
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.show_frame()

    def open_csv(self):
        csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            return
        self.loaded_annotations = pd.read_csv(csv_path)
        self.loaded_annotations["Ego Left Lane"] = self.loaded_annotations["Ego Left Lane"].apply(self.safe_eval)
        self.loaded_annotations["Ego Right Lane"] = self.loaded_annotations["Ego Right Lane"].apply(self.safe_eval)
        self.current_frame_number = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        self.show_frame()
    
    def safe_eval(self, x):
        try:
            return ast.literal_eval(x)
        except (SyntaxError, ValueError):
            return []

    def show_frame(self):
        if self.current_frame_number >= self.total_frames-1:
            return
        ret, self.frame = self.cap.read()
        if not ret:
            return
        
        self.frame = cv2.resize(self.frame, (0, 0), fx=self.scaling_factor, fy=self.scaling_factor)
        self.frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.img = ImageTk.PhotoImage(Image.fromarray(self.frame_rgb))

        self.canvas.config(width=self.img.width(), height=self.img.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

        if self.loaded_annotations is not None and self.current_frame_number < len(self.loaded_annotations):
            annotation = self.loaded_annotations.iloc[self.current_frame_number]
            self.lane_points = {
                "left": annotation["Ego Left Lane"],
                "right": annotation["Ego Right Lane"]
            }
            self.draw_points()

    def on_click(self, event):
        x, y = event.x, event.y
        if self.selected_line is None:
            return
        # Scale the point back to the original resolution
        x_original = int(x / self.scaling_factor)
        y_original = int(y / self.scaling_factor)
        self.lane_points[self.selected_line].append((x_original, y_original))
        self.draw_points()
    
    def draw_points(self):
        self.canvas.delete("point")
        self.canvas.delete("line")
        for side, points in self.lane_points.items():
            color = "red" if side == "left" else "blue"
            for point in points:
                x, y = point
                # Scale point to fit display
                x_scaled = int(x * self.scaling_factor)
                y_scaled = int(y * self.scaling_factor)
                self.canvas.create_oval(x_scaled-3, y_scaled-3, x_scaled+3, y_scaled+3, fill=color, tag="point")
            
            if len(points) >= 2:
                _, predicted_coords = line_fit(points, [1920, min(y for x,y in points)], compute_r2=False)
                self.lane_points[side] = predicted_coords
                for i in range(len(predicted_coords)-1):
                    x1, y1 = predicted_coords[i]
                    x2, y2 = predicted_coords[i+1]
                    x1_scaled = int(x1 * self.scaling_factor)
                    y1_scaled = int(y1 * self.scaling_factor)
                    x2_scaled = int(x2 * self.scaling_factor)
                    y2_scaled = int(y2 * self.scaling_factor)
                    self.canvas.create_line(x1_scaled, y1_scaled, x2_scaled, y2_scaled,
                                            fill=color, width=2, tag="line")

    def select_left_line(self):
        self.selected_line = "left"
        self.lane_points["left"] = []
        self.draw_points()

    def select_right_line(self):
        self.selected_line = "right"
        self.lane_points["right"] = []
        self.draw_points()
                
    def next_frame(self):
        if self.current_frame_number >= self.total_frames -1:
            return
        self.annotations.append({
            "Frame Number": self.current_frame_number,
            "Ego Left Lane": str(self.lane_points["left"]),
            "Ego Right Lane": str(self.lane_points["right"]),
        })
        self.lane_points = {"left": [], "right": []}
        self.current_frame_number += 1
        self.show_frame()
    
    def next_frame_on_key(self, event):
        if event.keysym == "space":
            self.next_frame()
        
    def write_annotations_to_csv(self, filename):
        if self.loaded_annotations is not None:
            # Merge loaded annotations and new annotations
            df_loaded = self.loaded_annotations.copy()
            df_new = pd.DataFrame(self.annotations)
            # Update the loaded annotations with new ones
            for idx, row in df_new.iterrows():
                try:
                    df_loaded.iloc[row["Frame Number"]] = row
                except:
                    print("You tried inserting more annotations than the presents in the original csv")
            df_loaded.to_csv(filename, index=False)
        else:
            df = pd.DataFrame(self.annotations)
            df.to_csv(filename, index=False)
        
    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()
        
    def save_annotations(self):
        if not self.annotations:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            self.write_annotations_to_csv(save_path)

def line_fit(coordinates, y_vals=[1300, 1000, 600], compute_r2=True):
    """
    Evaluates how well the point form a line

    Parameters:
        coordinates (list of tuples): List of (x,y) coordinate tuples.
        y_vals_to_predict (list of int): List of y_coordinates to predict x_coordinates
    
    Returns:
        r2 (float): Coefficient of determination indicating goodness of fit.
        predicted_coords (list of tuples): Predicted (x, y) coordinates for the given y-coordinates
    """
    if len(coordinates) < 2:
        return 0, np.nan # Not enough points to form a line
    
    # Separate the coordinates into X and y
    coordinates_array = np.array(coordinates, dtype=np.int32)
    X = coordinates_array[:,0].reshape(-1, 1)
    y = coordinates_array[:,1]

    # Perform linear regression
    model = LinearRegression().fit(X,y)

    if compute_r2:
        # Predict y values based on the model
        y_pred = model.predict(X)

        # Calculate the coefficient of determination (R^2)
        r2 = r2_score(y, y_pred)
    else:
        r2 = None

    # Perform linear regression (y -> X) for prediction
    min_value = np.min(y)
    y_vals_to_predict = y_vals.copy()
    #y_vals_to_predict.append(min_value)
    model_predict = LinearRegression().fit(y.reshape(-1, 1), X)
    x_vals_predicted = model_predict.predict(np.array(y_vals_to_predict).reshape(-1,1)).reshape(-1)

    predicted_coords = [(int(x), int(y)) for x, y in zip(x_vals_predicted, y_vals_to_predict)]

    return r2, predicted_coords

if __name__ == "__main__":
    root = tk.Tk()
    app = LaneDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()  