import cv2
import torch
import scipy.special
from scipy.spatial.distance import cdist
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import copy
import os
import csv

try:
    from LaneDetector.ufldDetector.utils.type import LaneModelType, lane_colors
    from LaneDetector.ufldDetector.core import LaneDetectBase
    from LaneDetector.ufldDetector.utils.common import merge_config, get_model
except:
    from .utils.type import LaneModelType, lane_colors
    from .core import LaneDetectBase
    from .utils.common import merge_config, get_model

class ModelConfig():
    
    def __init__(self, model_path, model_config, model_type):
        
        if model_type == LaneModelType.UFLDV2_TUSIMPLE:
            _, self.cfg = merge_config( [model_config, "--test_model", model_path])
        else:
            NotImplemented
    
    



class UltrafastLaneDetectorV2(LaneDetectBase):
    _defaults = {
        "model_path"    : "models/tusimple_res18.pth",
        "model_config"  : "configs/tusimple_res18.py",
        "model_type"    : LaneModelType.UFLDV2_TUSIMPLE
    }

    def __init__(self, model_path : str = None, model_config : str = None, model_type : LaneModelType = None, 
                 save_coords : bool = False, video_name : str = None):
        LaneDetectBase.__init__(self)
        if (None not in [model_path, model_config, model_type]):
            self.model_path, self.model_config, self.model_type = model_path, model_config, model_type

        # Load model configuration based on the model type
        if (self.model_type not in [LaneModelType.UFLDV2_TUSIMPLE]):
            raise Exception("UltrafastLaneDetectorV2 can't use %s type." % self.model_type.name)
        
        self.cfg = ModelConfig(self.model_path, self.model_config, self.model_type).cfg
        self.cfg = self.cfg._cfg_dict
        self.batch_size = 1
        assert self.cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']
        if self.cfg.dataset == "Tusimple":
            self.cls_num_per_lane = 56
        elif self.cfg.dataset == "CULane":
            self.cls_num_per_lane = 18
        else:
            raise NotImplementedError
        
        self.save_coords = save_coords
        self.frame_number = 0
        if video_name is not None:
            self.video_name = os.path.splitext(video_name)[0]
        # Initialize model
        self.model = self._initialize_model()

        # Initialize image transformation
        self.img_transform = self._initialize_image_transform()

    def _initialize_model(self):
        cfg = self.cfg

        # Load model 
        net = get_model(cfg)

        # Laod the weigths from the downloaded model
        try:
            if torch.backends.mps.is_built():
                net = net.to("mps")
                state_dict = torch.load(cfg.test_model, map_location='mps')['model'] # Apple GPU
            else:
                net = net.cuda()
                state_dict = torch.load(cfg.test_model, map_location='cuda')['model'] # CUDA
		
        except:
            state_dict = torch.load(cfg.test_model, map_location='cpu')['model'] # CPU
        
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        # Load the weights into the model
        net.load_state_dict(compatible_state_dict, strict=False)
        net.eval()

        return net
    
    def _initialize_image_transform(self):
        # Create transfom operation to resize and normalize the input images
        img_transforms = transforms.Compose([
            transforms.Resize((int(self.cfg.train_height / self.cfg.crop_ratio), self.cfg.train_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return img_transforms
    
    def detect_lanes(self, image):

        input_tensor = self.prepare_input(image)

        #Perform inference on the image
        output = self.inference(input_tensor)
        
        # Process the output data
        self.coords = self.pred2coords(output)
        #self.lane_info.lanes_points, self.lane_info.lanes_status = self._process_output(output)

        visualization_img = self.draw_coords(image, self.coords, self.cfg)
        #visualization_img = self.draw(image)

        return visualization_img, self.coords
    
    def prepare_input(self, image):
        # Transform the image for inference
        self.img_h, self.img_w = image.shape[0], image.shape[1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)
        input_img = self.img_transform(img_pil)
        input_img = input_img[:, -self.cfg.train_height:, :]
        input_tensor = input_img[None, ...]

        if not torch.backends.mps.is_built():
            input_tensor = input_tensor.cuda()

        return input_tensor
    
    def inference(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)

        return output
    
    def pred2coords(self, pred, local_width=1):
        exist_row_prob = pred['exist_row'].softmax(1).cpu()
        exist_col_prob = pred['exist_col'].softmax(1).cpu()
        #print(exist_row_prob[0,0,:,:])
        row_anchor = self.cfg.row_anchor
        col_anchor = self.cfg.col_anchor

        original_image_height = self.img_h
        original_image_width = self.img_w

        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

        max_indices_row = pred['loc_row'].argmax(1).cpu()
        # n , num_cls, num_lanes
        valid_row = pred['exist_row'].argmax(1).cpu()
        # n, num_cls, num_lanes

        max_indices_col = pred['loc_col'].argmax(1).cpu()
        # n , num_cls, num_lanes
        valid_col = pred['exist_col'].argmax(1).cpu()
        # n, num_cls, num_lanes

        pred['loc_row'] = pred['loc_row'].cpu()
        pred['loc_col'] = pred['loc_col'].cpu()

        coords = []

        row_lane_idx = [0,1,2,3]
        col_lane_idx = [0,1,2,3]
        #print()
        for i in row_lane_idx:
            tmp = []
            if valid_row[0,:,i].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0,k,i]:
                        all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))

                        out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                        #print(i, int(row_anchor[k] * original_image_height),exist_row_prob[0,0,k,i],exist_row_prob[0,1,k,i])
                        if exist_row_prob[0,1,k,i] > 0.95:
                            #print(i, int(row_anchor[k] * original_image_height),exist_row_prob[0,0,k,i], exist_row_prob[0,1,k,i], count)
                            tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
                #print()
            coords.append(tmp)

        for i in col_lane_idx:
            tmp = []
            if valid_col[0,:,i].sum() > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0,k,i]:
                        all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                        
                        out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5

                        out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                        if exist_col_prob[0,1,k,i] > 0.95:
                            tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)
        return coords

    def _process_output(self, output, local_width :int = 1) -> Tuple[np.ndarray, list]:
        original_image_width = self.img_w
        original_image_height = self.img_h

        batch_size, num_grid_row, num_cls_row, num_lane_row = output['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = output['loc_col'].shape

        exist_row_prob = output['exist_row'].softmax(1).cpu()
        exist_col_prob = output['exist_col'].softmax(1).cpu()

        # row_anchor = self.cfg.row_anchor
        # col_anchor = self.cfg.col_anchor

        max_indices_row = output['loc_row'].argmax(1).cpu()
        # n , num_cls, num_lanes
        valid_row = output['exist_row'].argmax(1).cpu()
        # n, num_cls, num_lanes

        max_indices_col = output['loc_col'].argmax(1).cpu()
        # n , num_cls, num_lanes
        valid_col = output['exist_col'].argmax(1).cpu()
        # n, num_cls, num_lanes

        output['loc_row'] = output['loc_row'].cpu()
        output['loc_col'] = output['loc_col'].cpu()
        row_lane_idx = [1,2]
        col_lane_idx = [1,2]

        # Parse the output of the model
        lanes_points = {"left-side" : [], "left-ego" : [] , "right-ego" : [], "right-side" : []}
        # lanes_detected = []
        lanes_detected =  {"left-side" : False, "left-ego" : False , "right-ego" : False, "right-side" : False}
        for i in row_lane_idx:
            tmp = []
            if valid_row[0,:,i].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0,k,i]:
                        all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                        out_tmp = (output['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                        if exist_row_prob[0,1,k,i] > 0.95:
                            tmp.append((int(out_tmp), int(self.cfg.row_anchor[k] * original_image_height)))
                if (i == 1) :
                    lanes_points["left-ego"].extend(tmp)
                    if (len(tmp) > 2) :
                        lanes_detected["left-ego"] = True
                else :
                    lanes_points["right-ego"].extend(tmp)
                    if (len(tmp) > 2) :
                        lanes_detected["right-ego"] = True

        for i in col_lane_idx:
            tmp = []
            if valid_col[0,:,i].sum() > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0,k,i]:
                        all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                        out_tmp = (output['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                        if exist_col_prob[0,1,k,i] > 0.95:
                            tmp.append((int(self.cfg.col_anchor[k] * original_image_width), int(out_tmp)))
                if (i == 1) :
                    lanes_points["left-side" ].extend(tmp)
                    if (len(tmp) > 2) :
                        lanes_detected["left-side"] = True
                else :
                    lanes_points["right-side"].extend(tmp)
                    if (len(tmp) > 2) :
                        lanes_detected["right-side"] = True

        return np.array(list(lanes_points.values()), dtype="object"), list(lanes_detected.values())
    
    def draw_coords(self, image, coords, cfg):
        im0 = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)

        for lane_num, lane_coords in enumerate(coords):
            
            #print(lane_num, len(lane_coords))    
            for coord in lane_coords:
                #cv2.circle(im0, coord, 5, lane_colors[lane_num], -1)
                font = cv2.FONT_HERSHEY_SIMPLEX 
                cv2.putText(im0, f"{lane_num}", coord, font, 0.5, lane_colors[lane_num], 2)
        if self.save_coords:
            self.save_lane_lines_csv(coords)
        return im0
    
    def save_lane_lines_csv(self, coords):
        csv_directory = "data/lane_detection_csv"
        self.frame_number += 1
        csv_file_name = f"ld_{self.video_name}.csv"
        csv_file_path = os.path.join(csv_directory, csv_file_name)

        #Ensure the directory exists; creates if it doesn't
        if not os.path.exists(csv_directory): 
            os.makedirs(csv_directory)

        write_mode = "a"
        if self.frame_number ==1:
            write_mode = "w" # Overwrite mode  

        with open(csv_file_path, write_mode, newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            if write_mode == "w":
                csv_writer.writerow(["Frame Number", "Lane Number", "Starting Point","Number of Points", "Point Coordinates"])
            for lane_num, lane_coords in enumerate(coords):
                if lane_num in [0, 1]:
                    # Standardize lane_coords to a fixed length with zero-padding
                    standardized_coords = self.standardize_coords(lane_coords, 50)
                    # Convert line coords to a single string
                    points_str = ';'.join([f'{x},{y}' for x, y in standardized_coords])
                    csv_writer.writerow([self.frame_number, lane_num, 0,len(lane_coords), points_str])
    
    def standardize_coords(self, lane_coords, max_length = 50):
        # Convert lane_coords list to a numpy array and pad/truncate as needed
        coords_array = np.zeros((max_length, 2))
        num_coords = min(len(lane_coords), max_length)
        if num_coords > 1:
            # Update the coords_aray with the values from lane_coords
            coords_array[:num_coords, :] = lane_coords[:num_coords]
        return coords_array.tolist()
    
    def draw(self, image):
        im0 = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)

        for lane_num, lane_coords in enumerate(self.lane_info.lanes_points):
            for coord in lane_coords:
                cv2.circle(im0, coord, 5, lane_colors[lane_num], -1)

        if self.save_coords:
            self.save_csv()
        
        return im0
    
    def save_csv(self):
        csv_directory = "lane_detection_csv"
        csv_file_name = f"ld_{self.video_name}.csv"
        csv_file_path = os.path.join(csv_directory, csv_file_name)
        self.frame_number += 1

        # Ensure the directory exists; create if it doesn't
        if not os.path.exists(csv_directory):
            os.makedirs(csv_directory)

        write_mode = "a"
        if self.frame_number == 1:
            write_mode = "w"
        
        with open(csv_file_path, write_mode, newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            if write_mode == "w":
                 # Write header row
                header_row = ["Frame Number"]
                for lane_num in range(len(self.lane_info._lanes_points)):
                    header_row.extend([f"Lane Number {lane_num}", "Number of Points", "Point Coordinates"])
                csv_writer.writerow(header_row)

            frame_data = [self.frame_number]
            for lane_num, lane_coords in enumerate(self.lane_info.lanes_points):
                # Standardize lane_coords to a fixed length with zero padding
                standardized_coords = self.standardize_coords(lane_coords, 50)
                # Convert line coords to a single string
                points_str = ';'.join([f'{x},{y}' for x, y in standardized_coords])
                # Append lane information to the frame_data list
                frame_data.extend([lane_num, len(lane_coords), points_str])

            csv_writer.writerow(frame_data)
        pass

    #############################
    #       Process CSV         #
    #############################

    def process_lane_data(self, frame_shape):
        # Load CSV data into a DataFrame
        csv_data_path = "data\lane_detection_csv\ld_"+self.video_name+".csv"
        print(f"csv_path: {csv_data_path}")
        data = pd.read_csv(csv_data_path, delimiter=";")
        PROCESS = True
        num_frames = int(data["Frame Number"].iloc[-1])

        if "Polygon Points" in data.columns:
            #Get the number of elements in the column
            num_elements = data["Polygon Points"].count()
            frame_count = data["Frame Number"].count()

            if num_elements == frame_count:
                PROCESS = False
                return lane_masks
        if PROCESS:
            left_lane_matrix, righ_lane_matrix = self.preprocess_lane_data(data, frame_shape, num_frames)

            left_lane_interpolated = self.interpolate_lane_data(left_lane_matrix, 7)
            righ_lane_intepolated = self.interpolate_lane_data(righ_lane_matrix, 7)
            lane_masks, polygon_pts = self.generate_lane_masks(left_lane_interpolated, righ_lane_intepolated, frame_shape)

            self.update_csv_polygon_points(csv_data_path, polygon_pts)
            print("Lane data processing completed.")
        else:
            lane_masks = self.create_masks_from_csv(data, frame_shape)
        return lane_masks

    def preprocess_lane_data(self, data, frame_shape, num_frames, Lane_Number = [0, 1]):
        LEFT_NUMBER = Lane_Number[0]
        RIGTH_NUMBER = Lane_Number[-1]
        # Initialize matrices to store lane points 
        x,y = 0,0
        left_lane_matrix = [[(x,y), (x,y)] for _ in range(num_frames)]
        right_lane_matrix = [[(x,y), (x,y)] for _ in range(num_frames)]

        # Iterate over each frame
        for frame in range (1, num_frames+1):
            # Filter data for current frame
            lane_frame = data[data["Frame Number"] == frame]

            # Process left lane data 
            left_lane_data = lane_frame[lane_frame["Lane Number"] == LEFT_NUMBER]
            if not left_lane_data.empty:
                # Parse and refine point coordinates
                left_lane_points = self.refine_lane_points(left_lane_data.iloc[0]["Starting Point"],
                                                        left_lane_data.iloc[0]["Number of Points"],
                                                        left_lane_data.iloc[0]["Point Coordinates"])
                
                # Fit a linear polynomial to left lane_points
                if len(left_lane_points) > 5:
                    x_left, y_left = zip(*left_lane_points)
                    left_lane_coeffs = np.polyfit(y_left, x_left, deg = 1)

                    # Update left lane matrix with the linear fit
                    poly_left = np.poly1d(left_lane_coeffs)
                    point = max(min(y_left)*0.8, 400)
                    
                    left_lane_matrix[frame - 1] = [ (int(poly_left(point)), int(point)), 
                                                    (int(poly_left(frame_shape[0])), frame_shape[0])]

            # Process right lane data 
            right_lane_data = lane_frame[lane_frame["Lane Number"] == RIGTH_NUMBER]
            if not right_lane_data.empty:
                # Parse and refine point coordinates
                right_lane_points = self.refine_lane_points(right_lane_data.iloc[0]["Starting Point"],
                                                    right_lane_data.iloc[0]["Number of Points"],
                                                    right_lane_data.iloc[0]["Point Coordinates"])
                # Fit a linear polynomial to left lane_points
                if len(right_lane_points) > 5:
                    x_right, y_right = zip(*right_lane_points)
                    right_lane_coeffs = np.polyfit(y_right, x_right, deg=1)
                    
                    # # Update right lane matrix with the linear fit
                    poly_right = np.poly1d(right_lane_coeffs)
                    point = max(min(y_right)*0.8, 400)

                    right_lane_matrix[frame - 1] = [(int(poly_right(point)), int(point)), 
                                                    (int(poly_right(frame_shape[0])), frame_shape[0])]
            # Check for intersection of left and right lanes
            if  (( left_lane_points) and ( right_lane_points)):
                # Calculate intersection y-coordinate (if any)
                if left_lane_coeffs[0] != right_lane_coeffs[0]:
                    x_intersect = (right_lane_coeffs[1] - left_lane_coeffs[1]) / (left_lane_coeffs[0] - right_lane_coeffs[0])
                    # Check if intersect is whitin width
                    poly = np.poly1d(left_lane_coeffs)
                    y_intersect = max(poly(x_intersect), min(y_left)*0.9, min(y_right)*0.9)
                    intersection_y = int(min(max(y_intersect, 0), frame_shape[0]))
                    
                    # Update both matrices with the intersection point
                    left_lane_matrix[frame - 1][0] = (int(poly_left(intersection_y)), intersection_y)
                    left_lane_matrix[frame - 1][1] = (int(poly_left(frame_shape[0])), frame_shape[0])
                    right_lane_matrix[frame - 1][0] = (int(poly_right(intersection_y)), intersection_y)
                    right_lane_matrix[frame - 1][1] = (int(poly_right(frame_shape[0])), frame_shape[0])

        return left_lane_matrix, right_lane_matrix

    def refine_lane_points(self, start_index, num_points, point_coords):
        # Parse point coordinates from string fromat and refine them
        all_points = self.parse_lane_points(point_coords)

        # Extracts subset of points
        refined_points = all_points[start_index:start_index+num_points]

        return refined_points

    def parse_lane_points(self, point_coords):
        # Parse point coordinates from string format
        lane_points = []
        for point in point_coords.split(";"):
            x, y = map(float, point.split(","))
            lane_points.append((x, y))

        return lane_points
        
    def interpolate_lane_data(self, lane_matrix, window_size = 7):
        # Perform interpolation between frame_window for no current frame lane.
        num_frames = len(lane_matrix)

        interpolated_matrix = copy.deepcopy(lane_matrix)

        for frame in range(num_frames):
            for col in range(len(lane_matrix[0])):
                # Determine the window boundaries around the current frame
                start_idx = max(0, frame-window_size)
                end_idx = min(num_frames, frame + window_size + 1)

                # Extract lane data within the window
                window_data = [lane_matrix[row][col] for row in range(start_idx, end_idx)]

                valid_indices = np.arange(len(window_data))
                
                # Filter out invalid (0,0) data points
                valid_data = [data for data in window_data if data != (0,0)]
                
                if len(valid_data) > 1:
                    # Prepare x and y values
                    x_values = np.arange(len(valid_data))
                    y_values = list(zip(*valid_data))

                    if len(valid_indices) > 1:
                        # Perform linear interpolation
                        interp_func_x = interp1d(x_values, y_values[0], kind="linear", fill_value="extrapolate")
                        interp_func_y = interp1d(x_values, y_values[1], kind="linear", fill_value="extrapolate")

                        # Calculate interpolated values for the current frame
                        interpolated_x = interp_func_x(len(valid_data)//2)  # Interpolate at the center of the window
                        interpolated_y = interp_func_y(len(valid_data)//2)  # Interpolate at the center of the window


                        # Update if curretn value is invalid
                        current_x, current_y = lane_matrix[frame][col]
                        if current_x == 0 and current_y == 0:
                            average_x = float(interpolated_x)
                            average_y = float(interpolated_y)
                            valid_data.append((average_x, average_y))
                        average_x = np.mean([data[0] for data in valid_data])
                        average_y = np.mean([data[1] for data in valid_data])
                        interpolated_matrix[frame][col] = (average_x, average_y)
        return interpolated_matrix
    
    def generate_lane_masks(self, left_lane_points, right_lane_points, frame_shape):
        lane_masks = []
        polygon_pts = []
        for left_points, right_points in zip(left_lane_points, right_lane_points):
            polygon, lane_mask = self.create_lane_mask(left_points, right_points, frame_shape)
            polygon_pts.append(polygon)
            polygon_pts.append(polygon) # To Be Properly Done 
            lane_masks.append(lane_mask)
        
        return lane_masks, polygon_pts
    
    def create_lane_mask(self, left_points, right_points, frame_shape):
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)

        lane_polygon_pts = []
        target_value = [(0,0), (0,0)]
        if target_value not in [left_points, right_points]:
            lane_polygon_pts.extend(reversed(left_points))
            lane_polygon_pts.extend(right_points)
            lane_polygon_pts = np.array(lane_polygon_pts, dtype=np.int32)
        else:
            left_points = [(0, frame_shape[0]), (0, frame_shape[0])]
            right_points = [(frame_shape[1], frame_shape[0]), (frame_shape[1], frame_shape[0])]
            lane_polygon_pts.extend(reversed(left_points))
            lane_polygon_pts.extend(right_points)
            lane_polygon_pts = np.array(lane_polygon_pts, dtype=np.int32)    
        cv2.fillPoly(mask, [lane_polygon_pts], color=(255))
        return lane_polygon_pts, mask

    def update_csv_polygon_points(self, csv_data_path, polygon_pts):
        # Convert each polygon to string representation
        polygon_strings = [self.polygon_to_string(polygon) for polygon in polygon_pts]
        # Read existing CSV file into a DataFrame
        df = pd.read_csv(csv_data_path, delimiter=";")

        # Add "Polygon Points column to the DataFrame"
        df["Polygon Points"] = polygon_strings

        # Save the updated DataFrame back to the CSV file
        df.to_csv(csv_data_path, sep=";",index=False)

    def polygon_to_string(self, polygon):

        # Convert polygon numpy array to string representation
        polygon_str = "[" + "".join(map(str, polygon.tolist())) + "]"
        return polygon_str
    
    def create_masks_from_csv(self, data, frame_shape):
        lane_masks = []

        unique_frames = data["Frame Number"].unique()
        for frame_num in unique_frames:
            # Filter data for the current frame number
            frame_data = data[data["Frame Number"] == frame_num]

 
            for points_str in frame_data["Polygon Points"]:
                
                polygon_pts = self.parse_polygon_points(points_str)
            
            # Create a mask for the current frame
            mask = np.zeros(frame_shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, np.array([polygon_pts], dtype=np.int32), color=(255))

            lane_masks.append(mask)

        return lane_masks
    
    def parse_polygon_points(self, points_str):
        clean_str = points_str.strip("'[]")

        coordinate_strs = clean_str.split('][')
        polygon_pts = []
        for coord_str in coordinate_strs:
            x, y = map(int, coord_str.split(","))
            polygon_pts.append((x,y))
        return polygon_pts
