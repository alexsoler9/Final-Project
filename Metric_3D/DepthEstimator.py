import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch
import torch.nn.functional as F

from pathlib import Path
import sys

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

# Add the "Metric3D" directory to sys.path
main_path = Path(__file__).parent
depthEstimator_path =  main_path 
sys.path.insert(0, str(depthEstimator_path))

from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import transform_test_data_scalecano, get_prediction
from mono.utils.transform import gray_to_colormap
from mono.utils.visualization import vis_surface_normal

project_path = main_path.parent
sys.path.insert(0, str(project_path))

class DepthEstimator:
    def __init__(self, model_selection="vit-small", device=None):
        if model_selection == "vit_small":
            self.cfg = Config.fromfile(project_path / "Metric_3D/mono/configs/HourglassDecoder/vit.raft5.small.py")
            self.model = get_configured_monodepth_model(self.cfg)
            self.model, _, _, _ = load_ckpt(project_path / "Metric_3D/weight/metric_depth_vit_small_800k.pth", self.model, strict_match=False)
        elif model_selection == "vit_large":
            self.cfg = Config.fromfile(project_path / 'Metric_3D/mono/configs/HourglassDecoder/vit.raft5.large.py')
            self.model = get_configured_monodepth_model(self.cfg)
            self.model, _, _, _ = load_ckpt(project_path / "Metric_3D/weight/metric_depth_vit_large_800k.pth", self.model, strict_match=False)
        else:
            raise NotImplementedError("Model not implemented")

        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        print(f"Running on {self.device}.")

    def draw_binary_mask(self, image_shape, x1, y1, x2, y2, width=40):
        # Initialize a binary mask with zeros, matching the size of the input image
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Calculate the bottom y-coordinate of the rectangle (y2+5)
        bottom_y = y2 + width//2
        
        # Ensure the coordinates are within the bounds of the mask
        y2 = min(max(y2-width//2, 0), image_shape[0] - 1)
        x1 = min(max(x1, 0), image_shape[1] - 1)
        x2 = min(max(x2, 0), image_shape[1] - 1)
        bottom_y = min(max(bottom_y, 0), image_shape[0] - 1)
        
        # Draw the rectangle in the mask
        mask[y2:bottom_y+1, x1:x2+1] = 1  # Set the row 'bottom_y' from column 'x1' to 'x2' (inclusive) to 1
        
        return mask
    
    def get_masked_depth(self, depthmap, mask):
        masked_depth_map = depthmap * mask
        pixel_depth_vals = masked_depth_map[masked_depth_map>0]
        mean_depth = np.mean(pixel_depth_vals)
        return masked_depth_map, mean_depth
    
    def draw_depth(self, image, depthmap, detections):
        image = image.copy()
        depth_list = []
        for idx,(xyxy, mask, confidence, class_id, trck_id, data) in enumerate(detections):
            x1, y1, x2, y2 = xyxy
            center = (x2 - x1)//2
            mask = self.draw_binary_mask(image.shape[:2], int(x1), int(y1), int(x2), int(y2))
            _, depth = self.get_masked_depth(depthmap, mask)
            #depth = depthmap[int(y2)][int(center)]
            depth_list.append(depth)
        detections.data["depth"] = np.array(depth_list, dtype=np.float32)

        return detections
    
    def prediction(self, frame, fx, fy,):
        cv_image = np.array(frame)
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        intrinsic = [fx, fy, img.shape[1]/2, img.shape[0]/2]
        rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(img, intrinsic, self.cfg.data_basic)
        pred_depth, pred_depth_scale, scale, output, confidence = get_prediction(
                            model           = self.model,
                            input           = rgb_input,
                            cam_model       = cam_models_stacks,
                            pad_info        = pad,
                            scale_info      = label_scale_factor,
                            gt_depth        = None,
                            normalize_scale = self.cfg.data_basic.depth_range[1],
                            ori_shape       = [img.shape[0], img.shape[1]],
                            )
        pred_depth = pred_depth.squeeze().cpu().numpy()
        pred_depth[pred_depth < 0] = 0 # Ensure all depth values are atleast 0
        pred_color = gray_to_colormap(pred_depth)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        if pred_depth.shape[:2] != frame.shape[:2]:
            pred_depth = cv2.resize(pred_depth, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        
        return pred_depth, pred_color