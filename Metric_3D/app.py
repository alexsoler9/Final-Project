import torch
import torch.nn.functional as F

import os
import os.path as osp
import time

#import cupy

import sys

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from mono.utils.logger import setup_logger
import glob

from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import transform_test_data_scalecano, get_prediction
from mono.utils.custom_data import load_from_annos, load_data

from mono.utils.avg_meter import MetricAverageMeter
from mono.utils.visualization import save_val_imgs, create_html, save_raw_imgs, save_normal_val_imgs
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image, ExifTags
import matplotlib.pyplot as plt

from mono.utils.unproj_pcd import reconstruct_pcd, save_point_cloud
from mono.utils.transform import gray_to_colormap
from mono.utils.visualization import vis_surface_normal
import plotly.graph_objects as go

from image_segmenter import ImageSegmenter
from ultralytics import YOLO, RTDETR
import supervision as sv

cfg_small = Config.fromfile("./mono/configs/HourglassDecoder/vit.raft5.small.py")
model_small = get_configured_monodepth_model(cfg_small, )
model_small, _, _, _ = load_ckpt("weight/metric_depth_vit_small_800k.pth", model_small, strict_match=False)
model_small.eval()

cfg_large = Config.fromfile('./mono/configs/HourglassDecoder/vit.raft5.large.py')
model_large = get_configured_monodepth_model(cfg_large, )
model_large, _,  _, _ = load_ckpt('weight/metric_depth_vit_large_800k.pth', model_large, strict_match=False)
model_large.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}.")
model_small.to(device)
model_large.to(device)

# Load YOLO model, from specific folder
#OD_model = YOLO("weight\yolov9c.pt")
OD_model = RTDETR("weight/rtdetr-l.pt")

def predict_depth_normal(img, model_selection="vit-small", fx=1000.0, fy=1000.0, state_cache={}):
    if model_selection == "vit-small":
        model = model_small
        cfg = cfg_small
    elif model_selection == "vit-large":
        model = model_large
        cfg = cfg_large
    else:
        return None, None, None, None, state_cache, "Not implemented model."
    
    if img is None:
        return None, None, None, None, state_cache, "Ensure the image is available."
    
    cv_image = np.array(img)
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    intrinsic = [fx, fy, img.shape[1]/2, img.shape[0]/2]
    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(img, intrinsic, cfg.data_basic)

    with torch.no_grad():
        pred_depth, pred_depth_scale, scale, output, confidence = get_prediction(
                            model           = model,
                            input           = rgb_input,
                            cam_model       = cam_models_stacks,
                            pad_info        = pad,
                            scale_info      = label_scale_factor,
                            gt_depth        = None,
                            normalize_scale = cfg.data_basic.depth_range[1],
                            ori_shape       = [img.shape[0], img.shape[1]],
                            )
        
        pred_normal = output["normal_out_list"][0][:, :3, :, :]
        H, W = pred_normal.shape[2:]
        pred_normal = pred_normal[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
    
    pred_depth = pred_depth.squeeze().cpu().numpy()
    pred_depth[pred_depth < 0] = 0 # Ensure all depth values are atleast 0
    pred_color = gray_to_colormap(pred_depth)

    pred_normal = torch.nn.functional.interpolate(pred_normal, [img.shape[0], img.shape[1]], mode="bilinear").squeeze()
    pred_normal = pred_normal.permute(1,2,0)
    pred_color_normal = vis_surface_normal(pred_normal)
    pred_normal = pred_normal.cpu().numpy()

    depth_file = ""
    normal_file = ""

    img = Image.fromarray(pred_color)
    img_normal = Image.fromarray(pred_color_normal)
    return img, pred_depth, img_normal, normal_file, state_cache,  "Success."

def get_camera(img):
    if img is None:
        return None, None, None, None, "Ensure the image is available."
    try:
        exif = img.getexif()
        exif.update(exif.get_ifd(ExifTags.IFD.Exif))
    except:
        exif = {}

    sensor_width = exif.get(ExifTags.Base.FocalPlaneYResolution, None)
    sensor_height = exif.get(ExifTags.Base.FocalPlaneXResolution, None)
    focal_length = exif.get(ExifTags.Base.FocalLength, None)

    # Convert sensor size to mm, see https://photo.stackexchange.com/questions/40865/how-can-i-get-the-image-sensor-dimensions-in-mm-to-get-circle-of-confusion-from
    w, h = img.size
    sensor_width = w / sensor_width * 25.4 if sensor_width is not None else None # 25.4 (mm/in)
    sensor_height = h / sensor_height * 25.4 if sensor_height is not None else None
    focal_length = focal_length * 1.0 if focal_length is not None else None

    message = "Obtained data from EXIF."
    if focal_length is None:
        message = "Focal length not found in EXIF. Manually input it."
    elif sensor_width is None or sensor_height is None:
        sensor_width = 16
        sensor_height = h / w * sensor_width
        message = f"Sensor size not found in EXIF. Using {sensor_width}x{sensor_height:.2f} mm as default."
    
    return sensor_width, sensor_height, focal_length, message

def get_inrinsic(img, sensor_width, sensor_height, focal_length):
    if img is None:
        return None, None, "Ensure the image is available."
    if sensor_width is None or sensor_height is None or focal_length is None:
        return 1000, 1000, "Insufficient information. Try providing camera info or use default 1000 for fx and fy."
    if sensor_width == 0 or sensor_height == 0 or focal_length == 0:
        return 1000, 1000, "Insufficient information. Try providing camera info or use default 1000 for fx and fy."
    
    # Compute focal lenght in pixels
    w, h = img.size
    fx = w / sensor_width * focal_length
    fy = h / sensor_height * focal_length

    return fx, fy, "Success!"



#####################################
#           AUXILIARY               #
#####################################

img_seg = ImageSegmenter(model_type = "yolov8s-seg")

def segmenter(image):
    image_segmentation, objects_data = img_seg.predict(image)
    return image_segmentation, objects_data

def get_masked_depth(depthmap, mask):
    masked_depth_map = depthmap*mask
    pixel_depth_vals = masked_depth_map[masked_depth_map>0]
    mean_depth = np.mean(pixel_depth_vals)
    return masked_depth_map, mean_depth

def draw_depth_video(image, depthmap, objects_data):
    image = image.copy()
    for data in objects_data:
        center = data[2]
        bottom_x, bottom_y = data[3]
        mask = data[4]
        
        depthmap_height, depthmap_width = depthmap.shape[:2]
        mask = cv2.resize(mask, (depthmap_width, depthmap_height), interpolation=cv2.INTER_NEAREST)
        
        _, depth = get_masked_depth(depthmap, mask)

        #KNOWN_DISTANCE = 3.53
        #POINT_X, POINT_Y = 539, 959
        scale=1#KNOWN_DISTANCE/depthmap[POINT_X][POINT_Y]

        #depth = depthmap[bottom_x][bottom_y]
        cv2.rectangle(image, (center[0]-15, center[1]-15), (center[0]+(len(str(round(depth*10, 2))+'m')*12), center[1]+15), data[5], -1)
        cv2.putText(image, str(round(depth*scale, 2))+'m', (center[0]-5, center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image

def run(model_selection="vit-small", fx=1000.0, fy=1000.0, state_cache={}, input_path="X:\Life\TFG\Coding\Testing\Videos\BikeBi\VID_20220426_161600.mp4"):
    cap = cv2.VideoCapture(input_path)
    # Check if it opened succesfully
    if not cap.isOpened():
        raise Exception("Error opening video file")

    if model_selection == "vit-small":
        model = model_small
        cfg = cfg_small
    elif model_selection == "vit-large":
        model = model_large
        cfg = cfg_large
    else:
        return None, None, None, None, state_cache, "Not implemented model."
    
    STRIDE = 5
    frame_index = 0
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Total frames:", total_frames)
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv_image = np.array(frame)
                img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                intrinsic = [fx, fy, img.shape[1]/2, img.shape[0]/2]
                rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(img, intrinsic, cfg.data_basic)
                inference_start_time = time.time()
                pred_depth, pred_depth_scale, scale, output, confidence = get_prediction(
                                    model           = model,
                                    input           = rgb_input,
                                    cam_model       = cam_models_stacks,
                                    pad_info        = pad,
                                    scale_info      = label_scale_factor,
                                    gt_depth        = None,
                                    normalize_scale = cfg.data_basic.depth_range[1],
                                    ori_shape       = [img.shape[0], img.shape[1]],
                                    )
                mean_confidence = confidence.mean()
                pred_depth = pred_depth.squeeze().cpu().numpy()
                pred_depth[pred_depth < 0] = 0 # Ensure all depth values are atleast 0
                pred_color = gray_to_colormap(pred_depth)
                pred_normal = output["normal_out_list"][0][:, :3, :, :]
                H, W = pred_normal.shape[2:]
                pred_normal = pred_normal[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
            

                pred_normal = torch.nn.functional.interpolate(pred_normal, [img.shape[0], img.shape[1]], mode="bilinear").squeeze()
                pred_normal = pred_normal.permute(1,2,0)
                pred_color_normal = vis_surface_normal(pred_normal)
                inference_end_time = time.time()
                fps = round(1/(inference_end_time - inference_start_time))
                ttf = inference_end_time - inference_start_time
                cv2.putText(pred_color, f"FPS: {fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 3)
                cv2.putText(pred_color, f"TTF: {round(ttf, 2)}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 3)
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                pred_color = cv2.resize(pred_color, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                pred_color_normal = cv2.resize(pred_color_normal, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                print(f"Frame shape: {frame.shape} \n Pred shape: {pred_color.shape}")
                combined_result = cv2.hconcat([frame,pred_color,pred_color_normal])
                cv2.imshow("ZoeDepth depth estimation", combined_result)

                frame_index += STRIDE
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                if cv2.waitKey(1) == ord("q"):
                    break
                # plt.matshow(pred_depth, cmap=plt.cm.jet, interpolation="bicubic")
                # plt.colorbar()
                # plt.title("Depth Map")
                # plt.xlabel("X-axis")
                # plt.ylabel("Y-axis")
                # plt.show()
            else:
                break
    cap.release()
    cv2.destroyAllWindows()

def test_system(img_path = "X:\Life\TFG\Coding\Testing\Images/Poliygon1.png"):
    img_path = img_path
    state_cache = {}
    
    # read image as PIL
    img_input = Image.open(img_path)
    # select model
    model = "vit-small"
    # Try to get camera parameters
    sensor_width, sensor_height, focal_len, message, = get_camera(img_input)
    print(message)
    # Computing camera intrinsics
    fx, fy, message = get_inrinsic(img_input, sensor_width, sensor_height, focal_len)
    print(message)
    depth_map_color, depth_file, normal_map_color, normal_map_file, state_cache, message = predict_depth_normal(img_input, model, fx, fy, state_cache)
    print(message)
    image_segmentation, data = segmenter(img_input)
    if len(data) > 0:
        plt.matshow(depth_file, cmap=plt.cm.jet, interpolation="bicubic")
        plt.colorbar()
        plt.title("Depth Map")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()
        image_depth = draw_depth(image_segmentation, depth_file, data)
        image_depth = cv2.cvtColor(image_depth, cv2.COLOR_BGR2RGB)
        image_depth = Image.fromarray(image_depth)
        image_depth.show()

    # show depth map ...
    depth_map_color.show()
    normal_map_color.show()

def draw_depth(image, depthmap, detections):
    image = image.copy()
    depth_list = []
    for idx,(xyxy, mask, confidence, class_id, trck_id, data) in enumerate(detections):
        x1, y1, x2, y2 = xyxy
        center = (x2 - x1)//2
        mask = draw_binary_mask(image.shape[:2], int(x1), int(x2), int(y2))
        _, depth = get_masked_depth(depthmap, mask)
        #depth = depthmap[int(y2)][int(center)]
        depth_list.append(depth)
    detections.data["depth"] = np.array(depth_list, dtype=np.float32)

    return detections

def draw_binary_mask(image_shape, x1, x2, y2, width=40):
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

def test_system_video(model_selection="vit-small", fx=1000.0, fy=1000.0, state_cache={}, input_path="X:\Life\TFG\Coding\Testing\Videos\BikeBi\VID_20220427_143651.mp4"):
    RECORD = False

    cap = cv2.VideoCapture(input_path)
    # Check if it opened succesfully
    if not cap.isOpened():
        raise Exception("Error opening video file")

    if model_selection == "vit-small":
        model = model_small
        cfg = cfg_small
    elif model_selection == "vit-large":
        model = model_large
        cfg = cfg_large
    else:
        return None, None, None, None, state_cache, "Not implemented model."
    
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_area = frame_height*frame_width
    ## Annotators
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(frame_width, frame_height))
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(frame_width, frame_height))
    # Initialize bbox annotator for drawing bounding boxes.
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    # Initialize label annotator for adding labels to objects
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)
    STRIDE = 1
    frame_index = 0
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if RECORD:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter('Depth_test.mp4', fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(frame_width), int(frame_height)))
    print("Total frames:", total_frames)
    selected_classes = [0]
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                annotated_frame = frame.copy()
                # Object detection
                # Pass frame to Object detector and obtain predictions
                result = OD_model.track(frame, persist=True , tracker="botsort.yaml", verbose=True)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = detections[np.isin(detections.class_id ,selected_classes)]
                #detections = detections[(detections.area / frame_area) > 0.005]
                if True: #detections.tracker_id is not None:
                
                    # Depth Estimation
                    # Compute Depth Estimation if there is a detection
                    if len(detections.class_id) > 0:
                        cv_image = np.array(frame)
                        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                        intrinsic = [fx, fy, img.shape[1]/2, img.shape[0]/2]
                        rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(img, intrinsic, cfg.data_basic)
                        pred_depth, pred_depth_scale, scale, output, confidence = get_prediction(
                                            model           = model,
                                            input           = rgb_input,
                                            cam_model       = cam_models_stacks,
                                            pad_info        = pad,
                                            scale_info      = label_scale_factor,
                                            gt_depth        = None,
                                            normalize_scale = cfg.data_basic.depth_range[1],
                                            ori_shape       = [img.shape[0], img.shape[1]],
                                            )
                        pred_depth = pred_depth.squeeze().cpu().numpy()
                        pred_depth[pred_depth < 0] = 0 # Ensure all depth values are atleast 0
                        pred_color = gray_to_colormap(pred_depth)
                        if pred_depth.shape[:2] != annotated_frame.shape[:2]:
                            pred_depth = cv2.resize(pred_depth, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                        
                        detections = draw_depth(annotated_frame, pred_depth, detections)
                        # Generate the labels
                        labels = [f"#{trck_id} {OD_model.names[class_id]}  {data['depth']:0.2f}m" 
                                for _, _, confidence, class_id, trck_id, data in detections]
                        
                        # Annotate the labels
                        annotated_frame = label_annotator.annotate(
                            scene=annotated_frame, detections=detections, labels=labels)
                        # Annotate the bounding box
                        annotated_frame = bounding_box_annotator.annotate(
                            scene=annotated_frame, detections=detections)
                #depthmap_height, depthmap_width = pred_depth.shape[:2]

                # inference_end_time = time.time()
                # fps = round(1/(inference_end_time - inference_start_time))
                # ttf = inference_end_time - inference_start_time
                # cv2.putText(pred_color, f"FPS: {fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 3)
                # cv2.putText(pred_color, f"TTF: {round(ttf, 2)}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 3)

                # Resize depth to original image
                # Pass information to function which
                ## 1- obtains the line corresponding to the bottom of the bounding box
                ## 2- Computes the depth using this bottom line
                ###Possibility of making a mask increasing the size to the bottom, ensuring that it resides inside the maximum height of the image. 
                # image_segmentation, data = segmenter(frame)
                

                annotated_frame = cv2.resize(annotated_frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
                if RECORD:
                    out.write(annotated_frame)
                cv2.imshow("ZoeDepth depth estimation", annotated_frame)
                if cv2.waitKey(1) == ord("q"):
                    break
                frame_index += STRIDE
                if STRIDE!= 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            else:
                break
    cap.release()
    cv2.destroyAllWindows()
    if RECORD: out.release()
    

if __name__ == "__main__":
    # input_path="X:\Life\TFG\Coding\Testing\Videos\CYCLING_IN_BARCELONA.mp4"
    test_system_video(model_selection="vit-small")
    #run(model_selection="vit-small", input_path="X:\Life\TFG\Coding\Testing\Videos\CYCLING_IN_BARCELONA.mp4")
    #run(model_selection="vit-large")