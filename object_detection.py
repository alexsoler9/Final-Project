import os
import sys
from tkinter.filedialog import askopenfilename
from pathlib import Path

# Add the "ObjectDetector" directory to sys.path
# main_path = Path(__file__).parent
# object_detector_path =  main_path / "ObjectDetector"
# sys.path.insert(0, str(object_detector_path))
#sys.path.insert(0, str(main_path)) 

from ObjectDetector.ObjectDetector import ObjectDetector

def object_detection_in_video(video_path, model_selection="yolov9e.pt"):
    model = ObjectDetector(model_selection)
    model.save_tracking(video_path)

if __name__ == "__main__":
    video_path = ""
    object_detection_in_video(video_path)