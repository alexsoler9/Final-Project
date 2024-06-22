## Installation

Follow these steps to run the project on your local machine.

### 1. Clone the repository

First, clone the repository to your local machine using the following command.

```bash
git clone https://github.com/alexsoler9/Final-Project.git
```

### 2. Navigate to the project directory
Change directory to the new cloned repository:

```bash
cd Final-Project
```

### 3. (Optional) Create a virtual environment.

It is recommended to create a virtual environment to manage dependencies.

Create the virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

For Windows:

```bash
.\venv\Scripts\activate
```

For macOS and Linux

```bash
source venv/bin/activate
```

### 4. Install PyTorch and torchvision

Install PyTorch 2.0.1 and torchvision 0.15.2 with CUDA 11.8 support:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### 5. Install remaining dependencies

Install the remaining required dependencies:

```bash
pip install -r requirements.txt
```

### 6. Download required models

The following models need to be downloaded from the respective Github repositories:

- From UFLDv2: Download [tu_simple_res18.pth](https://drive.google.com/file/d/1Clnj9-dLz81S3wXiYtlkc4HVusCb978t/view?usp=sharing)
- From Metric3D: Download [metric_depth_vit_small_800k.pth](https://drive.google.com/file/d/1YfmvXwpWmhLg3jSxnhT7LvY0yawlXcr_/view?usp=drive_link)

Put **tu_simple_res18.pth** inside "Final-Project/LaneDetector/models".

Put **metric_depth_vit_small_800k.pth** inside "Final-Project/Metric_3D/weight".

**Note:**
Sometimes when using the object detector for the first time, the following error can appear: Ultralytics requirement ['dill'] not found. Usually, it will auto-update itself. After that, you just need to restart the runtime.

## Utility App Usage

To simplify usage, a utiity app has been created. Follow these steps:

1. Ensure you are in the project directory.
2. Execute **app.py** to launch the app. A window should pop up.
3. Press **Select Video** to choose a video file.
4. Once a video is selected, you can choose to either:
   - Perform Lane Detection
   - Perform Object Detection 
5. After performning Lane Detection, you can use the **Lane Postprocess** function.
6. After performing Object detection, you can use the **Object Postrprocess** function.
7. (Optional) Once **Object Postprocess** is complete, you can execute **Depth Estimation**
8. After both **Lane Postprocess** and **Object Postprocess** are complete, you can use **Zone Separation**.