YOLOv5 Object Detection and Tracking
This repository contains a Python script that sets up YOLOv5 for object detection and tracking using a live video stream. The script utilizes the OpenCV library for video processing, the PyTorch library for the YOLOv5 model, and the Sort algorithm for object tracking.

Requirements
Python 3.6 or higher
OpenCV
PyTorch
YOLOv5
NumPy
SciPy
torchvision
scikit-image
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Set up a virtual environment (optional, but recommended):
bash
Copy code
python -m venv venv
source venv/bin/activate  # For Linux and macOS
.\venv\Scripts\activate  # For Windows
Install the required packages:
Copy code
pip install -r requirements.txt
Download the YOLOv5 model weights:
bash
Copy code
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt
Note: If you don't have wget installed, you can download the weights file manually from here and place it in the root directory of the repository.

Usage
Run the main script:
css
Copy code
python main.py
The script will display a window with the live video stream. Detected objects will be highlighted with bounding boxes and assigned tracking IDs.

To exit the script, press q.

Customizing the Script
You can customize the script by modifying the following variables in main.py:

stream_url: Set this variable to the URL of your desired video stream.
img_size: Adjust the input image size for the YOLOv5 model. A larger size may improve detection accuracy but will also increase processing time.
conf_thres: Adjust the confidence threshold for object detection. Higher values will result in fewer detected objects, but may reduce false positives.
iou_thres: Adjust the Intersection over Union (IoU) threshold for non-maximum suppression. Higher values will allow more overlapping detections, while lower values will be more strict.
References
YOLOv5: https://github.com/ultralytics/yolov5
Sort: https://github.com/abewley/sort
OpenCV: https://opencv.org/
PyTorch: https://pytorch.org/
SciPy: https://www.scipy.org/
scikit-image: https://scikit-image.org/
torchvision: https://pytorch.org/vision/stable/index.html
License
This project is licensed under the MIT License. See the LICENSE file for details.