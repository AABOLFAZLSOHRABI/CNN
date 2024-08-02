# -*- coding: utf-8 -*-

!nvidia-smi

"""# Download Dataset"""

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="your api key")
project = rf.workspace("cigarette-detector").project("cigarettes-reality-2")
version = project.version(21)
dataset = version.download("yolov8")

"""# Train Model"""

!pip install ultralytics

from ultralytics import YOLO

# Load a model
model = YOLO('yolov10n.pt')

# Define training arguments including early stopping
train_args = {
    'data': '/content/Cigarettes-reality-2-21/data.yaml',
    'epochs': 100,
    'imgsz': 640,
    'patience': 5  # This will enable early stopping with patience of 5 epochs
}

# Train the model
results = model.train(**train_args)

"""# Test Model"""

# UTF-8 to locale
import locale
locale.getpreferredencoding = lambda: 'UTF-8'

!pip install -q supervision roboflow

HOME = os.getcwd()
print(HOME)

import os

# ایجاد پوشه برای تصاویر
image_folder = os.path.join(HOME, 'images')
os.makedirs(image_folder, exist_ok=True)

# لیست لینک‌های تصاویر
image_urls = [
    "https://media.post.rvohealth.io/wp-content/uploads/2020/08/15_Tips_for_Quitting_Smoking-732x549-thumbnail-1-732x549.jpg",
    "https://www.verywellhealth.com/thmb/O_1iqLRSMpdMlaKd7DIlHfxWqfQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-183029444-570e9f433df78c7d9e56800a.jpg",
    "https://assets.weforum.org/article/image/yfrFLZ-XkAWKPlhWCBnkCOAzMw9BPfPQ29JLfrCZyUQ.jpg"
]

# دانلود تصاویر
for idx, url in enumerate(image_urls):
    image_path = os.path.join(image_folder, f'image_{idx}.jpg')
    !wget -O {image_path} {url}

# انجام پیش‌بینی با مدل سفارشی بر روی تمامی تصاویر در پوشه
!yolo task=detect mode=predict conf=0.25 save=True \
model={HOME}/runs/detect/train/weights/best.pt \
source={image_folder}

import cv2
import supervision as sv
from ultralytics import YOLO
import glob

# بارگذاری مدل سفارشی
model = YOLO(f'/content/runs/detect/train2/weights/best.pt')

# بارگذاری تصاویر
image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))

# پردازش و نمایش هر تصویر
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image from path: {image_path}")

    results = model(source=image_path, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(results)

    # استفاده از نام‌های جدید کلاس‌ها
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # نمایش تصویر
    annotated_image = box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    sv.plot_image(annotated_image)

"""Abolfazl Sohrabi

[GitHub](https://github.com/AABOLFAZLSOHRABI)
"""
