# -*- coding: utf-8 -*-
"""
Abolfazl Sohrabi
"""

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="**********") #your API Roboflow
project = rf.workspace("-jwzpw").project("continuous_fire")
version = project.version(6)
dataset = version.download("yolov8")

"""# Train Model"""

!pip install ultralytics

from ultralytics import YOLO

# Load a model
model = YOLO('yolov10n.pt')  # load a pretrained model (recommended for training)

# Define training arguments including early stopping
train_args = {
    'data': '/content/continuous_fire-6/data.yaml',
    'epochs': 100,
    'imgsz': 640,
    'patience': 5  # This will enable early stopping with patience of 5 epochs
}

# Train the model
results = model.train(**train_args)

"""**Abolfazl Sohrabi**"""