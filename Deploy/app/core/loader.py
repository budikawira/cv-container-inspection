"""
Model Loaders
"""
from ultralytics import YOLO
import os

def load_models():
    base_dir = os.path.join("assets")
    cnr_model = YOLO(os.path.join(base_dir, "cnr.pt"))
    ocr_model = YOLO(os.path.join(base_dir, "ocr.pt"))
    return {"cnr": cnr_model, "ocr": ocr_model}

