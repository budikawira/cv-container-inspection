"""
CNR (Container Number Records) Libraries
"""
import cv2
import os
from typing import Tuple
import numpy as np
from typing import List

def save_yolo_predictions(results, image_path, output_dir="predictions"):
    """
    Save YOLO prediction results in YOLO label format (.txt).

    Args:
        results (list): List of YOLO prediction results (from model("image.jpg")).
        image_path (str): Path to the input image (used to name the .txt file).
        output_dir (str): Directory to save .txt files.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get filename without extension
    image_name = os.path.basename(image_path)
    txt_filename = os.path.splitext(image_name)[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_filename)

    # Get the first (and only) result
    result = results[0]
    orig_img = result.orig_img
    h, w = orig_img.shape[:2]  # height, width

    # Open .txt file for writing
    with open(txt_path, "w") as f:
        for box in result.boxes:
            # Class ID
            cls_id = int(box.cls.item())

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Convert to YOLO format: normalized center, width, height
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bbox_width = (x2 - x1) / w
            bbox_height = (y2 - y1) / h

            # Write line in YOLO format
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    print(f"YOLO prediction saved to {txt_path}")
    return txt_path

def crop_predicted_objects(model, image_path: str, results, output_dir: str = "crops") -> List[str]:
    """
    Crops detected objects from an image based on YOLO prediction results.

    Args:
        model: Trained YOLO model.
        image_path (str): Path to the input image.
        results: Results from model() — can be list or Results object.
        output_dir (str): Directory to save cropped images.

    Returns:
        bool: True if successful, False otherwise.
    """
    crop_filenames = []
    image_name = f"{os.path.basename(image_path).split('-')[0]}-cnr"
    try:
        os.makedirs(output_dir, exist_ok=True)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image at {image_path}")
            return crop_filenames

        # Handle list of results
        if isinstance(results, list):
            if len(results) == 0:
                print("No results found.")
                return crop_filenames
            result = results[0]  # take first result
        else:
            result = results

        if result.boxes is None or len(result.boxes) == 0:
            print("No objects detected.")
            return True

        # Get data from result.boxes
        xyxy = result.boxes.xyxy.cpu().numpy()  # shape: (N, 4)
        confs = result.boxes.conf.cpu().numpy()  # shape: (N,)
        class_ids = result.boxes.cls.cpu().numpy()  # shape: (N,)
        names = model.names

        height, width = image.shape[:2]

        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            cropped_image = image[y1:y2, x1:x2]
            class_id = int(class_ids[i])
            confidence = float(confs[i])
            class_name = names[class_id]

            crop_filename = f"{image_name}_{i:03d}_{class_id}_{class_name}_{confidence:.2f}.jpg"
            crop_filenames.append(crop_filename)
            crop_path = os.path.join(output_dir, crop_filename)

            success = cv2.imwrite(crop_path, cropped_image)
            if not success:
                print(f"Failed to save: {crop_path}")
                return crop_filenames

        return crop_filenames

    except Exception as e:
        print(f"Error during cropping: {e}")
        return crop_filenames