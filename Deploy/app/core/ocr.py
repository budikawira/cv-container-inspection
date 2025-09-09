import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from ultralytics.engine.results import Results
import torch
import cv2

def get_sorted_label_string(model, results, sort_by='x'):
    """
    Extracts class labels from model results and returns a string
    sorted by the x_min or y_min of each bounding box, along with confidence details.

    Parameters:
        model: the detection model (used to map class indices to names)
        results: model output (can be list or single Results object)
        sort_by: 'x' for left-to-right, 'y' for top-to-bottom

    Returns:
        tuple:
            - str: concatenated class labels (e.g., "ABC123")
            - list[dict]: [{'label': 'A', 'confidence': 0.98}, ...]
            - float: average confidence
    """
    # Handle list of results (common when model() is called on image)
    if isinstance(results, list):
        if len(results) == 0:
            return "", [], 0.0
        boxes = results[0].boxes
    else:
        boxes = results.boxes

    # If no boxes detected
    if boxes is None or len(boxes) == 0:
        return "", [], 0.0

    # Extract data
    xyxy = boxes.xyxy.cpu().numpy()      # [N, 4] -> x1, y1, x2, y2
    cls_ids = boxes.cls.cpu().numpy()    # [N]
    confs = boxes.conf.cpu().numpy()     # [N]

    # Choose sort key: x_min (index 0) or y_min (index 1)
    sort_index = 0 if sort_by == 'x' else 1
    sort_key = xyxy[:, sort_index]

    # Combine detection data
    detections = []
    for i in range(len(boxes)):
        detections.append({
            'cls_id': int(cls_ids[i]),
            'confidence': float(confs[i]),
            'x_min': float(xyxy[i, 0]),
            'y_min': float(xyxy[i, 1])
        })

    # Sort by sort_key (x_min or y_min), ascending
    sorted_detections = sorted(detections, key=lambda d: d['x_min' if sort_by == 'x' else 'y_min'])

    # Generate label string
    label_str = ''.join(model.names[d['cls_id']] for d in sorted_detections)

    # Detailed list with label and confidence
    detailed_labels = [
        {'label': model.names[d['cls_id']], 'confidence': d['confidence']}
        for d in sorted_detections
    ]

    # Calculate average confidence
    avg_confidence = sum(d['confidence'] for d in sorted_detections) / len(sorted_detections)

    return label_str, detailed_labels, avg_confidence


def preprocess_image_for_ocr(input_image, scale_factor=2, denoise_strength=10, sharpness_factor=1.5):
    """
    Preprocess an image for OCR that accepts both PIL Images and NumPy arrays.
    
    Args:
        input_image (PIL.Image.Image or np.ndarray): Input image
        scale_factor (int): Factor to upscale the image. Default 2.
        denoise_strength (int): Strength for denoising. Default 10.
        sharpness_factor (float): Factor for sharpening. Default 1.5.
    
    Returns:
        numpy array: Processed image as a PIL Image object.
    """
    
    # Convert to PIL Image if it's a NumPy array
    if isinstance(input_image, np.ndarray):
        # Convert BGR to RGB if it's a color image
        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(input_image)
    else:
        img = input_image.copy()
    
    # 1. Convert to grayscale (Luminosity mode)
    # if img.mode != 'L':
    #     img = img.convert('L')
    
    # 2. Resize (Upscale) using LANCZOS filter
    if scale_factor != 1:
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 3. Convert to numpy array for OpenCV denoising if needed
    if denoise_strength > 0:
        import cv2
        img_np = np.array(img)
        denoised_np = cv2.fastNlMeansDenoising(img_np, h=denoise_strength, 
                                              templateWindowSize=7, 
                                              searchWindowSize=21)
        img = Image.fromarray(denoised_np)
    
    # 4. Enhance contrast
    img = ImageOps.autocontrast(img, cutoff=2)
    
    # 5. Apply sharpening
    if sharpness_factor > 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness_factor)
        
    # # 6. Convert to black and white (binary)
    # img_bw = img.convert('L')  # Ensure grayscale
    # threshold = 128
    # # Apply threshold: pixels > threshold become white (255), else black (0)
    # binary = img_bw.point(lambda p: 255 if p > threshold else 0)
    
    # # Convert back to RGB
    # img = Image.merge('RGB', (binary, binary, binary))
    
    return np.array(img)

def remove_overlapping_detections(results, iou_threshold=0.5):
    """
    Removes overlapping OCR detections based on IoU.
    Keeps the one with higher confidence in each overlap group.

    Args:
        results: YOLO Results object or list of Results
        iou_threshold (float): IoU threshold (e.g., 0.5 = 50% overlap)

    Returns:
        Results: New Results object with overlapping boxes removed
    """
    from ultralytics.engine.results import Results
    import torch
    import numpy as np

    # Handle list input
    if isinstance(results, list):
        result = results[0]
    else:
        result = results

    if result.boxes is None or len(result.boxes) == 0:
        return result

    # Get data
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
    scores = result.boxes.conf.cpu().numpy()      # (N,)
    classes = result.boxes.cls.cpu().numpy()      # (N,)
    orig_img = result.orig_img                    # needed
    path = result.path
    names = result.names

    # Sort by confidence descending
    indices = np.argsort(-scores)
    keep = []

    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter_area = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    # Greedy NMS: keep high-confidence, remove overlapping
    for i in indices:
        include = True
        for k in keep:
            if calculate_iou(boxes_xyxy[i], boxes_xyxy[k]) > iou_threshold:
                include = False
                break
        if include:
            keep.append(i)

    # Reconstruct box tensor: [x1, y1, x2, y2, conf, cls]
    if len(keep) == 0:
        # Empty detection
        new_boxes = torch.empty((0, 6), device=result.boxes.data.device)
    else:
        new_boxes = result.boxes.data[keep]  # ✅ This already has shape [N, 6]

    # ✅ Correct: Only pass valid parameters
    filtered_result = Results(
        orig_img=orig_img,
        path=path,
        names=names,
        boxes=new_boxes  # ← This is the only required detection input
    )

    return filtered_result