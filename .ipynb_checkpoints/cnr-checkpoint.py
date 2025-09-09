import numpy as np
from PIL import Image, ImageOps, ImageEnhance

def get_sorted_label_string(model, results, sort_by='x'):
    """
    Extracts class labels from model results and returns a string
    sorted by the x_min or y_min of each bounding box.

    Parameters:
        model: the detection model (used to map class indices to names)
        results: model output containing .boxes with .xyxy and .cls
        sort_by: 'x' for horizontal sort (left to right),
                 'y' for vertical sort (top to bottom)

    Returns:
        str: concatenated class labels sorted by specified axis
    """
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()  # shape: [N, 4]
    cls = boxes.cls.cpu().numpy()    # shape: [N]

    # Choose sorting axis
    if sort_by == 'y':
        sort_key = xyxy[:, 1]  # y_min
        reverse = False        # top to bottom (smaller y first)
    else:
        sort_key = xyxy[:, 0]  # x_min
        reverse = False        # left to right

    # Sort and extract labels
    sorted_labels = [label for _, label in sorted(zip(sort_key, cls), reverse=reverse)]

    # Convert to string using model.names
    label_str = ''.join(str(model.names[int(c)]) for c in sorted_labels)
    return label_str


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
    # Import OpenCV inside the function
    import cv2
    
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
    
def crop_predicted_objects(image, results):
    """
    Crops detected objects from an image based on YOLO prediction results and returns them with metadata.

    Args:
        image (np.ndarray): The original image read by OpenCV.
        results (ultralytics.yolo.engine.results.Results): The results object from model.predict().

    Returns:
        tuple: A tuple containing three elements:
            - list[np.ndarray]: A list of cropped images.
            - list[str]: A list of class names for each cropped image.
            - list[float]: A list of confidence scores for each detection.
        Returns empty lists if no objects are detected.
    """
    cropped_images = []
    class_names = []
    confidences = []
    
    # Check if there are any detections
    if results.boxes is None:
        return cropped_images, class_names, confidences
    
    # Get the names of the classes from the model
    names = results.names
    
    # Get the bounding boxes, confidence scores, and class IDs from the results object
    boxes = results.boxes
    confs = boxes.conf
    class_ids = boxes.cls

    # Iterate through each detected object
    for i, box in enumerate(boxes):
        # Get the bounding box coordinates in xyxy format
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within image boundaries
        height, width = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Crop the image only if the bounding box is valid
        if x2 > x1 and y2 > y1:
            cropped_image = image[y1:y2, x1:x2]
            
            cropped_images.append(preprocess_image_for_ocr(cropped_image))
            # cropped_images.append(cropped_image)
            
            # Get the class name and confidence for this detection
            class_id = int(class_ids[i].item())
            class_name = names[class_id]
            confidence = confs[i].item()
            
            class_names.append(class_name)
            confidences.append(confidence)
            
    return cropped_images, class_names, confidences