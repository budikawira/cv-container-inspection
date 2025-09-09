"""
Web Service
pip install python-multipart
pip install fastapi
"""
from fastapi import FastAPI, UploadFile, File, Form
from contextlib import asynccontextmanager
from app.core.loader import load_models
from fastapi.security import OAuth2PasswordBearer
import os
from typing import List
from fastapi.responses import JSONResponse
import shutil
import app.core.utils as utils
from PIL import Image
import app.core.cnr as cnr
import app.core.ocr as ocr

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Loading YOLO models...")
    models = load_models()
    app.state.models = models
    print("✅ Models loaded.")
    yield
    print("🧹 Cleaning up models...")
    app.state.models = None

# Create a FastAPI application
app = FastAPI(lifespan=lifespan)

@app.post("/upload-multiple/")
async def upload_multiple_images(
    images: List[UploadFile] = File(...),
    labels: List[str] = Form(...)
):
    if len(images) != len(labels):
        return JSONResponse(
            status_code=400,
            content={"error": f"Number of images and labels must match: {len(images)} images, {len(labels)} labels"}
        )

    response = []
    for image, label in zip(images, labels):
        # create or use folder YYYYmmdd
        folder = os.path.join(UPLOAD_DIR, utils.generate_date_string())
        # create folder HHMMSS
        folder = os.path.join(folder, utils.generate_time_string())
        os.makedirs(folder)
        image_path = os.path.join(folder, f"{label}-src.jpg")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        #Object Detection: Detect CNR
        cnr_model = app.state.models["cnr"]
        results = cnr_model.predict(Image.open(image_path),conf=0.4)
        result_path = os.path.join(folder, f"{label}-cnr.jpg")
        results[0].save(filename=result_path)
        cnr.save_yolo_predictions(results, result_path, folder)
        crop_filenames = cnr.crop_predicted_objects(cnr_model, image_path, results, folder)

        #Object Detection: Detect Letters (OCR)
        ocr_model = app.state.models["ocr"]
        for crop_filename in crop_filenames:
            params = crop_filename.split('_')
            print(f"params : {params}")
            crop_filepath = os.path.join(folder, crop_filename)
            preproc_img = ocr.preprocess_image_for_ocr(Image.open(crop_filepath))
            ocr_results = ocr_model.predict(preproc_img,conf=0.4)
            ocr_results = ocr.remove_overlapping_detections(ocr_results, iou_threshold=0.5)

            #Check vertical or horizontal
            orientation = 'x'
            if (params[3].startswith("V ")):
                orientation = 'y'
            text, text_lbls, avg_conf = ocr.get_sorted_label_string(ocr_model, ocr_results, sort_by=orientation)
            print(f"letters predictions : {text_lbls}")
            utils.save_text(f"{text}\n{avg_conf}", os.path.join(folder, utils.remove_extension(crop_filename) + ".txt"))
            print(f"text : {text}")

        response.append({
            "filename": image.filename,
            "status": "uploaded"
        })

    return JSONResponse(content={"results": response})

# Define a route at the root web address ("/")
@app.get("/")
async def read_root():
    """
    Main page
    """
    return {"message": "Container Inspection Computer Vision!"}