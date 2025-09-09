# FastAPI Image Upload Service

A simple and scalable API for uploading multiple images with associated labels, built using **FastAPI** and served with **Uvicorn**.

---

## 🚀 Features

- Upload multiple images in a single request
- Attach custom labels to each image
- Secure file saving with unique filenames
- Detect Container Number Records (CNR) Object
- Read the CNR text
- Detect Container Damages

---

## 🛠️ Tech Stack

- **FastAPI** – Modern Python web framework for building APIs
- **Uvicorn** – Lightning-fast ASGI server
- **Python 3.8+** – Required

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fastapi-upload-service.git
   cd fastapi-upload-service

2. Install required libraries from requirements.txt
   ```bash
   TBD

3. Run the server
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

## 🧪 Testing with Swagger UI
After starting the server, open your browser and go to:
👉 [http://localhost:8000/docs](http://localhost:8000/docs)
   
   
