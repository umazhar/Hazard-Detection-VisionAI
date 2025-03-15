from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = FastAPI(title="YOLO Object Detection API", description="Upload an image and get an annotated version with detected objects.")

# Load YOLO model from Ultralytics (downloads if not found)
model = YOLO("yolov8n.pt")  # Change to "yolov8s.pt" for a larger model

# Enable CORS (for frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"message": "YOLO Object Detection API is running!"}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Accepts an image file, runs YOLO detection, and returns a base64-encoded annotated image.
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        # Perform object detection
        results = model(image_np)

        # Draw bounding boxes
        annotated_image = results[0].plot()  # OpenCV image with annotations

        # Convert to base64 for easy API return
        _, buffer = cv2.imencode(".jpg", annotated_image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return {"annotated_image": img_base64}
    except Exception as e:
        return {"error": str(e)}

