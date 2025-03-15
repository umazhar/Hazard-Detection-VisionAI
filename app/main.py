from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = FastAPI(title="YOLO Object Detection API", description="Upload an image and get an annotated version with detected objects.")

# Load YOLO model from Ultralytics (downloads if not found)
model = YOLO("../yolov8n.pt")  # Change to "yolov8s.pt" for a larger model

# Enable CORS (for frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve HTML templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
def read_root(request: Request):
    """Serve the frontend HTML file."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Accepts an image file, runs YOLO detection, and returns detection data with annotated image.
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        # Store original image as base64
        original_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        _, original_buffer = cv2.imencode(".jpg", original_img)
        original_base64 = base64.b64encode(original_buffer).decode("utf-8")

        # Perform object detection
        results = model(image_np)

        # Draw bounding boxes
        annotated_image = results[0].plot()  # OpenCV image with annotations

        # Convert to base64 for easy API return
        _, buffer = cv2.imencode(".jpg", annotated_image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        # Extract detection information
        detections = []
        for i, detection in enumerate(results[0].boxes.data.tolist()):
            x1, y1, x2, y2, confidence, class_id = detection
            class_id = int(class_id)
            class_name = results[0].names[class_id]
            
            detections.append({
                "id": i,
                "class": class_name,
                "confidence": float(confidence),  # Convert to float for JSON serialization
                "box": [round(float(x)) for x in [x1, y1, x2, y2]]  # Convert to float then round
            })

        return {
            "original_image": f"data:image/jpeg;base64,{original_base64}",
            "output_image": f"data:image/jpeg;base64,{img_base64}",
            "detections": detections
        }
    except Exception as e:
        return {"error": str(e)}