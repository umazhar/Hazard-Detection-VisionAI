import requests
import base64
import cv2
import numpy as np

url = "http://127.0.0.1:8000/detect/"
image_path = "water.jpg"

# Send the image to the API
with open(image_path, "rb") as img_file:
    files = {"file": img_file}
    response = requests.post(url, files=files)

# Parse response
response_json = response.json()

if "annotated_image" in response_json:
    # Decode base64 string
    img_base64 = response_json["annotated_image"]
    decoded_bytes = base64.b64decode(img_base64)
    np_arr = np.frombuffer(decoded_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Save the image as "annotated_output.jpg"
    cv2.imwrite("annotated_output.jpg", image)

    print("Annotated image saved as 'annotated_output.jpg'")
else:
    print("Error:", response_json.get("error", "Unknown error"))
