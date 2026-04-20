import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import os

# --- Configuration ---
st.set_page_config(page_title="Sewing Defect Detector", layout="wide")

# Initialize Roboflow
ROBOFLOW_API_KEY = st.secrets.get("roboflow_api_key") or os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    st.error(
        "Roboflow API key is missing. Add `roboflow_api_key` to `.streamlit/secrets.toml` "
        "or set the `ROBOFLOW_API_KEY` environment variable."
    )
    st.stop()

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("saifulae20").project("defects-w7oxz")
model = project.version(1).model

def process_image(img_array):
    """Sends image to Roboflow and draws bounding boxes on the result."""
    # Convert RGB (Streamlit) to BGR (OpenCV)
    image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Save temporary image for Roboflow API
    temp_path = "temp_input.jpg"
    cv2.imwrite(temp_path, image_bgr)
    
    # Run Prediction
    prediction = model.predict(temp_path, confidence=40, overlap=30).json()
    
    # Draw annotations
    for pred in prediction.get('predictions', []):
        # Extract coordinates (Roboflow returns center x, y and width, height)
        x0 = int(pred['x'] - pred['width'] / 2)
        y0 = int(pred['y'] - pred['height'] / 2)
        x1 = int(pred['x'] + pred['width'] / 2)
        y1 = int(pred['y'] + pred['height'] / 2)
        
        label = f"{pred['class']} ({pred['confidence']:.2f})"
        
        # Draw Box
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (0, 255, 0), 3)
        # Draw Label background
        cv2.putText(image_bgr, label, (x0, y0 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Convert back to RGB for Streamlit display
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), prediction.get('predictions', [])

# --- UI Layout ---
st.title("Sewing Defect Detection")
st.write("Point your camera at the fabric to detect defects in real-time.")

col1, col2 = st.columns(2)

with col1:
    img_file = st.camera_input("Take a picture to analyze")

if img_file is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # Process
    with st.spinner('Analyzing fabric...'):
        processed_img, preds = process_image(opencv_image)

    with col2:
        st.subheader("Analysis Result")
        st.image(processed_img, use_column_width=True)
        
        if preds:
            for p in preds:
                st.success(f"Detected: **{p['class']}** with {p['confidence']:.1%} confidence.")
        else:
            st.info("No defects detected.")
