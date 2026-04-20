#streamlit run streamlit_app2.py
import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import os
import tempfile

# --- Configuration ---
st.set_page_config(page_title="Sewing Defect Detector", layout="wide")

# Roboflow API Key (hardcoded)
ROBOFLOW_API_KEY = "f9c3ehSW7N64x9RyzxbT"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("saifulae20").project("defects-w7oxz")
model = project.version(1).model

def process_image(img_array):
    """Sends image to Roboflow and processes the result."""
    # Convert RGB (Streamlit) to BGR (OpenCV)
    image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    prediction = {}
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, image_bgr)

        response = model.predict(temp_path)
        if hasattr(response, "json"):
            prediction = response.json()
        else:
            prediction = response
    except Exception as e:
        st.error(f"Roboflow prediction failed: {e}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), []
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    # For classification models, there are no bounding boxes to draw
    # Just return the original image and the predictions
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), prediction.get('predictions', [])

# --- UI Layout ---
st.title("Sewing Defect Classification")
st.write("Point your camera at the fabric to classify defects in real-time.")

col1, col2 = st.columns(2)

with col1:
    img_file = st.camera_input("Take a picture to classify")
    img_upload = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

if img_file is not None or img_upload is not None:
    if img_file is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    elif img_upload is not None:
        # Load uploaded image
        image = Image.open(img_upload)
        opencv_image = np.array(image)

    # Process
    with st.spinner('Analyzing fabric...'):
        processed_img, preds = process_image(opencv_image)

    with col2:
        st.subheader("Analysis Result")
        st.image(processed_img, use_column_width=True)
        
        if preds:
            for p in preds:
                # For classification models, show the top prediction
                if 'top' in p and 'confidence' in p:
                    confidence_pct = p['confidence'] * 100
                    st.success(f"Predicted: **{p['top']}** with {confidence_pct:.1f}% confidence.")
                else:
                    # Fallback for other prediction formats
                    st.info(f"Prediction result: {p}")
        else:
            st.info("No predictions available.")
