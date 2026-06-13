import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import os
import gdown

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI in Sewing: MobileNetV2 Engine", layout="wide")

# File ID extracted from your shared link
DRIVE_FILE_ID = '1Z_6SWe02HrlFi51RYYSzHi-dSnUrbu5V'
MODEL_PATH = 'mobilenetv2_satin_core.keras'
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Broken stitch", "Holes", "Raw edge", "Seam puckering", 
    "Skipped stitch", "Staggered stitch", "Unbalanced stitch", 
    "Variable stitch density"
]

# --- 2. DYNAMIC MODEL DOWNLOAD & LOAD ---
@st.cache_resource
def load_apparel_model():
    """Downloads model from Google Drive if not already present."""
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner('Downloading 28MB MobileNet model from Google Drive...'):
                url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
                gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None
            
    try:
        model = load_model(MODEL_PATH)
        st.sidebar.success("✅ MobileNetV2 Engine Ready")
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

# --- 3. OPENCV PREPROCESSING ---
def preprocess_image(pil_image):
    """Matches the exact logic of your successful research script."""
    open_cv_image = np.array(pil_image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    # Grayscale -> Resize -> 3-Channel Repeat -> Normalize
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    img_array = np.expand_dims(resized, axis=-1)
    img_array = np.repeat(img_array, 3, axis=-1)
    img_array = img_array.astype('float32') / 255.0
    
    return np.expand_dims(img_array, axis=0)

# --- 4. INITIALIZATION ---
model = load_apparel_model()

# --- 5. UI LAYOUT ---
st.title("AI in Sewing: Quality Inspector")
st.markdown("Automated defect detection using MobileNetV2.")
st.divider()

st.sidebar.header("Scanner Settings")
input_mode = st.sidebar.radio("Input Source:", ("Upload Image", "Live Camera"))

img_source = None
if input_mode == "Upload Image":
    img_source = st.file_uploader("Select image", type=['jpg', 'jpeg', 'png'])
else:
    img_source = st.camera_input("Scan sample")

if img_source is not None and model is not None:
    # Process image
    image = Image.open(img_source).convert('RGB')
    processed_input = preprocess_image(image)
    
    # Prediction
    with st.spinner('Analyzing textile surface...'):
        prediction = model.predict(processed_input)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        label = CLASS_NAMES[class_index]

    # --- 6. RESULTS DISPLAY ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Captured Sample")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("AI Analysis")
        
        # Display detection alert
        if confidence > 55:
            st.error(f"**DEFECT DETECTED:** {label}")
            st.metric("Probability", f"{confidence:.1f}%")
        else:
            st.success("No clear sewing defects detected.")
            st.write(f"Highest similarity: {label} ({confidence:.1f}%)")

        # Probability chart
        with st.expander("View Full Defect Distribution"):
            for i, prob in enumerate(prediction[0]):
                st.write(f"{CLASS_NAMES[i]}")
                st.progress(float(prob))
else:
    if model is None:
        st.warning("Waiting for model to initialize...")
    else:
        st.info("Ready for input. Please upload an image or use the camera.")