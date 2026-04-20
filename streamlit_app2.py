#streamlit run streamlit_app2.py
import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import os
import tempfile

try:
    import tomllib as toml_parser
except ModuleNotFoundError:
    try:
        import tomli as toml_parser
    except ModuleNotFoundError:
        try:
            import toml as toml_parser
        except ModuleNotFoundError:
            toml_parser = None

# --- Configuration ---
st.set_page_config(page_title="Sewing Defect Detector", layout="wide")

# Load secrets from Streamlit or local .streamlit/secrets.toml as fallback

def load_local_secret(key: str) -> str | None:
    if toml_parser is None:
        return None

    secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        return None

    try:
        with open(secrets_path, "rb") as f:
            data = toml_parser.load(f)
        return data.get(key)
    except Exception:
        return None

st_secret = st.secrets.get("roboflow_api_key")
env_secret = os.getenv("ROBOFLOW_API_KEY")
local_secret = load_local_secret("roboflow_api_key")
ROBOFLOW_API_KEY = st_secret or env_secret or local_secret

secret_source = None
if st_secret:
    secret_source = "st.secrets"
elif env_secret:
    secret_source = "environment variable"
elif local_secret:
    secret_source = ".streamlit/secrets.toml"

secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml")
st.write(f"Secrets file exists: {os.path.exists(secrets_path)}")
st.write(f"st.secrets key present: {st_secret is not None}")
st.write(f"env key present: {env_secret is not None}")
st.write(f"local secret loaded: {local_secret is not None}")

if not ROBOFLOW_API_KEY:
    st.error(
        "Roboflow API key is missing. Add `roboflow_api_key` to `.streamlit/secrets.toml` "
        "or set the `ROBOFLOW_API_KEY` environment variable."
    )
    st.stop()

st.write(f"Roboflow API key loaded from: {secret_source}")

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
                # For classification models, show the top prediction
                if 'top' in p and 'confidence' in p:
                    confidence_pct = p['confidence'] * 100
                    st.success(f"Predicted: **{p['top']}** with {confidence_pct:.1f}% confidence.")
                else:
                    # Fallback for other prediction formats
                    st.info(f"Prediction result: {p}")
        else:
            st.info("No predictions available.")
