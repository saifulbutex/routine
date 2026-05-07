# streamlit run streamlit_app2.py
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import gdown

try:
    import tensorflow as tf
    load_model = tf.keras.models.load_model
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "TensorFlow is not installed. Install dependencies with `pip install -r requirements.txt` "
        "or add `tensorflow-cpu` to your environment."
    ) from e

# --- Configuration ---
st.set_page_config(page_title="Sewing Defect Detector", layout="wide")

MODEL_PATH = "best_model.h5"
GDRIVE_ID = "1PQ1_xwU7tx9aYP-0kvWSpklmrRZuyZru"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"
IMG_WIDTH = 224  # default, will be updated from model
IMG_HEIGHT = 224
CLASS_NAMES = ["good", "defect"]  # Update this list to match your model class order

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    model = load_model(MODEL_PATH)
    # Get the input shape from the model
    input_shape = model.input_shape  # e.g., (None, 224, 224, 3)
    global IMG_HEIGHT, IMG_WIDTH
    IMG_HEIGHT, IMG_WIDTH = input_shape[1], input_shape[2]
    return model

model = load_my_model()


def preprocess_image(img_array):
    image_rgb = img_array
    image_resized = cv2.resize(image_rgb, (IMG_WIDTH, IMG_HEIGHT))
    image_norm = image_resized.astype("float32") / 255.0
    return np.expand_dims(image_norm, axis=0)


def norm_coord(value, size):
    if 0 <= value <= 1:
        return int(value * size)
    return int(value)


def draw_annotation_box(image, box, label):
    x1, y1, x2, y2 = box
    annotated = image.copy()
    color = (0, 255, 0)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        annotated,
        label,
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated


def parse_detection(pred):
    if not isinstance(pred, dict):
        return None

    if "bbox" in pred and isinstance(pred["bbox"], (list, tuple)) and len(pred["bbox"]) == 4:
        x1, y1, w, h = pred["bbox"]
        return x1, y1, w, h
    if {"x", "y", "width", "height"}.issubset(pred.keys()):
        return pred["x"], pred["y"], pred["width"], pred["height"]
    if {"x_min", "y_min", "x_max", "y_max"}.issubset(pred.keys()):
        return pred["x_min"], pred["y_min"], pred["x_max"] - pred["x_min"], pred["y_max"] - pred["y_min"]
    return None


def annotate_image(img_array, preds):
    annotated = img_array.copy()
    h, w = annotated.shape[:2]
    drawn = False

    if isinstance(preds, list):
        for pred in preds:
            box = parse_detection(pred)
            if box is not None:
                x, y, width, height = box
                x1 = np.clip(norm_coord(x, w), 0, w - 1)
                y1 = np.clip(norm_coord(y, h), 0, h - 1)
                x2 = np.clip(norm_coord(x + width, w), 0, w - 1)
                y2 = np.clip(norm_coord(y + height, h), 0, h - 1)
                label = pred.get("label", pred.get("class", "defect"))
                confidence = pred.get("confidence")
                if confidence is not None:
                    label = f"{label} {confidence * 100:.1f}%"
                annotated = draw_annotation_box(annotated, (x1, y1, x2, y2), label)
                drawn = True

    if not drawn and isinstance(preds, list) and len(preds) == 1 and isinstance(preds[0], dict):
        label = preds[0].get("label") or preds[0].get("class")
        if label is not None and label.lower() != "good":
            annotated = draw_annotation_box(
                annotated,
                (0, 0, w - 1, h - 1),
                f"{label} {preds[0].get('confidence', 0) * 100:.1f}%" if preds[0].get("confidence") is not None else label,
            )
            drawn = True

    return annotated


def process_image(img_array):
    """Preprocesses an uploaded image and predicts with the local model."""
    try:
        image_batch = preprocess_image(img_array)
        preds = model.predict(image_batch)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        return img_array, []

    if isinstance(preds, np.ndarray) and preds.size == 0:
        return img_array, []

    if isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[0] == 1:
        preds = preds[0]

    labels = []
    annotated_image = img_array.copy()

    if isinstance(preds, np.ndarray):
        preds = preds.flatten()

    if isinstance(preds, (list, np.ndarray)) and len(preds) > 1 and not (isinstance(preds[0], dict) or isinstance(preds[0], list)):
        if len(preds) == 1:
            score = float(preds[0])
            label = "defect" if score >= 0.5 else "good"
            confidence = score if score >= 0.5 else 1.0 - score
            labels.append({"label": label, "confidence": confidence})
        elif len(preds) == len(CLASS_NAMES):
            class_id = int(np.argmax(preds))
            label = CLASS_NAMES[class_id]
            labels.append({"label": label, "confidence": float(preds[class_id])})
        else:
            class_id = int(np.argmax(preds))
            label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
            labels.append({"label": label, "confidence": float(preds[class_id])})
    elif isinstance(preds, list):
        labels = preds
    elif isinstance(preds, dict):
        labels = [preds]
    else:
        labels = []

    annotated_image = annotate_image(img_array, labels)
    return annotated_image, labels


# --- UI Layout ---
st.title("Sewing Defect Classification")
st.write("Use your back camera to take a photo or upload an image. If permission is requested, allow it and choose the rear camera.")

# Force file inputs to use the back camera on supported mobile browsers.
st.markdown(
    """
    <script>
    const setBackCamera = () => {
      const inputs = document.querySelectorAll('input[type=file]');
      inputs.forEach(input => {
        if (input.accept && input.accept.includes('image') && input.getAttribute('capture') !== 'environment') {
          input.setAttribute('capture', 'environment');
        }
      });
    };
    setBackCamera();
    const observer = new MutationObserver(setBackCamera);
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    img_file = st.camera_input("Take a back-camera picture", key="back_camera")
    img_upload = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"], key="upload")

if img_file is not None or img_upload is not None:
    opencv_image = None

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        opencv_image = np.array(image)
    if opencv_image is None and img_upload is not None:
        image = Image.open(img_upload).convert("RGB")
        opencv_image = np.array(image)

    if opencv_image is not None:
        height, width = opencv_image.shape[:2]
        if width < 300 or height < 300:
            st.warning("Image resolution is low. For better detection, upload a higher-resolution photo or use a better camera capture.")

        with st.spinner('Analyzing fabric...'):
            processed_img, preds = process_image(opencv_image)

        with col2:
            st.subheader("Analysis Result")
            st.image(processed_img, use_column_width=True)

            if preds:
                defect_preds = [p for p in preds if p.get('label', '').lower() not in ('good', 'no defect', 'none')]
                if defect_preds:
                    for p in defect_preds:
                        st.success(f"Detected: **{p['label']}** with {p['confidence'] * 100:.1f}% confidence.")
                else:
                    st.info("No defect detected.")
            else:
                st.info("No defect detected.")
    else:
        st.error("Could not read the image. Please try another photo or upload.")
