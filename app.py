import os
import json
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# -------------------
# CONFIG
# -------------------
IMG_SIZE = (224, 224)  # must match what you trained MobileNetV2 on
LOCAL_MODEL_PATH = "models/mobilenetv2_best.h5"  # or .keras
CLASS_MAP_PATH = "models/class_indices.json"

# If using Option B (download at runtime), set a direct URL here:
MODEL_URL = ""  # e.g., "https://huggingface.co/youruser/yourrepo/resolve/main/mobilenetv2_best.h5"
# For Google Drive via gdown, install gdown and set MODEL_FILE_ID:
USE_GDOWN = False
MODEL_FILE_ID = ""  # like "1AbCDeFGhiJKLmnop" if using gdown

# -------------------
# HELPERS
# -------------------
@st.cache_data(show_spinner=False)
def load_class_map(path: str):
        with open(path, "r") as f:
                    idx2class = json.load(f)  # {"0":"classA","1":"classB",...}
    # Ensure numeric keys are sorted order
        keys_sorted = sorted(idx2class.keys(), key=lambda k: int(k))
        classes = [idx2class[k] for k in keys_sorted]
        return classes

@st.cache_resource(show_spinner=True)
def get_model():
        # Option A: load from repo file
        if os.path.exists(LOCAL_MODEL_PATH):
                    model_path = LOCAL_MODEL_PATH
else:
        # Option B: download at runtime
            if MODEL_URL:
                            import urllib.request
                            tmp_dir = tempfile.gettempdir()
                            model_path = os.path.join(tmp_dir, os.path.basename(MODEL_URL))
                            if not os.path.exists(model_path):
                                                st.info("Downloading model...")
                                                urllib.request.urlretrieve(MODEL_URL, model_path)
            elif USE_GDOWN and MODEL_FILE_ID:
                            import gdown, tempfile
                            tmp_dir = tempfile.gettempdir()
                            model_path = os.path.join(tmp_dir, "mobilenetv2_best.h5")
                            if not os.path.exists(model_path):
                                                st.info("Downloading model from Google Drive...")
                                                url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
                                                gdown.download(url, model_path, quiet=False)
            else:
                            raise FileNotFoundError(
                                                "Model file not found. Either add it to 'models/' or set MODEL_URL / gdown FILE_ID."
                            )

        # compile=False avoids issues with missing custom metrics at load time
        model = tf.keras.models.load_model(model_path, compile=False)
    return model

def preprocess_image(img: Image.Image, target_size):
        img = img.convert("RGB").resize(target_size)
        arr = np.array(img).astype("float32") / 255.0  # scale to [0,1]
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

def predict_topk(model, img_array, class_names, k=3):
        preds = model.predict(img_array, verbose=0)[0]  # (num_classes,)
    top_idx = np.argsort(preds)[::-1][:k]
    return [(class_names[i], float(preds[i])) for i in top_idx]

# -------------------
# UI
# -------------------
st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="centered")

st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload a fish image to get the predicted species and confidence scores.")

# Load resources
try:
        class_names = load_class_map(CLASS_MAP_PATH)
except Exception as e:
        st.error(f"Failed to load class map at {CLASS_MAP_PATH}: {e}")
        st.stop()

try:
        model = get_model()
except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_column_width=True)

    arr = preprocess_image(image, IMG_SIZE)
    with st.spinner("Predicting..."):
                top = predict_topk(model, arr, class_names, k=3)

    # Show top-1
    pred_label, pred_conf = top[0]
    st.success(f"**Prediction:** {pred_label}  \n**Confidence:** {pred_conf:.2%}")

    # Show top-3 as a small table/bar
    st.subheader("Top-3 scores")
    for label, score in top:
                st.write(f"- {label}: {score:.2%}")
