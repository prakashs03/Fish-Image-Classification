import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Fish Image Classification", layout="wide")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource(show_spinner=True)
def get_model():
    try:
        model = tf.keras.models.load_model("models/mobilenetv2_best.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = get_model()

# ----------------------------
# Class names (from your dataset)
# ----------------------------
class_names = [
    "fish sea_food red_sea_bream",
    "fish sea_food horse_mackerel",
    "fish sea_food black_sea_sprat",
    "fish sea_food striped_red_mullet",
    "fish sea_food trout",
    "fish sea_food gilt_head_bream",
    "fish sea_food sea_bass",
    "fish sea_food shrimp"
]

# Clean labels for display
def clean_label(label):
    return label.replace("fish", "").replace("sea_food", "").replace("_", " ").strip().title()

# ----------------------------
# Preprocess image
# ----------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# ----------------------------
# Prediction
# ----------------------------
def predict(img):
    try:
        img_array = preprocess_image(img)
        preds = model.predict(img_array)[0]

        # Top-k results
        top_k = 3
        top_indices = preds.argsort()[-top_k:][::-1]
        results = [(class_names[i], preds[i] * 100) for i in top_indices]
        return results
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ----------------------------
# Streamlit UI
# ----------------------------
st.title(" Fish Image Classification")
st.write("Upload an image of a fish and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    from PIL import Image
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    if model:
        results = predict(img)

        if results:
            df = pd.DataFrame(results, columns=["Class", "Confidence (%)"])
            df["Class"] = df["Class"].apply(clean_label)
            df["Confidence (%)"] = df["Confidence (%)"].map(lambda x: f"{x:.2f}%")

            st.write("### Prediction Results")
            st.table(df)
