import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Fish Image Classification", layout="wide")

# ----------------------------
# Load the Model
# ----------------------------
@st.cache_resource(show_spinner=True)
def get_model():
    model = tf.keras.models.load_model("models/mobilenetv2_best.h5", compile=False)
    return model

model = get_model()

# ----------------------------
# Class Labels
# ----------------------------
CLASS_NAMES = [
    "animal fish",
    "animal fish bass",
    "black sea sprat",
    "gilt head bream",
    "hourse mackerel",
    "red mullet",
    "red sea bream",
    "sea bass",
    "shrimp",
    "striped red mullet",
    "trout"
]

# ----------------------------
# Prediction Function
# ----------------------------
def predict_image(uploaded_file):
    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get predictions
    preds = model.predict(img_array)
    preds = preds[0]  # Flatten the array

    # Get top 3 predictions
    top_3_idx = preds.argsort()[-3:][::-1]
    top_3_classes = [CLASS_NAMES[i] for i in top_3_idx]
    top_3_confs = [preds[i] * 100 for i in top_3_idx]

    # Return top prediction and top 3
    return top_3_classes, top_3_confs, img

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.title("🐟 Fish Image Classification")
st.write("Upload a fish image and let the model identify its species!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    top_classes, top_confs, img = predict_image(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        # ✅ Display directly from uploaded file buffer (stable for Streamlit Cloud)
        st.image(uploaded_file, caption="📸 Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("🎯 Prediction Results")

        st.success(f"**Predicted Class:** {top_classes[0]} ({top_confs[0]:.2f}%)")

        st.write("### 🔝 Top 3 Predictions:")
        for i in range(3):
            st.write(f"{i+1}. {top_classes[i]} — {top_confs[i]:.2f}%")

        # Bar chart for top 3 predictions
        fig, ax = plt.subplots()
        ax.barh(top_classes[::-1], top_confs[::-1], color='skyblue')
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Top 3 Predicted Classes")
        st.pyplot(fig)
