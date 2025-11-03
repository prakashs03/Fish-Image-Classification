import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# Load trained model
MODEL_PATH = "models/mobilenetv2_best.h5"   # update if you saved with another name
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (same order as training)
class_labels = [
    "animal fish",
    "animal fish bass",
    "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream",
    "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet",
    "fish sea_food red_sea_bream",
    "fish sea_food sea_bass",
    "fish sea_food shrimp",
    "fish sea_food striped_red_mullet",
    "fish sea_food trout"
]

# Streamlit UI
st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload an image of a fish and the model will predict its category.")

# File uploader
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))  # match training size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    preds = model.predict(img_array)[0]  # Flatten predictions

    # Sort top 3 predictions
    top_3_idx = preds.argsort()[-3:][::-1]
    top_3_classes = [class_labels[i] for i in top_3_idx]
    top_3_confs = [preds[i] * 100 for i in top_3_idx]

    # Show uploaded image
    st.image(uploaded_file, caption="📸 Uploaded Image", use_column_width=True)

    # Display top prediction
    st.subheader("🎯 Prediction Result")
    st.success(f"**Predicted Class:** {top_3_classes[0]} ({top_3_confs[0]:.2f}%)")

    # Display top 3 predictions
    st.write("### 🔝 Top 3 Predictions:")
    for i in range(3):
        st.write(f"{i+1}. {top_3_classes[i]} — {top_3_confs[i]:.2f}%")

    # Plot bar chart
    fig, ax = plt.subplots()
    ax.barh(top_3_classes[::-1], top_3_confs[::-1], color='skyblue')
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Top 3 Predicted Classes")
    st.pyplot(fig)
