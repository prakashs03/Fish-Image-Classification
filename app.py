import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Fish Image Classification", layout="wide")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource(show_spinner=True)
def get_model():
    model = tf.keras.models.load_model("models/mobilenetv2_best.h5", compile=False)
    return model

model = get_model()

# ----------------------------
# Define Classes (11 total)
# ----------------------------
class_names = [
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
def predict(img):
    try:
        img = image.load_img(img, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100

        return class_names[predicted_class], confidence, predictions[0]
    except Exception as e:
        return None, None, str(e)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title(" Fish Image Classification ")
st.write("Upload an image of a fish and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    label, confidence, all_preds = predict(uploaded_file)

    if label is not None:
        st.success(f"Prediction: **{label}** ({confidence:.2f}% confidence)")

        # Show full probability table
        st.subheader("Prediction Results")
        results = {class_names[i]: f"{all_preds[i]*100:.2f}%" for i in range(len(class_names))}
        st.table(results.items())
    else:
        st.error(f"Prediction error: {all_preds}")
