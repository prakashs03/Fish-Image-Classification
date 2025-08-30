import streamlit as st
import tensorflow as tf
import numpy as np
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
        model = tf.keras.models.load_model("models/mobilenetv2_best_tf")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = get_model()

# ----------------------------
# Debug: Show model summary & output shape
# ----------------------------
if model is not None:
    st.write(" Model loaded successfully!")
    st.write("### Model Summary")
    model.summary(print_fn=lambda x: st.text(x))  # Print inside Streamlit
    st.write("**Model output shape:**", model.output_shape)

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(uploaded_file):
    try:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return img_array
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

# ----------------------------
# Main App
# ----------------------------
st.title(" Fish Image Classification")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    img_array = preprocess_image(uploaded_file)
    if img_array is not None:
        try:
            predictions = model(img_array, training=False).numpy()
            st.write("Raw Predictions:", predictions)

            predicted_class_index = int(np.argmax(predictions))
            
            #  Placeholder CLASS_NAMES – update once we know output shape
            CLASS_NAMES = ["Class1", "Class2", "Class3"]  

            if predicted_class_index < len(CLASS_NAMES):
                predicted_class = CLASS_NAMES[predicted_class_index]
                confidence = np.max(predictions)
                st.success(f"Predicted: {predicted_class} ({confidence:.2f} confidence)")
            else:
                st.error("Prediction error: CLASS_NAMES length does not match model output.")
        except Exception as e:
            st.error(f"Prediction error: {e}")
