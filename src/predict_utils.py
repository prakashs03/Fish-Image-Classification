import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_image(model, img_path, target_size, class_indices):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    class_labels = {v: k for k, v in class_indices.items()}
    
    predicted_class = class_labels[np.argmax(preds)]
    confidence = float(np.max(preds))
    
    return predicted_class, confidence
