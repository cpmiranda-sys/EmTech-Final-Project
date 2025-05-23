import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from joblib import load as joblib_load
from PIL import Image
import numpy as np
import cv2

# Constants
IMG_SIZE = (224, 224)
THRESHOLD = 0.7

@st.cache_resource
def load_trained_model_and_encoder():
    model = load_model("animal_classifier.h5")
    label_encoder = joblib_load("label_encoder.joblib")
    return model, label_encoder

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    image_array = cv2.resize(image_array, IMG_SIZE)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_array = img_to_array(image_array) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def main():
    st.title("🐾 Animal Image Classifier")
    st.write("Upload an image to identify the type of animal.")

    model, label_encoder = load_trained_model_and_encoder()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(image)
        preds = model.predict(img_array)
        predicted_index = np.argmax(preds)
        confidence = preds[0][predicted_index]

        if confidence < THRESHOLD:
            predicted_label = "Unknown Animal"
        else:
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        st.write(f"**Prediction:** {predicted_label}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

        st.write("### All Class Probabilities:")
        class_names = label_encoder.classes_
        for i, prob in enumerate(preds[0]):
            st.write(f"{class_names[i]}: {prob * 100:.2f}%")

if __name__ == "__main__":
    main()
