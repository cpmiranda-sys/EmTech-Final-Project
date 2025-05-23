import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from joblib import load as joblib_load
from PIL import Image
import tempfile

# Load model and label encoder
@st.cache_resource
def load_model_and_encoder():
    model = load_model('best_model.keras')
    label_encoder = joblib_load('label_encoder.joblib')
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

# Prediction function
def predict_animal_from_image(image, model, label_encoder, threshold=0.7):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_array, axis=0).astype("float32") / 255.0

    preds = model.predict(img_array, verbose=0)
    max_prob = np.max(preds)
    pred_class_idx = np.argmax(preds)

    if max_prob < threshold:
        return "Unknown Animal", float(max_prob)
    else:
        pred_class = label_encoder.inverse_transform([pred_class_idx])[0]
        return pred_class, float(max_prob)

# Streamlit UI
st.set_page_config(page_title="Animal Classifier", layout="centered")
st.title("🐾 Animal Image Classifier")
st.write("Upload an image of an animal, and the model will predict its category.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            prediction, confidence = predict_animal_from_image(image, model, label_encoder)
        st.success(f"**Prediction:** {prediction}")
        st.info(f"**Confidence:** {confidence:.2f}")
