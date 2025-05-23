import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import joblib
from PIL import Image

# Constants
IMG_SIZE = (224, 224)

@st.cache_resource
def load_trained_model():
    try:
        # Check if files exist
        import os
        if not os.path.exists("best_model.keras"):
            st.error("Model file 'best_model.keras' not found!")
            return None, None
        
        if not os.path.exists("label_encoder.joblib"):
            st.error("Label encoder file 'label_encoder.joblib' not found!")
            return None, None
        
        model = load_model("best_model.keras")  # Update this to match your actual file name
        le = joblib.load("label_encoder.joblib")
        return model, le
    except Exception as e:
        st.error(f"Error loading model or label encoder: {str(e)}")
        return None, None

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = img_to_array(image) / 255.0  # Rescale as in training
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def main():
    st.title("Animal Classification App")
    st.write("Upload an image to classify the animal type.")
    
    model, le = load_trained_model()
    
    # Check if model and label encoder loaded successfully
    if model is None or le is None:
        st.stop()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        img_array = preprocess_image(image)
        preds = model.predict(img_array)
        predicted_index = np.argmax(preds)
        confidence = preds[0][predicted_index] * 100
        
        # Get class names from label encoder
        class_names = le.classes_
        
        st.write(f"**Prediction:** {class_names[predicted_index]}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        
        st.write("### All Class Probabilities:")
        for i in range(min(len(class_names), preds.shape[1])):
            st.write(f"{class_names[i]}: {preds[0][i]*100:.2f}%")

if __name__ == "__main__":
    main()
