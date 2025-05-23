import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from PIL import Image
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Animal Classification App",
    page_icon="🐾",
    layout="wide"
)

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    """Load the trained model and label encoder"""
    try:
        # Load the trained model
        model = keras.models.load_model('best_model.keras')
        
        # Load the label encoder (you'll need to save this during training)
        # If you don't have it saved, you'll need to recreate it with the same classes
        try:
            with open('label_encoder.pkl', 'rb') as f:
                le = pickle.load(f)
        except FileNotFoundError:
            st.error("Label encoder not found. Please ensure 'label_encoder.pkl' exists.")
            st.info("You need to save your label encoder during training using: pickle.dump(le, open('label_encoder.pkl', 'wb'))")
            return None, None
            
        return model, le
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert RGB to BGR (OpenCV format)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Convert back to RGB for display and prediction
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    img_normalized = np.expand_dims(img_rgb, axis=0) / 255.0
    
    return img_normalized, img_rgb

def predict_animal(image, model, le, threshold=0.7):
    """Predict the animal in the image"""
    try:
        # Preprocess the image
        processed_img, display_img = preprocess_image(image)
        
        # Make prediction
        preds = model.predict(processed_img, verbose=0)
        max_prob = np.max(preds)
        pred_class_idx = np.argmax(preds)
        
        if max_prob < threshold:
            return "Unknown Animal", max_prob, preds[0], display_img
        else:
            pred_class = le.inverse_transform([pred_class_idx])[0]
            return pred_class, max_prob, preds[0], display_img
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None, None

def main():
    # App title and description
    st.title("🐾 Animal Classification App")
    st.markdown("Upload an image of an animal and let our AI model identify it!")
    
    # Load model
    model, le = load_model()
    
    if model is None or le is None:
        st.stop()
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.05,
        help="Minimum confidence required for classification"
    )
    
    # Display available classes
    with st.sidebar.expander("Available Animal Classes"):
        classes = le.classes_
        for i, animal_class in enumerate(classes):
            st.write(f"{i+1}. {animal_class}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an animal image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image in JPG, JPEG, PNG, or BMP format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict button
            if st.button("🔍 Classify Animal", type="primary"):
                with st.spinner("Analyzing image..."):
                    prediction, confidence, all_probs, processed_img = predict_animal(
                        image, model, le, confidence_threshold
                    )
                    
                    if prediction is not None:
                        # Store results in session state
                        st.session_state.prediction = prediction
                        st.session_state.confidence = confidence
                        st.session_state.all_probs = all_probs
                        st.session_state.processed_img = processed_img
                        st.session_state.classes = le.classes_
    
    with col2:
        st.header("Results")
        
        # Display results if available
        if hasattr(st.session_state, 'prediction') and st.session_state.prediction is not None:
            # Display processed image
            st.image(
                st.session_state.processed_img, 
                caption="Processed Image (224x224)", 
                use_column_width=True
            )
            
            # Display prediction
            if st.session_state.prediction == "Unknown Animal":
                st.warning(f"⚠️ **{st.session_state.prediction}**")
                st.info(f"Confidence: {st.session_state.confidence:.2%}")
                st.info("The model is not confident enough to classify this image. Try adjusting the confidence threshold or upload a clearer image.")
            else:
                st.success(f"🎯 **Predicted Animal: {st.session_state.prediction}**")
                st.info(f"Confidence: {st.session_state.confidence:.2%}")
            
            # Display top predictions
            st.subheader("Top 5 Predictions")
            top_indices = np.argsort(st.session_state.all_probs)[::-1][:5]
            
            for i, idx in enumerate(top_indices):
                animal_name = st.session_state.classes[idx]
                prob = st.session_state.all_probs[idx]
                
                # Create progress bar for visualization
                st.write(f"{i+1}. **{animal_name}**")
                st.progress(prob)
                st.write(f"Confidence: {prob:.2%}")
                st.write("---")
        
        else:
            st.info("👆 Upload an image and click 'Classify Animal' to see results")
    
    # Additional information
    st.markdown("---")
    st.markdown("### About this App")
    st.markdown("""
    This app uses a deep learning model trained to classify different animals. 
    The model analyzes uploaded images and provides predictions with confidence scores.
    
    **Tips for better results:**
    - Use clear, well-lit images
    - Ensure the animal is the main subject
    - Avoid heavily cropped or blurry images
    - Try different confidence thresholds if needed
    """)

if __name__ == "__main__":
    main()
