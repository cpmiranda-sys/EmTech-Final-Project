import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Constants
IMG_SIZE = (150, 150)
CLASS_NAMES = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']  

@st.cache_resource
def load_trained_model():
    model = load_model("best_model.keras")
    return model

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = img_to_array(image) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def main():
    st.title("Animal Classification Predictor")
    st.write("Upload an image to identify an Animal.")

    model = load_trained_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = load_img(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(image)
        preds = model.predict(img_array)
        predicted_index = np.argmax(preds)
        confidence = preds[0][predicted_index] * 100

        st.write(f"**Prediction:** {CLASS_NAMES[predicted_index]}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.write("### All Class Probabilities:")
        for i in range(min(len(CLASS_NAMES), preds.shape[1])):
            st.write(f"{CLASS_NAMES[i]}: {preds[0][i]*100:.2f}%")

if __name__ == "__main__":
    main()
