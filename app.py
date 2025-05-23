import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('animal_classification.h5')
  return model
model=load_model()
st.write("""
# Alzheimer MRI Detection"""
)
file=st.file_uploader("Choose an Animal Image photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np

# def import_and_predict(image_data,model):
#     size=(128,128)
#     image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
#     img=np.asarray(image)
#     img_reshape=img[np.newaxis,...]
#     prediction=model.predict(img_reshape)
#     return prediction

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (224, 224)
    
    # Resize the image to the expected input shape of the model
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    # Convert the image to grayscale if necessary
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Reshape the image to add a channel dimension
    img_reshape = img.reshape((1,) + img.shape + (1,))

    # Make predictions using the Keras model
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("Please upload an image file")
