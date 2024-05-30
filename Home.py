import streamlit as st
from PIL import Image 
from img_caption import img_caption, autoplay_audio
import numpy as np
from keras.applications.vgg16 import preprocess_input

st.subheader('Welcome to the Image Captioning Project')
uploaded_img = st.file_uploader('Upload the image you want to caption')

if uploaded_img is not None:
    image = Image.open(uploaded_img)
    st.image(image)
    
    # Call the img_caption function
    caption = img_caption(image)
    
    st.write('Caption:', caption)

    autoplay_audio('output.mp3')
    