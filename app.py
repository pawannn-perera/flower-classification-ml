import os
import keras
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import pandas as pd

st.markdown("""
    <style>
    .header {
        font-size:50px; 
        color: Red;
        text-align:center;
        font-family: 'Courier New', Courier, monospace;
    }
    .subheader {
        font-size:20px;
        color: #4B7BFF;
        text-align:center;
        font-family: 'Helvetica', sans-serif;
    }
    .image-container {
        display: flex;
        justify-content: center;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# List of flower names for prediction
flower_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# Load the pre-trained model
model = load_model('./model/mymodel.keras')

def classify_images(image):
    # Open and preprocess the image for the model
    input_image = Image.open(image).resize((100, 100))
    input_image_array = np.array(input_image)
    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

    # Predict the class of the image
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])

    # Prepare results as a dictionary for all classes
    results_dict = {flower_names[i]: round(float(result[i] * 100), 2) for i in range(len(flower_names))}

    # Get the most likely class and format the result
    outcome = 'The Image belongs to "' + flower_names[np.argmax(result)] + '" with a confidence score of "'+ str(np.max(result)*100)[:5] + '%"'
    return outcome, results_dict

# Page selection
page = st.sidebar.selectbox("Select Page", ["Home", "About"])

if page == "Home":
    # Home page content
    st.markdown('<h1 class="header">Flower Classification</h1>', unsafe_allow_html=True)
    
    # Small description about flower types
    st.markdown("""
    ### Types of Flowers
    This application is designed to classify images of the following types of flowers:
    - **Daisy**
    - **Dandelion**
    - **Rose**
    - **Sunflower**
    - **Tulip**

    Please upload clear images of these flowers for the best classification results.
    """)
    
    # File uploader for image
    uploaded_file = st.file_uploader('Upload an Image')
    if uploaded_file is not None:
        # Display the image in the center
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(uploaded_file, width=250)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Classify the uploaded image
        outcome, results_dict = classify_images(uploaded_file)
        
        # Display the classification result beautifully
        st.markdown('<h2 class="subheader">Classification Result:</h2>', unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: green;'>{outcome}</h3>", unsafe_allow_html=True)
        
        # Display the results as progress bars for each flower category
        st.markdown('<h2 class="subheader">Prediction Scores:</h2>', unsafe_allow_html=True)
        for flower, score in results_dict.items():
            st.markdown(f"**{flower.capitalize()}**")
            st.progress(score / 100)

elif page == "About":
    # About page content
    st.sidebar.title("About")
    st.markdown("""
    ### How to Use
    1. **Upload an Image**: Click the 'Upload an Image' button to choose a flower (categories: Daisy, Dandelion, Rose, Sunflower, and Tulip) image from your device. The image should be in common formats such as JPG, PNG, or JPEG.
    2. **Image Display**: Once the image is uploaded, it will be displayed in the center of the app for your reference.
    3. **Classification**: The model will process the image and predict the flower type. The result, including the flower category and confidence score, will be shown prominently on the screen.
    4. **Prediction Scores**: Below the classification result, you will see prediction scores represented as progress bars for each flower category. These bars illustrate the confidence levels for each class, with a higher bar indicating a higher confidence score.""")
