import os
import keras
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import pandas as pd

# Custom CSS to improve aesthetics
st.markdown("""
    <style>
    .header {
        font-size: 48px; 
        color: #FF4B4B;
        text-align: center;
        font-family: 'Arial', sans-serif;
        margin-top: 20px;
    }
    .subheader {
        font-size: 24px;
        color: #4B7BFF;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .description {
        font-size: 18px;
        text-align: center;
        font-family: 'Arial', sans-serif;
        color: #333;
        margin: 20px;
    }
    .image-container {
        display: flex;
        justify-content: center;
        padding: 20px;
    }
    .upload-button {
        display: block;
        margin: 20px auto;
        font-size: 18px;
        background-color: #4B7BFF;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-align: center;
        cursor: pointer;
    }
    .upload-button:hover {
        background-color: #357AE8;
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
    outcome = f'The Image belongs to "{flower_names[np.argmax(result)]}" with a confidence score of "{np.max(result) * 100:.2f}%"'
    return outcome, results_dict

# Page selection
page = st.sidebar.selectbox("Select Page", ["Home", "About"])

if page == "Home":
    # Home page content
    st.markdown('<h1 class="header">Flower Classification</h1>', unsafe_allow_html=True)
    
    # Small description about flower types
    st.markdown("""
    <div class="description">
    This application classifies images of flowers into five categories:
    - **Daisy**: White petals with a yellow center.
    - **Dandelion**: Bright yellow petals with a fluffy seed head.
    - **Rose**: Layered petals with a fragrant scent.
    - **Sunflower**: Large yellow petals and a dark center.
    - **Tulip**: Smooth, tulip-shaped petals in various colors.

    Upload clear images of these flowers for the best results.
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader for image
    uploaded_file = st.file_uploader('Upload an Image', label_visibility="collapsed")
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
    <div class="description">
    This application uses a pre-trained deep learning model to classify images of flowers into five categories: Daisy, Dandelion, Rose, Sunflower, and Tulip.

    ### How to Use
    1. **Upload an Image**: Click the 'Upload an Image' button to choose a flower image from your device. The image should be in common formats such as JPG, PNG, or JPEG.
    2. **Image Display**: Once uploaded, the image will be displayed for your reference.
    3. **Classification**: The model will predict the flower type and display the result with a confidence score.
    4. **Prediction Scores**: Prediction scores will be shown as progress bars for each flower category, representing the model's confidence in each prediction.

    Developed with TensorFlow and Keras, this model processes images through a neural network to make predictions. Feel free to upload multiple images to see how the model performs!
    </div>
    """, unsafe_allow_html=True)
