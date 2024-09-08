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
        font-size:50px; 
        color: #FF4B4B;
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

# Main title
st.markdown('<h1 class="header">Flower Classification</h1>', unsafe_allow_html=True)

# About section
st.sidebar.title("About")
st.sidebar.info("""
This application uses a pre-trained deep learning model to classify images of flowers into five categories: Daisy, Dandelion, Rose, Sunflower, and Tulip. 
Upload an image of a flower, and the model will predict the category with the corresponding confidence score.
Developed using TensorFlow and Keras, the model processes images by resizing them and passing them through a neural network to make predictions.
""")

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
