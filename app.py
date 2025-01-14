import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import RandomRotation
from PIL import Image

# Define a custom RandomRotation layer to handle potential argument issues
class CustomRandomRotation(RandomRotation):
    def __init__(self, **kwargs):
        kwargs.pop('value_range', None)  # Remove 'value_range' if present
        super().__init__(**kwargs)

# Register the custom layer
custom_objects = {'RandomRotation': CustomRandomRotation}

# Load the pre-trained model with custom objects
def load_trained_model():
    try:
        model = load_model('./model/mymodel.keras', custom_objects=custom_objects)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_trained_model()

# Flower categories
flower_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# Preprocess the image
def preprocess_image(image):
    try:
        input_image = Image.open(image).convert('RGB').resize((100, 100))
        input_image_array = np.array(input_image) / 255.0  # Normalize to [0, 1]
        input_image_exp_dim = np.expand_dims(input_image_array, axis=0)
        return input_image_exp_dim
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Classify uploaded images
def classify_images(image):
    try:
        input_image_exp_dim = preprocess_image(image)
        if input_image_exp_dim is None:
            return None, None
        
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])

        # Create a dictionary of prediction scores
        results_dict = {flower_names[i]: round(float(result[i] * 100), 2) for i in range(len(flower_names))}
        # Get the most likely class
        outcome = f'The Image belongs to "{flower_names[np.argmax(result)]}" with a confidence score of "{np.max(result)*100:.2f}%"'
        return outcome, results_dict
    except Exception as e:
        st.error(f"Classification failed: {str(e)}")
        return None, None

# Streamlit Sidebar for Navigation
page = st.sidebar.selectbox("Select Page", ["Home", "About"])

if page == "Home":
    # Home page content
    st.title("Flower Classification App 🌸")
    st.write("""
    Upload an image of a flower to classify it into one of the following categories:
    - Daisy
    - Dandelion
    - Rose
    - Sunflower
    - Tulip
    """)

    # File uploader
    uploaded_file = st.file_uploader('Upload a Flower Image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Classify the image
        outcome, results_dict = classify_images(uploaded_file)

        if outcome and results_dict:
            # Display classification results
            st.subheader("Classification Result")
            st.success(outcome)

            # Display prediction scores
            st.subheader("Prediction Scores")
            for flower, score in results_dict.items():
                st.write(f"**{flower}**")
                st.progress(score / 100)

elif page == "About":
    # About page content
    st.title("About This App")
    st.write("""
    This application uses a pre-trained deep learning model to classify images of flowers into one of five categories: Daisy, Dandelion, Rose, Sunflower, and Tulip.
    
    ### How to Use
    1. **Upload an Image**: Use the file uploader to upload an image of a flower.
    2. **View Results**: The app will classify the image and display the predicted flower type along with confidence scores.
    3. **Supported Formats**: JPG, PNG, and JPEG are supported.
    """)
