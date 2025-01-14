import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import (
    RandomRotation, RandomFlip, RandomZoom, 
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    Rescaling, InputLayer
)
from PIL import Image

# Constants
FLOWER_NAMES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
IMAGE_SIZE = (100, 100)
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png']
MODEL_PATH = './model/mymodel.keras'

def create_model():
    """Create the model architecture manually to match the saved weights"""
    model = Sequential([
        InputLayer(input_shape=(100, 100, 3)),
        Sequential([
            InputLayer(input_shape=(100, 100, 3)),
            RandomFlip("horizontal"),
            RandomRotation(factor=[-0.1, 0.1]),
            RandomZoom(height_factor=0.1)
        ]),
        Rescaling(scale=1./255),
        Conv2D(16, 3, activation='relu', padding='same'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu', padding='same'),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5)  # 5 flower classes
    ])
    return model

@st.cache_resource
def load_trained_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("Model file not found! Please ensure the model file exists.")
            return None

        # Create new model with matching architecture
        model = create_model()
        
        try:
            # Try to load weights directly
            model.load_weights(MODEL_PATH)
        except:
            try:
                # Alternative: try loading full model and transfer weights
                temp_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                model.set_weights(temp_model.get_weights())
            except Exception as e:
                st.error(f"Error transferring weights: {str(e)}")
                return None

        # Compile the model
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    try:
        input_image = Image.open(image).convert('RGB').resize(IMAGE_SIZE)
        input_image_array = np.array(input_image)
        input_image_exp_dim = np.expand_dims(input_image_array, axis=0)
        return input_image_exp_dim
    except IOError as e:
        st.error(f"Error opening image file: {str(e)}")
        return None
    except ValueError as e:
        st.error(f"Error processing image: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during image preprocessing: {str(e)}")
        return None

def classify_images(image, model):
    try:
        if model is None:
            st.error("Model not loaded. Cannot perform classification.")
            return None, None

        input_image_exp_dim = preprocess_image(image)
        if input_image_exp_dim is None:
            return None, None
        
        with st.spinner('Making predictions...'):
            predictions = model.predict(input_image_exp_dim, verbose=0)
            result = tf.nn.softmax(predictions[0])

        results_dict = {FLOWER_NAMES[i]: round(float(result[i] * 100), 2) 
                       for i in range(len(FLOWER_NAMES))}
        
        predicted_class = FLOWER_NAMES[np.argmax(result)]
        confidence = np.max(result) * 100
        outcome = f'The Image belongs to "{predicted_class}" with a confidence score of "{confidence:.2f}%"'
        
        return outcome, results_dict
    except Exception as e:
        st.error(f"Classification failed: {str(e)}")
        return None, None

def main():
    st.title("Flower Classification App 🌸")
    
    # Load model at startup
    model = load_trained_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        'Upload a Flower Image', 
        type=SUPPORTED_FORMATS
    )
    
    if uploaded_file is not None:
        # Display image and predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            
        with col2:
            outcome, results_dict = classify_images(uploaded_file, model)
            
            if outcome and results_dict:
                st.success(outcome)
                
                # Display prediction scores
                st.subheader("Confidence Scores")
                for flower, score in sorted(results_dict.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True):
                    st.write(f"{flower}: {score}%")
                    st.progress(score / 100)

if __name__ == "__main__":
    main()