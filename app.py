import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import RandomRotation, RandomFlip, RandomZoom
from PIL import Image

# Constants
FLOWER_NAMES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
IMAGE_SIZE = (100, 100)
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png']
MODEL_PATH = './model/mymodel.keras'

# Custom layer definitions to handle loading issues
class CustomRandomRotation(RandomRotation):
    def __init__(self, factor, **kwargs):
        # Remove problematic kwargs
        kwargs.pop('value_range', None)
        super().__init__(factor, **kwargs)

class CustomRandomFlip(RandomFlip):
    def __init__(self, mode='horizontal', **kwargs):
        super().__init__(mode, **kwargs)

class CustomRandomZoom(RandomZoom):
    def __init__(self, height_factor, **kwargs):
        super().__init__(height_factor, **kwargs)

# Register custom objects
custom_objects = {
    'RandomRotation': CustomRandomRotation,
    'RandomFlip': CustomRandomFlip,
    'RandomZoom': CustomRandomZoom
}

# Load the pre-trained model with custom objects
@st.cache_resource
def load_trained_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("Model file not found! Please ensure the model file exists.")
            return None

        # Configure Keras to use the legacy loading behavior
        tf.keras.utils.disable_interactive_logging()
        
        # Load model with custom objects
        model = load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            compile=False  # Prevent compilation issues
        )
        
        # Recompile the model with appropriate settings
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

# Preprocess the image
def preprocess_image(image):
    try:
        input_image = Image.open(image).convert('RGB').resize(IMAGE_SIZE)
        input_image_array = np.array(input_image) / 255.0  # Normalize to [0, 1]
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

# Classify uploaded images
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

        # Create a dictionary of prediction scores
        results_dict = {FLOWER_NAMES[i]: round(float(result[i] * 100), 2) 
                       for i in range(len(FLOWER_NAMES))}
        
        # Get the most likely class
        predicted_class = FLOWER_NAMES[np.argmax(result)]
        confidence = np.max(result) * 100
        outcome = f'The Image belongs to "{predicted_class}" with a confidence score of "{confidence:.2f}%"'
        
        return outcome, results_dict
    except Exception as e:
        st.error(f"Classification failed: {str(e)}")
        return None, None

def main():
    # Load model at startup
    model = load_trained_model()

    # Streamlit Sidebar for Navigation
    page = st.sidebar.selectbox("Select Page", ["Home", "About"])

    if page == "Home":
        # Home page content
        st.title("Flower Classification App 🌸")
        st.write("""
        Upload an image of a flower to classify it into one of the following categories:
        """)
        
        # Display flower categories as bullets
        for flower in FLOWER_NAMES:
            st.write(f"- {flower}")

        # File uploader
        uploaded_file = st.file_uploader(
            'Upload a Flower Image', 
            type=SUPPORTED_FORMATS
        )
        
        if uploaded_file is not None:
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Display the uploaded image
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

            with col2:
                with st.spinner('Classifying image...'):
                    # Classify the image
                    outcome, results_dict = classify_images(uploaded_file, model)

                if outcome and results_dict:
                    # Display classification results
                    st.subheader("Classification Result")
                    st.success(outcome)

            if results_dict:
                # Display prediction scores
                st.subheader("Prediction Scores")
                for flower, score in sorted(results_dict.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True):
                    col1, col2 = st.columns([3, 7])
                    with col1:
                        st.write(f"**{flower}**")
                    with col2:
                        st.progress(score / 100)
                        st.write(f"{score}%")

    elif page == "About":
        # About page content
        st.title("About This App")
        st.write("""
        This application uses a pre-trained deep learning model to classify images 
        of flowers into five different categories.
        
        ### How to Use
        1. **Upload an Image**: Use the file uploader to upload an image of a flower.
        2. **View Results**: The app will classify the image and display the predicted 
           flower type along with confidence scores.
        3. **Supported Formats**: JPG, PNG, and JPEG are supported.
        
        ### Technical Details
        - The model is built using TensorFlow/Keras
        - Images are preprocessed to size 100x100 pixels
        - Predictions are shown with confidence scores for all categories
        
        ### Tips for Best Results
        - Use clear, well-lit images
        - Ensure the flower is the main subject in the image
        - Images should be in focus and not blurry
        """)

if __name__ == "__main__":
    main()