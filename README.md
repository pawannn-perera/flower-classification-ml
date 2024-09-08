# Flower Classification App

Welcome to the Flower Classification App! This application allows you to classify images of flowers into five categories. The model identifies the type of flower and provides confidence scores for each category.

## Features

- **Flower Classification**: Upload an image of a flower, and the app will classify it into one of the following categories:
  - Daisy
  - Dandelion
  - Rose
  - Sunflower
  - Tulip

- **Prediction Scores**: View the confidence scores for each flower category as progress bars.

## How to Use

1. **Select Page**: Choose between the "Home" and "About" pages from the sidebar.
   
2. **Home Page**:
   - **Upload an Image**: Click the 'Upload an Image' button to select a flower image from your device. Supported formats include JPG, PNG, and JPEG.
   - **View Image**: The uploaded image will be displayed in the center of the app.
   - **Classify Image**: The model will process the image and provide a classification result along with a confidence score. The result will be shown prominently on the screen.
   - **Prediction Scores**: Below the classification result, you will see progress bars representing the confidence scores for each flower category.

3. **About Page**:
   - Provides information about the app, how it works, and details about the model used for classification.

## Technical Details

- **Image Processing**: Uploaded images are resized to 100x100 pixels before being processed by the model.
- **Libraries Used**:
  - TensorFlow
  - Keras
  - Streamlit
  - PIL (Python Imaging Library)
  - NumPy
  - pandas
