import streamlit as st
import requests
import base64
import cv2
import numpy as np


# Go with cd on the 'api-interface' folder
# Start the local server with the "streamlit run api-interface.py"

# Define the Local API endpoint :
# API_ENDPOINT = 'http://localhost:8000/predict'

# Define the Online API endpoint :
API_ENDPOINT = "https://cars-api-sentiment.azurewebsites.net/predict"

# Create a Streamlit app
st.title('Interface for Image Segmentation API')

# Upload the image
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(img, channels="BGR")

    # Serialize the image
    _, img_encoded = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(img_encoded).decode()

    # Send the image to the API
    response = requests.post(API_ENDPOINT, json={'image': img_base64})

    # Get the prediction from the API response
    prediction_base64 = response.json()['mask']

    # Deserialize the prediction
    prediction_bytes = base64.b64decode(prediction_base64)
    prediction = np.frombuffer(prediction_bytes, dtype=np.float32)

    # Reshape the prediction to its original shape
    prediction_shape = (256, 256, 8) 
    prediction = prediction.reshape(prediction_shape)

    # Get the original image shape
    original_shape = img.shape[:2]

    # Resize the prediction to the original image shape
    prediction_resized = cv2.resize(prediction, original_shape[::-1])

    # Define colors for each category
    label_to_num = {
        'flat': [214, 112, 218],         # Magenta for 'flat'
        'human': [0, 0, 255],            # Red for 'human'
        'vehicle': [255, 0, 0],          # Blue for 'vehicle'
        'construction': [105, 105, 105], # Gray for 'construction'
        'object': [0, 215, 255],         # Yellow for 'object'
        'nature': [50, 205, 50],         # Green for 'nature'
        'sky': [235, 206, 135],          # Light blue for 'sky'
        'void': [0, 0, 0]                # Black for 'void'
    }

    # Create an empty image with the same shape as the original image
    colored_prediction = np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)

    # Assign colors based on prediction
    for i, (label, color) in enumerate(label_to_num.items()):
        colored_prediction[prediction_resized[..., i] > 0.45] = color

    # Display the colored prediction
    st.image(colored_prediction, caption='Colored API Prediction', channels="BGR")