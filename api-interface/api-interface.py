import zipfile
import io
import streamlit as st
import requests
import base64
import cv2
import numpy as np


# Define the Online API endpoint :
API_ENDPOINT = "https://cars-api-sentiment.azurewebsites.net/predict"

# Create a Streamlit app
st.title('Interface for Image Segmentation API')

# Upload the zip file
uploaded_file = st.file_uploader("Choose a ZIP file", type=['zip'])

if uploaded_file is not None:
    # Open the zip file
    with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue()), 'r') as z:
        # Get the list of files
        file_list = [f for f in z.namelist() if '_leftImg8bit.png' in f]
        mask_list = [f for f in z.namelist() if '_color.png' in f]
        # Select an image
        selected_image = st.selectbox('Select an image', file_list)
        selected_mask = st.selectbox('Select a mask', mask_list)

        # Open the selected image
        with z.open(selected_image) as f:
            # Read the image file
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            # Display the uploaded image
            st.image(img, caption='Original Image', channels="BGR")

        # Open the selected mask
        with z.open(selected_mask) as f:
            # Read the mask file
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            mask = cv2.imdecode(file_bytes, 1)

            # Display the original mask
            st.image(mask, caption='Original Mask', channels="BGR")

        # Serialize the image
        _, img_encoded = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(img_encoded).decode()

        if st.button('Predict'):
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
            original_shape = img.shape[:2]
            prediction_resized = cv2.resize(prediction, original_shape[::-1])
            
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

            colored_prediction = np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)

            for i, (label, color) in enumerate(label_to_num.items()):
                colored_prediction[prediction_resized[..., i] > 0.45] = color

            st.image(colored_prediction, caption='Colored API Prediction', channels="BGR")