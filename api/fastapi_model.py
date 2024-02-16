import os
import numpy as np
import cv2
import base64

from azure.storage.blob import BlobServiceClient
from fastapi import FastAPI
from uvicorn import run
from pydantic import BaseModel
from typing import Optional
from tensorflow import keras
from keras.utils import CustomObjectScope
from keras import backend as K
from dotenv import load_dotenv


# Do the command to launch locally :
# <= uvicorn fastapi_model:app --reload =>

app = FastAPI()
current_model_version = 1

class ImageInput(BaseModel):
    image: str
    model_version: Optional[int] = 1

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean((2. * intersection + smooth)/(union + smooth), axis=0)

@app.get('/')
def menu():
    return {"Welcome To":"Image Segmentation Model"}

@app.on_event("startup")
def load_model(version: int=current_model_version):
    global model

    # Use local Model
    # model_base_path = os.getenv('MODEL_BASE_PATH', '../model')
    # model_filename = f'{model_base_path}/best_final_model_v{version}.h5'

    # Use online Model
    load_dotenv('.azure-secret')
    storage_model_azure_key = os.getenv('MODEL_STORAGE_KEY')
    blob_service_client = BlobServiceClient.from_connection_string(storage_model_azure_key)
    blob_client = blob_service_client.get_blob_client("image-model", f"best_final_model_v{version}.h5")
    model_filename = f"best_final_model_v{version}.h5"
    with open(model_filename, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

    with CustomObjectScope({'dice_coef': dice_coef}):
        model = keras.models.load_model(model_filename)
    return model

@app.put("/model/version/{version}")
def set_model_version(version: int):
    global current_model_version
    current_model_version = version
    load_model(version)
    return {"message": f"Model version set to {version}"}

@app.get("/model/version")
def get_model_version():
    return {"model_version": current_model_version}

def preprocess(image):
    # Convert the 1D array back into an image
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # Ensure the image has 3 color channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(image, (256, 256))
    img = img / 255.0
    img = img.astype('float32')
    # Add an extra dimension for batch size
    img = np.expand_dims(img, axis=0)
    return img

@app.post('/predict')
def predict_sentiment(input: ImageInput, model_version: int=current_model_version):
    # Load the model with the specified version
    load_model(version=model_version)

    # Decode the image
    image_bytes = base64.b64decode(input.image)
    image = np.fromstring(image_bytes, np.uint8)

    process_image = preprocess(image)

    # Predict sentiment
    prediction = model.predict(process_image)

    # Serialize the prediction
    serialized_prediction = base64.b64encode(prediction).decode('utf-8')

    # Return prediction
    return {"mask": serialized_prediction}
    
if __name__ == "__main__":
    port = int(os.environ.get("WEBSITES_PORT", 8000))
    run("__main__:app", host="0.0.0.0", port=port, log_level="info")