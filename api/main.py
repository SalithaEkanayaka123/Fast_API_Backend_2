import os
import pyrebase
import pathlib
import shutil

import aiofiles as aiofiles
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


from methods.audio_methods import preprocess_dataset, audio_labels, create_upload_file

app = FastAPI()

model_1 = tf.keras.models.load_model("../saved_models/1")
model_2 = tf.keras.models.load_model("../saved_models/2")
model_plesispa = tf.keras.models.load_model("../saved_models/plesispa beetle model")
model_whitefly = tf.keras.models.load_model("../saved_models/whitefly_model/1")
model_whitefly_2 = tf.keras.models.load_model("../saved_models/whitefly_model/2")
audio_model = tf.keras.models.load_model("../saved_models/audio_model/audio_model.h5")
CLASS_NAMES = ['Large ', 'Small', 'Unclear']
CLASS_NAMES_2 = ['apple1', 'apple2', 'apple3']
CLASS_NAMES_Whitefly = ['healthy_coconut', 'whietfly_infected_coconut']
CLASS_NAMES_Plesispa = ['clean', 'infected']

config = {
  "apiKey": os.environ['FIREBASE_API_KEY'],
  "authDomain": "esp32-file-database.firebaseapp.com",
  "databaseURL": "https://esp32-file-database-default-rtdb.asia-southeast1.firebasedatabase.app",
  "projectId": "esp32-file-database",
  "storageBucket": "esp32-file-database.appspot.com",
  "messagingSenderId": "474708048889",
  "appId": "1:474708048889:web:f4b097e89318fcfbb85b28",
  "measurementId": "G-0S4HR5CH8X"
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = model_2.predict(img_batch)
    predicted_class = CLASS_NAMES_2[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

@app.post("/predictwhitefly")
async def predict_whitefly(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = model_whitefly_2.predict(img_batch)
    predicted_class = CLASS_NAMES_Whitefly[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


@app.post("/predictplesispa")
async def predict_whitefly(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = model_plesispa.predict(img_batch)
    predicted_class = CLASS_NAMES_Plesispa[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

@app.post("/audio")
async def audio_predict(
    # Save the file.
    file: UploadFile = File(...)
):
    print(await create_upload_file(await file.read())['info'])
    audio = preprocess_dataset(await file.read())
    audio_batch = np.expand_dims(audio, 0)
    predictions = audio_model.predict(audio_batch)
    predicted_class = audio_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

@app.post("/upload-file/")
async def create_upload_file(file: UploadFile = File(...)):
    working_dir = pathlib.Path().absolute()
    file_location = f"{working_dir}\\..\\temp\\{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    audio = preprocess_dataset([str(file_location)])
    for spectrogram, label in audio.batch(1):
        predictions = audio_model(spectrogram)
        predicted_class = audio_labels[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
