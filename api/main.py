# FastAPI related imports and Machine learning related.
import pathlib
import shutil
import cv2

import aiofiles as aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Authentication related imports.
from sqlalchemy.orm import Session

from auth import AuthHandler
from db.database import get_db
from db.model import User
from db.schema import CreateUsers
from schemas import AuthDetails

# PostgreSQL database import statements.


from methods.audio_methods import preprocess_dataset, audio_labels, create_upload_file

app = FastAPI()

# Demo array for maintain users.
"""
This array is for demo purpose, 
need to be replaced with the user table. (SQL)
"""
auth_handler = AuthHandler()
users = []

model_1 = tf.keras.models.load_model("../saved_models/1")
model_2 = tf.keras.models.load_model("../saved_models/2")
model_plesispa = tf.keras.models.load_model("../saved_models/Plesispa beetle model version 2")
model_whitefly = tf.keras.models.load_model("../saved_models/whitefly_model/1")
model_whitefly_2 = tf.keras.models.load_model("../saved_models/whitefly_model/2")
audio_model = tf.keras.models.load_model("../saved_models/audio_model/audio_model.h5")
CLASS_NAMES = ['Large ', 'Small', 'Unclear']
CLASS_NAMES_2 = ['apple1', 'apple2', 'apple3']
CLASS_NAMES_Whitefly = ['healthy_coconut', 'whietfly_infected_coconut']
CLASS_NAMES_Plesispa = ['clean', 'infected']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.get("/protect/ping")
async def ping(username=Depends(auth_handler.auth_wrapper)):
    return "Hello, I am alive (Protected)"


# User related endpoints. (With password hashing).
@app.post('/register', status_code=201)
def register(auth_details: AuthDetails):
    if any(x['username'] == auth_details.username for x in users):
        raise HTTPException(status_code=400, detail="Username is taken")
    hashed_password = auth_handler.get_password_hash(auth_details.password)
    users.append({
        'username': auth_details.username,
        'password': hashed_password
    })
    return

@app.post('/login')
def login (auth_details: AuthDetails):
    user = None
    for x in users:
        if x['username'] == auth_details.username:
            user = x
            break
    if(user is None) or (not auth_handler.verify_password(auth_details.password, user['password'])):
        raise HTTPException(status_code=401, detail='Invalid username and, or password')
    token = auth_handler.encode_token(user['username'])
    return {'token': token}



def read_file_as_image(data) -> np.ndarray:

    image = np.array(Image.open(BytesIO(data)))
    image = cv2.resize(image, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)
    #image = image.resize(image , (416, 416))
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
async def predict_plesispa(
    file: UploadFile = File(...)
):
   # data = data.resize((416, 416), Image.ANTIALIAS)
    image = read_file_as_image(await file.read())
    #
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

# Database INSERT  (Related to users) - Unprotected.
@app.post('/add-user')
def add_user(details: CreateUsers, db: Session=Depends(get_db)):
    to_create = User(
        title=details.title,
        description=details.description
    )
    db.add(to_create)
    db.commit()
    return {
        "success": True,
        "create_id": to_create.id
    }

# Database GET (Related to user) - Unprotected.
@app.get("/get-user")
def get_by_id(id: int,  db: Session = Depends(get_db())):
    return db.query(User).filter(User.id == id).first()

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
