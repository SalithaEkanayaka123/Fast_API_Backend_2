# FastAPI related imports and Machine learning related.
import io
import pathlib
import shutil
import cv2
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
import os
# Authentication related imports.
from sqlalchemy.orm import Session

from Main.api.auth import AuthHandler
from Main.db.database import get_db, engine
from Main.db.model import User
from Main.db.schema import CreateUsers, Request, Response, RequestUser
from Main.db.crud import insert_user, get_by_name,remove_user, update_user
from Main.api.schemas import AuthDetails

from io import BytesIO


# PostgreSQL database import statements. (Autogenerate tables).
import Main.db.model as model

model.Base.metadata.create_all(bind=engine)

from Main.methods.audio_methods import preprocess_dataset, audio_labels, create_upload_file

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://192.168.8.102:5000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auth_handler = AuthHandler()

# Store all the username from the db when application start.
users = []

model_plesispa = tf.keras.models.load_model("./Main/saved_models/Plesispa beetle model version 2")
model_whitefly_2 = tf.keras.models.load_model("./Main/saved_models/whitefly_model/2")
audio_model = tf.keras.models.load_model("./Main/saved_models/audio_model/audio_model.h5")
CLASS_NAMES_Whitefly = ['healthy_coconut', 'whietfly_infected_coconut']
CLASS_NAMES_Plesispa = ['clean', 'infected']

# ----------------------------------------------------------------------------
# Created By  : Anawaratne M.A.N.A.
# Created Date: 2022/7/18
# version ='1.0'
# ---------------------------------------------------------------------------
""" JWT Auth related endpoints """


# ---------------------------------------------------------------------------
# Status : Work in Progress.
# ---------------------------------------------------------------------------

# User registration (With password hashing).
@app.post('/register', status_code=201)
def register(auth_details: CreateUsers, db: Session = Depends(get_db)):
    db_user = get_by_name(name=auth_details.username, db=db)
    if db_user:
        raise HTTPException(status_code=400, detail="Username is taken")
    hashed_password = auth_handler.get_password_hash(auth_details.password)
    auth_details.password = hashed_password
    response = insert_user(details=auth_details, db=db)
    return response


# User login with JWT auth.
@app.post('/login')
def login(auth_details: AuthDetails, db: Session = Depends(get_db)):
    user = None
    db_user = get_by_name(name=auth_details.username, db=db)
    if db_user:
        user = User(
            name=db_user.name,
            hash_password=db_user.hash_password
        )
        print(user.hash_password)
    if (user is None) or (not auth_handler.verify_password(auth_details.password, user.hash_password)):
        raise HTTPException(status_code=401, detail='Invalid username and, or password')
    token = auth_handler.encode_token(user.name)
    return {'token': token}


# Testing endpoint with JWT (Protected routes).
@app.get("/ping")
async def ping(db: Session = Depends(get_db)):
    return "Something"


# Testing endpoint with JWT (Protected routes).
@app.get("/protect/ping", dependencies=[Depends(auth_handler.auth_wrapper)])
async def ping():
    return "Hello, I am alive (Protected)"


# ----------------------------------------------------------------------------
# Created By  : @Team
# Created Date: -
# version ='1.0'
# ---------------------------------------------------------------------------
""" Machine learning model related endpoints """


# ---------------------------------------------------------------------------
# Status : Work in Progress.
# ---------------------------------------------------------------------------

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    image = cv2.resize(image, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)
    # image = image.resize(image , (416, 416))
    return image



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


#
#
#--------------------------------------------------------------------------------------------

@app.post("/predictwhiteflystring")
async def predict_whitefly_str(request: Request):
    image_url = request.image_url
    filename = image_url.split("/")[-1]
    filename = filename.split("?alt=media&token")[0]
    filename = filename.split("-")[0]
    filename = filename.split("%")[1] + '.jpeg'

    r = requests.get(image_url, stream=True)

    if r.status_code == 200:
        try:
            print('done')
            r.raw.decode_content = True
            print(r.raw.decode_content)
            basepath = os.path.dirname(__file__)
            print(basepath)
            with open(basepath + filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
            print('Image successfully Downloaded: ' + filename)

            file_like = open(basepath + filename, "rb")
            print(file_like.read())

            byteImgIO = io.BytesIO()
            byteImg = Image.open(basepath + filename)
            byteImg.save(byteImgIO, "JPEG")
            byteImgIO.seek(0)
            byteImg = byteImgIO.read()


            image = read_file_as_image(byteImg)
            img_batch = np.expand_dims(image, 0)

            predictions = model_whitefly_2.predict(img_batch)
            predicted_class = CLASS_NAMES_Whitefly[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            return {
                'class': predicted_class,
                'confidence': float(confidence)
            }
        except:
            return {
                'class': "unknown",
                'confidence': "unknown"
            }
    else:
        return {
            'class': "unknown",
            'confidence': "unknown"
        }


#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

@app.post("/predictplesispastr")
async def predict_plesispa_str(request: Request):
    image_url = request.image_url
    filename = image_url.split("/")[-1]
    filename = filename.split("?alt=media&token")[0]
    filename = filename.split("-")[0]
    filename = filename.split("%")[1] + '.jpeg'

    r = requests.get(image_url, stream=True)

    if r.status_code == 200:
        try:
            print('done')
            r.raw.decode_content = True
            print(r.raw.decode_content)
            with open( filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
            print('Image successfully Downloaded: ' + filename)

            file_like = open(filename, "rb")
            print(file_like.read())

            byteImgIO = io.BytesIO()
            byteImg = Image.open(""+filename)
            byteImg.save(byteImgIO, "JPEG")
            byteImgIO.seek(0)
            byteImg = byteImgIO.read()


            image = read_file_as_image(byteImg)

            img_batch = np.expand_dims(image, 0)

            predictions = model_plesispa.predict(img_batch)
            predicted_class = CLASS_NAMES_Plesispa[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            return {
                'class': predicted_class,
                'confidence': float(confidence)
            }
        except:
            return {
                'class': "unknown",
                'confidence': "unknown"
            }
    else:
        return {
            'class': "unknown",
            'confidence': "unknown"
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


# ----------------------------------------------------------------------------
# Created By  : Anawaratne M.A.N.A.
# Created Date: 2022/7/19
# version ='1.0'
# ---------------------------------------------------------------------------
""" Database CRUD related endpoints """


# ---------------------------------------------------------------------------
# Status : Work in Progress.
# ---------------------------------------------------------------------------

# Database INSERT  (Related to users) - Testing method / Hash the password.
@app.post('/add-user')
def add_user(details: CreateUsers, db: Session = Depends(get_db)):
    to_create = User(
        name=details.username,
        hash_password=details.password
    )
    db.add(to_create)
    db.commit()
    return {
        "success": True,
        "create_id": to_create.id
    }


# Database GET (Related to user)
@app.get("/get-user")
def get_by_id(id: int, db: Session = Depends(get_db)):
    return db.query(User).filter(User.id == id).first()


# Database GET-ALL (Related to user) - TEST
@app.get("/get-users")
def get_by_id(db: Session = Depends(get_db)):
    return db.query(User).offset(0).limit(100).all()


# Database GET-ALL (Related to user) - TEST
@app.get("/get-users-name")
def get_by_name(name: str, db: Session = Depends(get_db)):
    return db.query(User).filter(User.name == name).first()


# Database DELETE (Related to user)
@app.post('/delete-user')
def delete_user(id: int, db: Session = Depends(get_db)):
    remove_user(db, id=id)
    return Response(status="Ok", code="200", message="Successfully delete data").dict(exclude_none=True)

# Database UPDATE (Related to user) - @Implement here.
@app.post('/update-user')
def update_user(request: RequestUser, db: Session = Depends(get_db)):
    hashed_password = auth_handler.get_password_hash(request.password)
    _user = update_user(db, id=request.id,
                             username=request.username, password=hashed_password)
    return Response(status="Ok", code="200", message="Successfully update data", result=_user)

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
