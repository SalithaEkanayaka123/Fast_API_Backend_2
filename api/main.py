from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

model_1 = tf.keras.models.load_model("../saved_models/1")
model_2 = tf.keras.models.load_model("../saved_models/2")
CLASS_NAMES = ['Large ', 'Small', 'Unclear']
CLASS_NAMES_2 = ['apple1', 'apple2', 'apple3']


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

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
