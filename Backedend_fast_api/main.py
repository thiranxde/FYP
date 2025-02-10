import os
import shutil
import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from typing import List
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from test import ModelManager
from pydantic import BaseModel
from io import BytesIO
import uuid
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from reinforcement import parse_feedback_and_image_prediction

app = FastAPI()

class ImageText(BaseModel):
    image: UploadFile

# Allow CORS for requests coming from http://localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)
# Mapping from class index to class name
class_names = {
    0: 'Body Dent',
    1: 'Bumper cracked',
    2: 'Bumper dent',
    3: 'Bumper Scratch',
    4: 'Door Dent',
    5: 'Door Scratch',
    6: 'Fender Damage',
    7: 'Glass shattered',
    8: 'Lamp Broken',
    9: 'Missing parts',
    10: 'Not Damaged',
    11: 'Other Scratches',
    12: 'Tire Flat',
}

@app.post("/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    try:
        # Process the uploaded images and make predictions
        predictions = []
        for uploaded_file in files:
            try:
                contents = await uploaded_file.read()
                img = Image.open(BytesIO(contents))
                img = img.resize((299, 299))  # Resize image to match model input size
                img_array = np.array(img) / 255.0  # Normalize image
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                model = ModelManager.get_model()

                prediction = model.predict(img_array)
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = class_names[predicted_class_index]
                predictions.append(predicted_class_name)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


@app.post("/predict-severity/")
async def predict_severity(files: UploadFile = File(...)):
    try:
        # Load the pre-trained severity prediction model
        model_path = "C:/IIT/4th year/FYP/Main mod/damage_classification_model_local.h5"
        severity_model = tf.keras.models.load_model(model_path)

        # Save the uploaded image temporarily
        with open("temp_image.jpg", "wb") as buffer:
            shutil.copyfileobj(files.file, buffer)
        
        # Preprocess the uploaded image
        img = Image.open("temp_image.jpg")
        img = img.resize((150, 150))  # Resize image to match model input size
        img_array = np.array(img) / 255.0  # Normalize image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make predictions
        predictions = severity_model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        
        # Assuming you have a mapping of class indices to class labels
        class_labels = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
        predicted_label = class_labels[predicted_class_index]
        
        # Remove the temporary image file
        os.remove("temp_image.jpg")
        
        return {"severity_prediction": predicted_label}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error predicting severity: {str(e)}"})
    

@app.post("/reinforcement/")
async def upload_image(is_accepted: str = Form(...), files: list[UploadFile] = File(...)):
    try:
        # Check if the accepted feedback is valid
        if is_accepted not in ["thumbs-up", "thumbs-down"]:
            raise HTTPException(status_code=400, detail="Invalid feedback. Accepted values are 'thumbs-up' or 'thumbs-down'.")
        
        for uploaded_file in files:
            try:
                contents = await uploaded_file.read()
                img = Image.open(BytesIO(contents))
                img = img.resize((299, 299))  # Resize image to match model input size
                img_array = np.array(img) / 255.0  # Normalize image
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                # Call function to process feedback and image prediction
                parse_feedback_and_image_prediction(img_array, is_accepted)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"message": "Feedback accepted"}

@app.post("/upload_number/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge detection
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Find location of number plate
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        raise HTTPException(status_code=400, detail="Number plate not found")

    # Masking the number plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Perform OCR using EasyOCR
    reader = easyocr.Reader(['en'])
    result = reader.readtext(gray)

    if result:
        text = result[0][-2]
    else:
        text = "OCR could not recognize any text"

    return {"extracted_text": text}

             