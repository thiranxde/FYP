import os
import shutil
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf

class ModelManager:
    _loaded_model = None

    @classmethod
    def load_model(cls):
        if cls._loaded_model is None:

                        # Define the InceptionV3 model architecture
            base_model = tf.keras.applications.InceptionV3(
                weights='imagenet',
                include_top=False,
                input_shape=(299, 299, 3)
            )

            # Freeze the layers of the pre-trained model
            base_model.trainable = False

            # Define the rest of your model architecture
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(13, activation='softmax')  # Change the number of units to match your output classes
            ])

            # Load the weights into the model
            model.load_weights("C:/IIT/4th year/FYP/Main mod/model_weights_epoch64_7.h5")  # Replace with the path to your model weights

            

            # Load the model from the specified path
            cls._loaded_model = model

            print("Model loaded successfully.")
        else:
            print("Model already loaded. Skipping loading.")

    @classmethod
    def get_model(cls):
        if cls._loaded_model is None:
            cls.load_model()
        return cls._loaded_model