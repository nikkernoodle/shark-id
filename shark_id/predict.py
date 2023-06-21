# funktion which takes the image as input
# give me back preprocessed image


from tensorflow import expand_dims
from tensorflow.keras import models
from PIL import Image
import cv2
import numpy as np


def preprocess_image(image):
    #image = Image.open(image_path)
    image = image.resize((224, 224))
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    image = np.array(image)
    return image


# funktion which loads the model
# gave back a model variable

def load_model(model_path = 'raw_data/model/model_b3.h5'):
    model = models.load_model(model_path)
    return model


# funktion for prediction with both funktion together (the preprocessed image funktion, model funktion)
#give back the prediction

def predict_image(image):
    model_path = 'raw_data/model/model_b3.h5'
    preprocessed_img = preprocess_image(image)
    loaded_model = load_model(model_path)
    img = expand_dims(preprocessed_img, axis=0)
    preds = loaded_model.predict(img)
    return preds


def predict_image_model(image, model):
    preprocessed_img = preprocess_image(image)
    img = expand_dims(preprocessed_img, axis=0)
    preds = model.predict(img)
    return preds
