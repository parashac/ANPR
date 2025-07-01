import cv2
import numpy as np
from keras.utils import img_to_array

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def preprocess_character(image):
    crop = cv2.resize(image, (64, 64))
    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    crop = crop.astype("float32") / 255.0
    crop = img_to_array(crop)
    crop = np.expand_dims(crop, axis=0)
    return crop
