import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model

def detect_license_plate(image):
    h, w, _ = image.shape
    img_resized = cv2.resize(image, (300, 300))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # Predict bounding box
    bbox = model.predict(img_array)[0]  # Format: [xc, yc, w, h]

    # Ensure output has 4 values
    if len(bbox) != 4:
        raise ValueError(f"Unexpected bbox shape: {bbox.shape}, expected (4,)")

    # Correct indexing
    xc, yc, bw, bh = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

    # Convert to integer coordinates
    x, y, bw, bh = int(xc - bw / 2), int(yc - bh / 2), int(bw), int(bh)
    x, y = max(0, x), max(0, y)

    return image, (x, y, bw, bh)

@tf.keras.utils.register_keras_serializable()
def custom_loss(y_true, y_pred):
    classification_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true[:, 0], y_pred[:, 0])
    regression_loss = tf.keras.losses.MeanSquaredError()(y_true[:, 1:], y_pred[:, 1:])
    return classification_loss + regression_loss

# Load the trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
model = load_model('final_anpr_model.keras')  # Load the trained model
