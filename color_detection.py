import cv2
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)


# API function to detect license plate color and classify vehicle type
def detect_license_plate_colordetect_license_plate_color(image):
    # Define color ranges in HSV
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])

    blue_lower = np.array([110, 100, 100])
    blue_upper = np.array([130, 255, 255])

    green_lower = np.array([50, 100, 100])
    green_upper = np.array([70, 255, 255])

    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 50])

    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 50, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([40, 255, 255])

    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create masks for each color range using cv2.inRange()
    red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    black_mask = cv2.inRange(hsv_image, black_lower, black_upper)
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

    # Check for the presence of the color in the mask and classify
    if np.count_nonzero(red_mask) > 0:
        plate_color = "Red"
        plate_category = "Private Vehicle"
    elif np.count_nonzero(blue_mask) > 0:
        plate_color = "Blue"
        plate_category = "Diplomatic Vehicle"
    elif np.count_nonzero(green_mask) > 0:
        plate_color = "Green"
        plate_category = "Tourist Vehicle"
    elif np.count_nonzero(black_mask) > 0:
        plate_color = "Black"
        plate_category = "Public Vehicle"
    elif np.count_nonzero(white_mask) > 0:
        plate_color = "White"
        plate_category = "Government Vehicle"
    elif np.count_nonzero(yellow_mask) > 0:
        plate_color = "Yellow"
        plate_category = "Public Vehicle"
    else:
        plate_color = "Unknown"
        plate_category = "Unknown Vehicle Type"

    return plate_color, plate_category


# API endpoint to upload and classify vehicle plate color
@app.route('/classify_vehicle', methods=['POST'])
def classify_vehicle():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Load the image
        image = cv2.imread(file_path)

        if image is None:
            return jsonify({"error": "Could not load image"})

        # Classify the vehicle by plate color
        plate_color, plate_category = detect_license_plate_colordetect_license_plate_color(image)

        # Return both plate color and category separately
        return jsonify({"plate_color": plate_color, "plate_category": plate_category})


if __name__ == '_main_':
    app.run(debug=True)