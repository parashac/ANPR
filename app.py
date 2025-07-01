from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
from license_plate_detection import detect_license_plate
from color_detection import detect_license_plate_colordetect_license_plate_color
from character_recognition import recognize_plate_characters

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
CROPPED_FOLDER = "static/cropped"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["CROPPED_FOLDER"] = CROPPED_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Read image
    image = cv2.imread(file_path)
    if image is None:
        return jsonify({"error": "Invalid image format or corrupted file"}), 400

    # Detect license plate
    plate_image, (x, y, w, h) = detect_license_plate(image)

    # Crop license plate
    cropped_plate = image[y:y+h, x:x+w]
    if cropped_plate.size == 0:
        return jsonify({"error": "Failed to crop license plate"}), 400

    # Save cropped plate
    cropped_plate_filename = "cropped_" + filename
    cropped_plate_path = os.path.join(CROPPED_FOLDER, cropped_plate_filename)
    cv2.imwrite(cropped_plate_path, cropped_plate)

    # Detect plate color & vehicle category
    plate_color, vehicle_category = detect_license_plate_colordetect_license_plate_color(cropped_plate)

    # Recognize plate characters
    plate_number = recognize_plate_characters(cropped_plate)

    return jsonify({
        "plate_image_url": f"/cropped/{cropped_plate_filename}",
        "plate_color": plate_color,
        "vehicle_category": vehicle_category,
        "plate_number": plate_number
    })

# Route to serve cropped images
@app.route("/cropped/<filename>")
def get_cropped_image(filename):
    return send_from_directory(CROPPED_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
