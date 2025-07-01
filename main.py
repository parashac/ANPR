from flask import Flask, request, redirect, render_template, jsonify, session, send_from_directory
import hashlib
import os
import cv2
from license_plate import detect_license_plate
from color_detection import detect_license_plate_color
from character_recognition import recognize_plate_characters
from database import create_user_in_db, get_user_from_db, is_username_taken, is_email_taken
from flask_cors import CORS  # Import CORS to handle cross-origin requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.secret_key = 'your_secret_key'  # Used for session management

# Folders for image uploads and cropped images
UPLOAD_FOLDER = "uploads"
CROPPED_FOLDER = "static/cropped"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["CROPPED_FOLDER"] = CROPPED_FOLDER

# Helper functions for password hashing and checking if user exists
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Routes for login, register, and home page

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'user_id' not in session:
        return redirect('/login')  # Redirect to login if not logged in

    if request.method == 'POST':
        image = request.files.get('file')
        if image:
            # Process the image here (use your API for detection)
            result = process_image(image)
            return jsonify(result)  # Send result as JSON

    return render_template('index.html')  # Render home page

# Image processing function to detect plate, color, and characters
def process_image(image):
    filename = image.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(file_path)
    image = cv2.imread(file_path)

    if image is None:
        return {"error": "Invalid image format or corrupted file"}

    plate_image, (x, y, w, h) = detect_license_plate(image)

    if plate_image is None:
        return {"error": "No license plate detected in the image"}

    cropped_plate = image[y:y+h, x:x+w]
    if cropped_plate.size == 0:
        return {"error": "Failed to crop license plate"}

    cropped_plate_filename = "cropped_" + filename
    cropped_plate_path = os.path.join(CROPPED_FOLDER, cropped_plate_filename)
    cv2.imwrite(cropped_plate_path, cropped_plate)

    plate_color, vehicle_category = detect_license_plate_color(cropped_plate)
    plate_number = recognize_plate_characters(cropped_plate)

    return {
        "plate_image_url": f"/cropped/{cropped_plate_filename}",
        "plate_color": plate_color,
        "vehicle_category": vehicle_category,
        "plate_number": plate_number
    }

# Route to serve cropped images
@app.route("/cropped/<filename>")
def get_cropped_image(filename):
    return send_from_directory(CROPPED_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
