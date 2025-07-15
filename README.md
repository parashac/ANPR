#  ANPR — Automatic Number Plate Recognition System

A full pipeline project for **Automatic Number Plate Recognition (ANPR)** using image processing and deep learning. This system detects number plates from images and classifies the license plate color, segments characters and recognizes them using a trained CNN model.

---

##  Features

-  **Number Plate Detection** using a fine-tuned CNN (VGG16-based)
-  **Plate Color Detection** using dominant color analysis
-  **Character Segmentation & Recognition** via CNN model
-  Organized module-wise directory: detection, color classification, recognition
-  Input support: Single image 
-  Trained model support (`.keras` weights included)
-  Web integration ready (Flask/Django compatible)

---

##  How It Works

1. **Image Input**
2. **License Plate Localization**
3. **Color Detection** (White, Yellow, Red, Green)
4. **Character Segmentation**
5. **CNN-based Character Recognition**
6. **Output Prediction**



