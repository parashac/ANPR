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

##  Demo Screenshots

###  Input Image
<img width="506" height="748" alt="Screenshot 2025-03-05 120252" src="https://github.com/user-attachments/assets/ab25d5a2-d8a7-49dd-9ac7-e90e589e019e" />

*This is an input image uploaded from files.*

---

###  Result Output
<img width="481" height="362" alt="Screenshot 2025-03-05 120419" src="https://github.com/user-attachments/assets/b807a2c2-281e-4f67-890d-342ec043f749" />

*This is the result of the above input image.*



