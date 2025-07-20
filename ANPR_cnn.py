
import os
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
import tensorflow as tf

IMAGE_SIZE = 300  

# Define dataset paths
train_img_dir = "D:/ANPR project/Dataset/last_option/train/images1"  # Training images folder
train_xml_dir = "D:/ANPR project/Dataset/last_option/train/labels1" # Training annotations folder

val_img_dir = "D:/ANPR project/Dataset/last_option/val/images" # Validation images folder
val_xml_dir = "D:/ANPR project/Dataset/last_option/val/labels" # Validation annotations folder

# Step 4: Function to parse XML annotations
def parse_xml(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Convert to (xcen, ycen, width, height)
        xcen = (xmin + xmax) / 2.0
        ycen = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin

        # Normalize values to 0-1
        xcen /= img_width
        ycen /= img_height
        w /= img_width
        h /= img_height

        return [xcen, ycen, w, h]

    return None  # If no valid bounding box is found

def load_dataset(image_dir, xml_dir):
    X, y = [], []

    data_path = os.path.join(image_dir, "*.jpg") 
    image_files = sorted(glob.glob(data_path))  
    for img_file in image_files:
        img = cv2.imread(img_file)
        height, width = img.shape[:2]
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # Get corresponding XML file
        xml_file = os.path.join(xml_dir, os.path.basename(img_file).replace(".jpg", ".xml"))

        if os.path.exists(xml_file):
            bbox = parse_xml(xml_file, width, height)
            if bbox:
                X.append(img_resized)
                y.append(bbox)

    return np.array(X, dtype="float32") / 255.0, np.array(y, dtype="float32")

X_train, y_train = load_dataset(train_img_dir, train_xml_dir)
print(f" Training Dataset Loaded: {len(X_train)} images")

X_val, y_val = load_dataset(val_img_dir, val_xml_dir)
print(f"Validation Dataset Loaded: {len(X_val)} images")

model = Sequential()
model.add(VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))  # Dropout to reduce overfitting
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))  # Dropout layer
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))  # Dropout layer
model.add(Dense(4, activation="sigmoid"))  # 4 outputs (xmin, ymin, xmax, ymax)

# Freeze VGG16 layers
model.layers[0].trainable = False

# Compile model with Mean Squared Error and Adam optimizer
model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.0001), metrics=["mae", "mse", "accuracy"])

# Model Summary
model.summary()

#Custom Callback to Track Average Accuracy and Loss per Epoch
class AverageMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Calculate average metrics after each epoch
        avg_train_loss = np.mean(logs['loss'])
        avg_train_mae = np.mean(logs['mae'])
        avg_train_mse = np.mean(logs['mse'])
        avg_train_acc = np.mean(logs['accuracy'])

        avg_val_loss = np.mean(logs['val_loss'])
        avg_val_mae = np.mean(logs['val_mae'])
        avg_val_mse = np.mean(logs['val_mse'])
        avg_val_acc = np.mean(logs['val_accuracy'])

        print(f"\nEpoch {epoch + 1} - avg_train_loss: {avg_train_loss:.4f}, avg_train_mae: {avg_train_mae:.4f}, "
              f"avg_train_mse: {avg_train_mse:.4f}, avg_train_acc: {avg_train_acc:.4f}")
        print(f"Validation avg_loss: {avg_val_loss:.4f}, Validation avg_mae: {avg_val_mae:.4f}, "
              f"Validation avg_mse: {avg_val_mse:.4f}, Validation avg_acc: {avg_val_acc:.4f}")
#Train the model
train_history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=12,
    verbose=1,
    callbacks=[AverageMetricsCallback()]  # Custom callback for average metrics
)

# Plot training & validation accuracy and loss
def plot_scores(history):
    epochs = range(len(history.history["mae"]))

    plt.figure(figsize=(12, 5))

    # Plot Mean Absolute Error (Accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["accuracy"], "b", label="Training ACCURACY")
    plt.plot(epochs, history.history["val_accuracy"], "r", label="Validation ACCURACY")
    plt.title("ACCURACY")
    plt.legend()

    # Plot Mean Squared Error (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["mse"], "b", label="Training MSE")
    plt.plot(epochs, history.history["val_mse"], "r", label="Validation MSE")
    plt.title("Mean Squared Error")
    plt.legend()

    plt.show()

plot_scores(train_history)

#Save the trained model
model.save("final_anpr_model.keras")
print(" Model saved as 'final_anpr_model.keras'")
