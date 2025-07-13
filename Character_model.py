import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths (Make sure these match your dataset location)
train_dir = "D:/ANPR project/Dataset/last_option/character/data/train"
test_dir = "D:/ANPR project/Dataset/last_option/character/data/test"

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    validation_split=0.15
)

val_datagen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=10,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=False,
                                 validation_split=0.15)

training_set = train_datagen.flow_from_directory(
    train_dir, target_size=(64, 64), batch_size=32, class_mode='categorical', subset='training'
)

val_set = val_datagen.flow_from_directory(
    train_dir, target_size=(64, 64), batch_size=32, class_mode='categorical', subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_set = test_datagen.flow_from_directory(
    test_dir, target_size=(64, 64), batch_size=32, class_mode='categorical', shuffle=False
)

# CNN Model
cnn = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]),
    MaxPooling2D(pool_size=2, strides=2),

    Conv2D(32, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2, strides=2),

    Conv2D(32, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2, strides=2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(34, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()  # Fix here: use cnn instead of model

# Model Checkpoints
checkpoint_dir = "D:/ANPR project/code/python/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Save the model weights after each epoch
checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.keras')
checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, save_freq='epoch', verbose=1)

# Early Stopping (to stop if the validation loss is not improving)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the Model
cnn_model = cnn.fit(training_set, validation_data=val_set, epochs=17, batch_size=32,
                    callbacks=[early_stopping, checkpoint])

# Plot Training History
history = cnn_model.history

epochs = range(1, len(history['accuracy']) + 1)  # Set epochs based on the length of the history

plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs, history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.xticks(epochs)
plt.legend()
plt.grid()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs, history['loss'], label='Train Loss', marker='o')
plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.xticks(epochs)
plt.legend()
plt.grid()

plt.show()
