import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
path = 'brain_dataset_detect'
categories = ['yes', 'no']

# Data lists
data = []
labels = []

# Read images
for category in categories:
    folder_path = os.path.join(path, category)
    label = categories.index(category)  # yes -> 0, no -> 1

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize images
            data.append(img)
            labels.append(label)

# Convert to numpy arrays
data = np.array(data) / 255.0  # Normalize pixel values
labels = np.array(labels)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print("Data Loaded Successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build CNN model
model = Sequential()

# 1st Convolution layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3rd Convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten + Fully Connected Layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # To prevent overfitting
model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Save the trained model
model.save('brain_tumor_detector.keras')

print("Model trained and saved successfully!")
