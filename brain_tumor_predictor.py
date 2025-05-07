import tensorflow as tf
import numpy as np
import cv2

import tensorflow as tf
model = tf.keras.models.load_model('models/brain_tumor_detector.keras')  # ✅ correct path


def predict_brain_tumor(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (128, 128))  # 🛠 Resize correctly to 128x128
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img)

    # Show result
    if prediction[0][0] > 0.5:
        print("🚫 No Brain Tumor Detected")
    else:
        print("⚠️ Brain Tumor Detected")

# --- Test it ---
test_image_path = 'brain_dataset_detect/yes/Y2.jpg'
predict_brain_tumor(test_image_path)
