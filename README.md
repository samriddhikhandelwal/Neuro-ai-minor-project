🧠 NeuroAI - Brain Tumor Detection Web App

NeuroAI is a Flask-based web application that predicts the presence of a brain tumor using a multimodal approach:

It accepts both symptoms (text input) and MRI brain scan images (image input).

The AI model is based on a CNN (Convolutional Neural Network) trained to classify brain MRI images as Tumor or No Tumor.

It provides results along with a clean, interactive frontend UI and feedback alerts.
----------------------------------------------------------------------------------------------------------------------------------
🛠 Features

Symptom-based and image-based brain tumor prediction.

AI-powered backend using TensorFlow/Keras.

Flask-based lightweight server and routing.

Profile page to save user info and image.

Contact form with message alert.

Beautiful UI styled with HTML, CSS and images.
-------------------------------------------------------------------------------------------------------------------------------------
File Structure:
health/
|
├── app.py                  # Main Flask application
├── backend.py              # (Optional logic file)
├── templates/              # All HTML pages
│   ├── index.html          # Home page (symptom & image input)
│   ├── profile.html        # Profile creation page
│   ├── result.html         # Output results page
│   ├── contact.html        # Contact Us form
│   ├── about.html          # About the project
|
├── static/                 # Static files
│   ├── uploads/            # Uploaded images (profile/MRI)
│   ├── neuroai-logo.png    # Brand logo
│   ├── prediction.jpg      # Backgrounds etc.
│   ├── style.css           # Common styles
|
├── models/                 # Saved AI model
│   ├── brain_tumor_detector.keras
|
├── brain_dataset_detect/   # Training data (if any)
├── README.md
├── .gitignore
└── requirements.txt        # Required Python packages
