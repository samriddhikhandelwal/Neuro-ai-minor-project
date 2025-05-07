# app.py
from flask import Flask, flash, get_flashed_messages, session , render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = 'your_secret_key'  # Needed for flashing messages
# Load your model (✅ correct and modern)
model = load_model('models/brain_tumor_detector.keras')

# Make sure upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')  # Your form page

@app.route('/predict', methods=['POST'])
def predict():
    # Get text input
    symptoms = request.form.get('symptoms')
    print(f"Symptoms received: {symptoms}")

    # Get uploaded image
    uploaded_file = request.files['image']
    if uploaded_file.filename != '':
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
        uploaded_file.save(file_path)
        print(f"Image saved at: {file_path}")

        # Preprocess image
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(img_array)
        print(f"Prediction raw output: {prediction}")

        if prediction[0][0] > 0.5:
            result = "🚫 No Brain Tumor Detected"
        else:
            result = "⚠️ Brain Tumor Detected"
    else:
        result = "No image uploaded."

    return render_template('result.html', symptoms=symptoms, prediction=result)

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/about')
def about():
    return render_template('about.html')

from flask import flash, redirect, url_for

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # You can print or store this message as needed
        print(f"Received message from {name} ({email}): {message}")
        flash("Message sent successfully!")
        return redirect(url_for('contact'))

    return render_template('contact.html')


@app.route('/profile', methods=['GET'])
def show_profile_form():
    return render_template('profile.html')  # assuming your template is named 'profile.html'

@app.route('/save_profile', methods=['POST'])
def save_profile():
    username = request.form['username']
    contact = request.form['contact']
    email = request.form['email']
    profile_image = request.files['profile_image']

    if profile_image:
        filename = profile_image.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        profile_image.save(filepath)

    # Just print the form data in terminal (or save to DB later)
    print(f"Saved Profile - Name: {username}, Contact: {contact}, Email: {email}, Image: {filename}")
    flash('Profile saved successfully!')
    return redirect(url_for('profile'))  # Redirect back to profile page


if __name__ == '__main__':
    app.run(debug=True)
