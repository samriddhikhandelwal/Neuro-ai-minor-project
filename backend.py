from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

app = Flask(__name__, template_folder='templates')
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit', methods=['POST'])
def submit():
    text_input = request.form.get('text_input')
    image_file = request.files.get('image_input')

    image_path = None
    if image_file and image_file.filename != '':
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)
    
    # For now just print or log what was submitted
    print("Text input:", text_input)
    if image_path:
        print("Image saved at:", image_path)
    else:
        print("No image uploaded.")

    return render_template('result.html', text=text_input, image_path=image_path)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    file = request.files.get('file')

    image_path = None
    if file and file.filename != '':
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

    # Simulate calling an AI model
    prediction = ai_model_predict(text, image_path)

    return render_template('result.html', text=text, prediction=prediction)

def ai_model_predict(symptoms_text, image_path):
    # This is a dummy logic — replace with real model prediction
    if 'fever' in symptoms_text.lower():
        return "Possible Disease: Flu or Viral Infection"
    elif image_path:
        return "Analyzing image... Possible Skin Condition Detected"
    else:
        return "Not enough data to detect a specific disease."

if __name__ == '__main__':
    app.run(debug=True, port=5000)
