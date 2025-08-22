from flask import Flask, render_template, request, send_file
import tensorflow as tf
import numpy as np
import cv2
import os
import csv
from datetime import datetime
import json

app = Flask(__name__)

# Constants
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'plant_disease_model.keras'
CLASS_INDEX_PATH = 'class_indices.json'
IMG_SIZE = (224, 224)

# Load model and class names
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_INDEX_PATH, 'r') as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}
class_names = list(class_indices.keys())

# Load and preprocess image
def preprocess_image(path, target_size=(224, 224)):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Prediction function
def predict_and_save_results(upload_folder):
   
    image_files = [f for f in os.listdir(upload_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    results = []
    output_file = os.path.join(upload_folder, 'prediction_results.csv')

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Predicted Class', 'Confidence (%)', 'Treatment', 'Timestamp'])

        for filename in image_files:
            filepath = os.path.join(upload_folder, filename)
            img = preprocess_image(filepath, IMG_SIZE)
            if img is None:
                continue

            preds = model.predict(img, verbose=0)[0]
            pred_index = np.argmax(preds)
            pred_class = index_to_class[pred_index]
            confidence = preds[pred_index] * 100
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            treatment = get_treatment_recommendation(pred_class, confidence)

            writer.writerow([filename, pred_class, f"{confidence:.2f}", treatment, timestamp])
            # results.append({
            #     'image': filename,
            #     'pred_class': pred_class,
            #     'confidence': f"{confidence:.2f}",
            #     'treatment': treatment,
            #     'timestamp': timestamp
            # })
            # Normalize class name for template
            if pred_class == 'Tomato_healthy':
                pred_class_clean = 'Healthy'
            else:
                pred_class_clean = pred_class.replace('Tomato_', '').replace('_', ' ').title()

            results.append({
                'image': filename,
                'pred_class': pred_class_clean,
                'confidence': f"{confidence:.2f}",
                'treatment': treatment,
                'timestamp': timestamp
            })


    return results, output_file

# Treatment info
def get_treatment_recommendation(pred_class, confidence):
    if confidence < 70:
        return "Low confidence. Please check manually."

    treatments = {
        'Tomato___Early_blight': "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves, ensure good air circulation, and avoid overhead watering.",
        'Tomato___Late_blight': "Apply fungicides with copper compounds or mancozeb. Remove and destroy infected plants immediately. Avoid working in wet conditions.",
        'Tomato___healthy': "Plant appears healthy. Continue regular monitoring, maintain proper spacing, and ensure adequate nutrition.",
        'Tomato_Early_blight': "Use basic fungicide. Remove affected leaves.",
        'Tomato_Late_blight': "Apply fungicide. Avoid watering leaves.",
        'Tomato_healthy': "Plant is healthy.",
    }

    return treatments.get(pred_class, "No treatment available.")


# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    csv_path = None

    if request.method == 'POST':
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        # Clear previous uploads
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))

        for file in request.files.getlist('images'):
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))

        results, csv_path = predict_and_save_results(UPLOAD_FOLDER)

    return render_template('index.html', results=results, csv_path=csv_path)

@app.route('/download')
def download_csv():
    path = request.args.get('path')
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)