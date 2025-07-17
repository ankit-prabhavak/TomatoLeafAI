# test_prediction.py

import tensorflow as tf
import numpy as np
import cv2
import os
import csv
from datetime import datetime

# Load the trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Class labels (must match the training order)
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Gather all image files from current directory
image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir('.') if f.lower().endswith(image_extensions)]

# 🛡️ Check if any images are found
if not image_files:
    print("⚠️ No images found in the current directory.")
    exit()

# Prepare CSV output file
output_file = 'prediction_results.csv'
with open(output_file, mode='w', newline='', encoding='utf-8') as file:

    writer = csv.writer(file)
    writer.writerow(['Image', 'Predicted Disease', 'Confidence (%)', 'Diagnosis', 'Timestamp'])

    # Process each image
    for img_path in image_files:
        img = cv2.imread(img_path)

        # Handle unreadable image
        if img is None:
            print(f"⚠️ Could not read image: {img_path}")
            continue

        # Preprocess image
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)  # (1, 128, 128, 3)

        # Predict
        pred = model.predict(img)
        pred_index = np.argmax(pred)
        pred_class = class_names[pred_index]
        confidence = pred[0][pred_index] * 100
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Diagnosis / Cure suggestion
        if pred_class == 'Early Blight':
            cure = "🩺 Use fungicides like chlorothalonil. Remove infected leaves and avoid overhead watering."
        elif pred_class == 'Late Blight':
            cure = "🩺 Apply fungicides with mancozeb. Destroy infected plants and rotate crops."
        elif pred_class == 'Healthy':
            cure = "✅ Plant is healthy. Continue regular monitoring and good spacing."
        else:
            cure = "⚠️ Diagnosis unavailable."

        # Display result
        print(f"🍅 Image: {img_path}")
        print(f"🔍 Predicted Disease: {pred_class}")
        print(f"🎯 Confidence: {confidence:.2f}%")
        print(f"🧾 Diagnosis: {cure}\n")

        # Write to CSV
        writer.writerow([img_path, pred_class, f"{confidence:.2f}", cure, timestamp])

print(f"✅ All predictions saved to '{output_file}'")

# Optional: Save the model again (only if updated or retrained)
# model.save('plant_disease_model.h5')
