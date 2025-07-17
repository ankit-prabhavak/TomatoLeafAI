# test_prediction.py

import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Class names (should match train generator order)
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Load and preprocess image
img_path = 'leaf.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (128, 128))
img = img / 255.0
img = np.expand_dims(img, axis=0)  # (1, 128, 128, 3)

# Predict
pred = model.predict(img)
pred_class = class_names[np.argmax(pred)]

print(f"Predicted Disease: {pred_class}")
