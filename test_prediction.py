import tensorflow as tf
import numpy as np
import cv2
import os
import csv
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Optional: Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuration
IMG_HEIGHT, IMG_WIDTH = 224, 224  # Match training dimensions
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for reliable prediction

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Enhanced image preprocessing with debugging"""
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            return None, None
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img_resized = cv2.resize(img, target_size)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_resized
        
    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        return None, None

def visualize_prediction(img_path, img_resized, predictions, class_names, pred_class, confidence):
    """Visualize the prediction with confidence scores"""
    plt.figure(figsize=(12, 4))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized)
    plt.title(f'Input Image: {os.path.basename(img_path)}')
    plt.axis('off')
    
    # Plot prediction probabilities
    plt.subplot(1, 2, 2)
    colors = ['red' if i == np.argmax(predictions) else 'blue' for i in range(len(predictions))]
    bars = plt.bar(range(len(predictions)), predictions * 100, color=colors)
    plt.title(f'Prediction Confidence\nPredicted: {pred_class} ({confidence:.1f}%)')
    plt.ylabel('Confidence (%)')
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    
    # Add value labels on bars
    for bar, prob in zip(bars, predictions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{prob*100:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'prediction_{os.path.splitext(os.path.basename(img_path))[0]}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("Plant Disease Prediction System")
    print("=" * 50)
    
    # Load the trained model
    try:
        model = tf.keras.models.load_model('plant_disease_model.keras')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Load class mapping
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        print(f"Class indices loaded: {class_indices}")
    except Exception as e:
        print(f"Error loading class indices: {str(e)}")
        return
    
    # Reverse the dictionary to get index -> class
    index_to_class = {v: k for k, v in class_indices.items()}
    class_names = list(class_indices.keys())
    
    print(f"Available classes: {class_names}")
    print(f"Index mapping: {index_to_class}")
    
    # Gather all image files in the current directory
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    # image_files = [f for f in os.listdir('.') if f.lower().endswith(image_extensions)]
    TEST_FOLDER = "test_images"
    image_files = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) if f.lower().endswith(image_extensions)]

    
    # Filter out generated prediction images
    image_files = [f for f in image_files if not f.startswith('prediction_')]
    
    # Check if any images are found
    if not image_files:
        print("No images found in the current directory.")
        return
    
    print(f"Image Found {len(image_files)} images to process")
    
    # Prepare CSV output file
    output_file = 'prediction_results.csv'
    results = []
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_path}")
        
        # Load and preprocess image
        img_batch, img_resized = load_and_preprocess_image(img_path, (IMG_HEIGHT, IMG_WIDTH))
        
        if img_batch is None:
            continue
        
        try:
            # Make prediction
            predictions = model.predict(img_batch, verbose=0)[0]
            
            # Debug: Print raw predictions
            print(f"Raw predictions: {predictions}")
            print(f"Prediction shape: {predictions.shape}")
            
            # Get predicted class
            pred_index = np.argmax(predictions)
            pred_class = index_to_class[pred_index]
            confidence = predictions[pred_index] * 100
            
            # Print all class probabilities
            print(f"::: Class probabilities:")
            for idx, (class_name, prob) in enumerate(zip(class_names, predictions)):
                marker = "👑" if idx == pred_index else "  "
                print(f"   {marker} {class_name}: {prob*100:.2f}%")
            
            # Check prediction confidence
            if confidence < CONFIDENCE_THRESHOLD * 100:
                print(f"Low confidence prediction ({confidence:.1f}%)")
            
            # Get treatment recommendation
            treatment = get_treatment_recommendation(pred_class, confidence)
            
            # Display results
            print(f"🍅 Image: {img_path}")
            print(f"🔍 Predicted Disease: {pred_class}")
            print(f"🎯 Confidence: {confidence:.2f}%")
            print(f"💊 Treatment: {treatment}")
            
            # Visualize prediction (optional - comment out if not needed)
            # visualize_prediction(img_path, img_resized, predictions, class_names, pred_class, confidence)
            
            # Store results
            results.append({
                'image': img_path,
                'predicted_class': pred_class,
                'confidence': confidence,
                'treatment': treatment,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'all_probabilities': {name: float(prob*100) for name, prob in zip(class_names, predictions)}
            })
            
        except Exception as e:
            print(f"Error predicting image {img_path}: {str(e)}")
            continue
    
    # Save results to CSV
    if results:
        save_results_to_csv(results, output_file)
        print(f"\nAll predictions completed and saved to '{output_file}'")
        
        # Print summary statistics
        print_summary_statistics(results)
    else:
        print("No successful predictions were made.")

def get_treatment_recommendation(pred_class, confidence):
    """Get treatment recommendation based on prediction"""
    if confidence < 70:
        return "Low confidence - recommend manual inspection by expert"
    
    treatments = {
        'Tomato___Early_blight': "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves, ensure good air circulation, and avoid overhead watering.",
        'Tomato___Late_blight': "Apply fungicides with copper compounds or mancozeb. Remove and destroy infected plants immediately. Avoid working in wet conditions.",
        'Tomato___healthy': "Plant appears healthy. Continue regular monitoring, maintain proper spacing, and ensure adequate nutrition.",
        'Tomato_Early_blight': "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves, ensure good air circulation, and avoid overhead watering.",
        'Tomato_Late_blight': "Apply fungicides with copper compounds or mancozeb. Remove and destroy infected plants immediately. Avoid working in wet conditions.",
        'Tomato_healthy': "Plant appears healthy. Continue regular monitoring, maintain proper spacing, and ensure adequate nutrition."
    }
    
    return treatments.get(pred_class, "Treatment information not available for this class.")

def save_results_to_csv(results, output_file):
    """Save results to CSV file"""
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Predicted_Class', 'Confidence_%', 'Treatment', 'Timestamp'])
        
        for result in results:
            writer.writerow([
                result['image'],
                result['predicted_class'],
                f"{result['confidence']:.2f}",
                result['treatment'],
                result['timestamp']
            ])

def print_summary_statistics(results):
    """Print summary statistics of predictions"""
    print("\n:: Prediction Summary:")
    print("=" * 30)
    
    # Count predictions by class
    class_counts = {}
    confidence_scores = []
    
    for result in results:
        pred_class = result['predicted_class']
        confidence = result['confidence']
        
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        confidence_scores.append(confidence)
    
    # Print class distribution
    print("Class Distribution:")
    for class_name, count in class_counts.items():
        percentage = (count / len(results)) * 100
        print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    # Print confidence statistics
    if confidence_scores:
        avg_confidence = np.mean(confidence_scores)
        min_confidence = np.min(confidence_scores)
        max_confidence = np.max(confidence_scores)
        
        print(f"\nConfidence Statistics:")
        print(f"   Average: {avg_confidence:.2f}%")
        print(f"   Range: {min_confidence:.2f}% - {max_confidence:.2f}%")
        print(f"   High confidence (>80%): {sum(1 for c in confidence_scores if c > 80)}/{len(confidence_scores)}")

if __name__ == "__main__":
    main()
