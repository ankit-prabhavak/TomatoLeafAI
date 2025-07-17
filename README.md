# TomatoLeafAI

**TomatoLeafAI** is a deep learning–based image classification model designed to detect common diseases in tomato plant leaves with high accuracy. Leveraging a custom-built **Convolutional Neural Network (CNN) architecture**, this model delivers high-accuracy disease diagnosis directly from tomato leaf images, making it ideal for practical, real-world agricultural applications. It is designed for both educational clarity and future-ready integration with cloud and mobile platforms.

This project is a key module within **Project Kisan** : An AI-driven initiative focused on empowering small and marginal farmers with accessible, technology-based agricultural solutions. TomatoLeafAI aims to assist farmers in identifying crop health issues early, reducing crop losses, and improving yield through timely interventions and expert recommendations.

---

## 🚀 Features

- Detects tomato leaf conditions:
  - Early Blight
  - Late Blight
  - Healthy
- Built using a custom **Convolutional Neural Network (CNN)** for improved accuracy and control.
- Supports deployment on:
  - Mobile devices via **TensorFlow Lite**
  - Cloud via **Firebase ML Kit**
- Written in clean, beginner-friendly **Python** code

---

## 🛠 Tech Stack

- Python 3.x
- TensorFlow / Keras
- OpenCV
- tqdm
- Custom CNN architecture

---

## 📁 Folder Structure

```text
project-kisan-disease/
├── raw_tomato_dataset/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   └── Tomato___healthy/
├── dataset/
│ ├── train/
│ └── val/
├── images/
│   └── tomatoleafai_workflow.jpg
├── plant_disease_model.py # Training script
├── test_prediction.py # Prediction script
├── requirements.txt # Dependencies
├── prediction_results.csv
├── plant_disease_model.h5 # Trained model
├── sample_image1.jpg # Add the leaf images for prediction directly in the root directory 
└── split_dataset.py # Dataset splitting script
```

---

## Disease Detection Workflow

![TomatoLeafAI Workflow](images/tomatoleafai_workflow.jpg)

---

## 🔍 How It Works

1. Capture or upload an image of a tomato leaf.
2. The model analyzes the image and classifies it as:
   - Early Blight
   - Late Blight
   - Healthy
3. Future updates will include voice-based feedback and Firebase integration.

---

## 🧪 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/TomatoLeafAI.git
cd TomatoLeafAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Place the following folders inside a raw_tomato_dataset/ directory:

```text
raw_tomato_dataset/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
└── Tomato___healthy/
```

Then run the dataset splitter to create training and validation sets:

```bash
python split_dataset.py
```

### 4. Train the Model

```bash
python plant_disease_model.py
```

The trained model will be saved as:
plant_disease_model.h5

### 5. Test the Model

Place a test image (e.g., leaf.jpg) in the project directory. Update the filename inside test_prediction.py and run:

```bash
python test_prediction.py
```

### 6. Convert to TensorFlow Lite (Optional)

For mobile or Firebase deployment.

---

## 🌱 Project Kisan

*TomatoLeafAI* is a core module of **Project Kisan**, submitted to the Google Hackathon 2025. The objective of the project is to build a voice-enabled AI assistant that delivers essential agricultural support to small and marginal farmers through user-friendly, technology-driven tools.

### Modules Included in Project Kisan

- **Disease Detection** – *In Progress*  
- **Market Price Insights** – *In Progress*  
- **Government Scheme Finder** – *In Progress*  
- **Multilingual Voice Support** – *In Progress*

---

## 📞 Contact Information

- **Email:** [ankitabcd1718@gmail.com](mailto:ankitabcd1718@gmail.com)  
- **GitHub:** [https://github.com/ankit-prabhavak](https://github.com/ankit-prabhavak)  
- **LinkedIn:** [https://linkedin.com/in/ankit-prabhavak](https://linkedin.com/in/ankit-prabhavak)

---

## Acknowledgements

- **Dataset:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- **Technologies Used:** TensorFlow, Keras, OpenCV, tqdm
