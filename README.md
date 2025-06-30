# TomatoLeafAI

**TomatoLeafAI** is an AI-powered model that detects diseases in tomato leaves using deep learning. It is a core component of **Project Kisan**, an AI assistant designed to support small-scale farmers with actionable agricultural insights.

---

## 🚀 Features

- Detects tomato leaf conditions:
  - Early Blight
  - Late Blight
  - Healthy
- Built on **MobileNetV2** for lightweight and efficient performance
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

---

## 📁 Folder Structure

project-kisan-disease/
├── raw_tomato_dataset/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   └── Tomato___healthy/
├── dataset/
│ ├── train/
│ └── val/
├── plant_disease_model.py # Training script
├── test_prediction.py # Prediction script
├── requirements.txt # Dependencies
├── plant_disease_model.h5 # Trained model
└── split_dataset.py # Dataset splitting script


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

raw_tomato_dataset/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
└── Tomato___healthy/
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








