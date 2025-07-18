import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# --------------------------- Configuration --------------------------- #
train_dir = 'dataset/train'
val_dir = 'dataset/val'
img_height, img_width = 224, 224  # Increased size for better feature extraction
batch_size = 16  # Reduced batch size for better convergence
epochs = 30  # Increased epochs with early stopping
model_save_path = 'plant_disease_model.keras'
class_index_path = 'class_indices.json'
history_path = 'training_history.json'

# --------------------------- Data Analysis --------------------------- #
def analyze_dataset(directory):
    """Analyze dataset distribution"""
    print(f"\n:: Dataset Analysis for: {directory}")
    classes = os.listdir(directory)
    for class_name in classes:
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            print(f"   {class_name}: {count} images")
    return classes

# Analyze both train and validation datasets
train_classes = analyze_dataset(train_dir)
val_classes = analyze_dataset(val_dir)

# --------------------------- Enhanced Data Generators --------------------------- #
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42  # For reproducibility
)

#  Save class indices for prediction use
with open(class_index_path, 'w') as f:
    json.dump(train_gen.class_indices, f, indent=2)

print(f"\n</> Class indices mapping: {train_gen.class_indices}")

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,  # Don't shuffle validation data
    seed=42
)

# --------------------------- Improved Model Architecture --------------------------- #
model = Sequential([
    # First Conv Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Second Conv Block
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Third Conv Block
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Fourth Conv Block
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Flatten and Dense layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(train_classes), activation='softmax')  # Dynamic number of classes
])

# --------------------------- Compilation with Learning Rate --------------------------- #
initial_learning_rate = 0.001
optimizer = Adam(learning_rate=initial_learning_rate)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --------------------------- Enhanced Callbacks --------------------------- #
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    model_save_path, 
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# --------------------------- Model Training --------------------------- #
print(f"\n-->> Starting training with {len(train_classes)} classes...")
print(f"-->> Training samples: {train_gen.samples}")
print(f"-->> Validation samples: {val_gen.samples}")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# --------------------------- Save Training History --------------------------- #
history_dict = {
    'loss': [float(x) for x in history.history['loss']],
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']]
}

with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)

# --------------------------- Training Visualization --------------------------- #
def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_training_history(history)

# --------------------------- Model Evaluation --------------------------- #
print(f"\n:: Final Training Results:")
print(f"   Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"   Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"   Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"   Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

# Load the best model for evaluation
best_model = tf.keras.models.load_model(model_save_path)

# Evaluate on validation set
val_loss, val_accuracy = best_model.evaluate(val_gen, verbose=0)
print(f"\n<<.>> Best Model Performance:")
print(f"   Validation Accuracy: {val_accuracy:.4f}")
print(f"   Validation Loss: {val_loss:.4f}")

print(f"\n[.] Training complete. Best model saved to '{model_save_path}'")
print(f"[.] Class indices saved to '{class_index_path}'")
print(f"[.] Training history saved to '{history_path}'")
