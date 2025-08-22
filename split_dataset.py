import os
import shutil
import random
from tqdm import tqdm

source_dir = 'raw_tomato_dataset'
train_dir = 'dataset/train'
val_dir = 'dataset/val'

split_ratio = 0.7  # 70% train, 30% val

# Create target folders
for subdir in os.listdir(source_dir):
    os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(val_dir, subdir), exist_ok=True)

# Go through each class
for category in tqdm(os.listdir(source_dir), desc="Splitting dataset"):
    imgs = os.listdir(os.path.join(source_dir, category))
    random.shuffle(imgs)

    split_point = int(len(imgs) * split_ratio)
    train_imgs = imgs[:split_point]
    val_imgs = imgs[split_point:]

    for img in train_imgs:
        shutil.copy(
            os.path.join(source_dir, category, img),
            os.path.join(train_dir, category, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(source_dir, category, img),
            os.path.join(val_dir, category, img)
        )

print("Dataset split complete!")
