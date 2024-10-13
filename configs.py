import os
# configs.py
from constants import LABEL_MAP, BATCH_SIZE

# Any additional configurations can go here

# Directories
RAW_DATA_DIR = r"data/train/raw/Task "
AUG_DATA_DIR = r"data/train/augmented/Task "
EXTERNAL_DATA_DIR = r"data/train/external/Task "
COMBINED_DATA_DIR = r"data/train/combined/Task "
TEST_DATA_DIR = r"data/test/"
TEMP_DATA_DIR = "data/temp/Task "
TRAIN_DATA_DIR = r"data/train/"
VAL_DATA_DIR = r"data/val/"
import os

import torch

# Directories
TRAIN_DATA_DIR = r"data/train/"
VAL_DATA_DIR = r"data/val/"
TEST_DATA_DIR = r"data/test/"
MODEL_SAVE_PATH = "/Users/User/Desktop/SpiralSense/saved_model.pth"
# Model settings
NUM_CLASSES = 3
MODEL_NAME = 'ResNet18'  # Specify the model name

# Training settings
RANDOM_SEED = 123
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# List of classes and label map

NUM_EPOCHS = 100
LEARNING_RATE = 0.1
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import transforms

# Define preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
from PIL import Image
import os

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Open image as RGB
        label = self.get_label_from_filename(self.image_files[idx])  # Custom function to get labels

        if self.transform:
            image = self.transform(image)

        return image, label


def load_data(train_dir, val_dir, test_dir, preprocess):
    def get_image_paths_and_labels(data_dir):
        image_paths = []
        labels = []
        for label_name, label_id in LABEL_MAP.items():
            class_dir = os.path.join(data_dir, label_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(label_id)
        return image_paths, labels

    train_image_paths, train_labels = get_image_paths_and_labels(train_dir)
    val_image_paths, val_labels = get_image_paths_and_labels(val_dir)
    test_image_paths, test_labels = get_image_paths_and_labels(test_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Add other transformations as needed
    ])

    train_dataset = CustomDataset(train_image_paths, train_labels, transform=transform)
    val_dataset = CustomDataset(val_image_paths, val_labels, transform=transform)
    test_dataset = CustomDataset(test_image_paths, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
