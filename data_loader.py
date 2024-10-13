import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from scipy.fftpack import fft
from scipy.stats import entropy, skew, kurtosis
from sklearn.linear_model import LinearRegression
from constants import LABEL_MAP, BATCH_SIZE  # Import constants here

# Function to compute curvature
def compute_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = (d2x * dy - dx * d2y) / (dx ** 2 + dy ** 2) ** (3 / 2)
    return curvature

# Dummy curve extraction function (replace with actual logic)
def extract_curve_data(image):
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x)
    return x, y

# Function to compute accuracy
def compute_accuracy(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, curvature_features in loader:
            images, labels, curvature_features = images.to(device), labels.to(device), curvature_features.to(device)
            outputs = model(images, curvature_features)

            if torch.isnan(outputs).any():
                print("NaN found in model outputs!")

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Feature extraction functions
def compute_regression_features(curvature, window_size=10):
    if curvature is None or len(curvature) < window_size:
        print("Insufficient curvature data for regression features.")
        return 0, 0

    num_segments = len(curvature) // window_size
    r2_scores = []

    for i in range(num_segments):
        start = i * window_size
        end = start + window_size
        segment = curvature[start:end]

        if len(segment) < window_size:
            continue

        X = np.arange(len(segment)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, segment)
        r2_score = model.score(X, segment)
        r2_scores.append(r2_score)
        print(f"Segment {i}: R2 Score = {r2_score}")

    mean_r2 = np.mean(r2_scores) if r2_scores else 0
    std_r2 = np.std(r2_scores) if r2_scores else 0
    print(f"Mean R2: {mean_r2}, Std R2: {std_r2}")
    return mean_r2, std_r2

def compute_fourier_features(curvature):
    valid_curvature = curvature[~np.isnan(curvature)]
    if len(valid_curvature) == 0:
        return np.zeros(5)
    fft_values = fft(valid_curvature)
    magnitude_spectrum = np.abs(fft_values)[:len(fft_values) // 2]
    return magnitude_spectrum[:5]

def compute_entropy_skewness_kurtosis(curvature):
    valid_curvature = curvature[~np.isnan(curvature)]
    if len(valid_curvature) < 2 or np.all(valid_curvature == valid_curvature[0]):
        return 0, 0, 0
    curvature_entropy = entropy(np.histogram(valid_curvature, bins=10, density=True)[0])
    curvature_skewness = skew(valid_curvature)
    curvature_kurtosis = kurtosis(valid_curvature)
    return curvature_entropy, curvature_skewness, curvature_kurtosis

def compute_inversions(curvature):
    if curvature is None or len(curvature) < 2:
        print("Curvature data is invalid for inversion calculation.")
        return 0
    sign_changes = np.diff(np.sign(curvature))
    return np.count_nonzero(sign_changes)

def compute_radius(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return w / 2

def compute_curvature_from_points(spiral_points):
    if len(spiral_points) < 3:
        return np.zeros(len(spiral_points))

    dx = np.gradient(spiral_points[:, 0])
    dy = np.gradient(spiral_points[:, 1])

    if np.all(dx == 0) and np.all(dy == 0):
        return np.zeros(len(spiral_points))

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denominator = (dx ** 2 + dy ** 2) ** (3 / 2)
    if np.all(denominator == 0):
        return np.zeros(len(spiral_points))
    curvature = (dx * ddy - dy * ddx) / (denominator + 1e-9)
    return curvature

def compute_curvature_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    x, y = extract_curve_data(image)
    curvature = compute_curvature(x, y)
    return curvature

def compute_features(image):
    radius = compute_radius(image)
    curvature = compute_curvature_from_image(image)

    fourier_features = compute_fourier_features(curvature)
    curvature_entropy, curvature_skewness, curvature_kurtosis = compute_entropy_skewness_kurtosis(curvature)
    num_inversions = compute_inversions(curvature)
    mean_r2, std_r2 = compute_regression_features(curvature)

    combined_features = np.array([
        radius,
        *fourier_features,
        curvature_entropy,
        curvature_skewness,
        curvature_kurtosis,
        num_inversions,
        mean_r2,
        std_r2
    ])

    fixed_length = 100
    return np.pad(combined_features, (0, max(0, fixed_length - len(combined_features))), mode='constant')[:fixed_length]

# Custom dataset class
from torch.utils.data import Dataset
from PIL import Image
import os

from torchvision import datasets

from torchvision import transforms
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image as a PIL image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Load image

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        # You need to compute curvature features here (ensure you pass the right data)
        curvature_features = compute_curvature_from_image(image_path)  # Pass the image path

        return image, label, curvature_features


    def calculate_curvature(self, img):
        if isinstance(img, np.ndarray):  # Check if img is a NumPy array
            curvature_values = compute_curvature_from_image(img)  # Directly pass the array
        else:
            # Handle the case where img is a file path
            img_array = np.array(Image.open(img).convert("RGB"))
            curvature_values = compute_curvature_from_image(img_array)
        return curvature_values

    def get_label(self, image_path):
        """Extract label from the image path based on LABEL_MAP."""
        for label_name, label_id in LABEL_MAP.items():
            if label_name in image_path:
                return label_id
        return -1  # Return -1 for unknown labels

def load_data(train_dir, val_dir, test_dir, batch_size, preprocess):
    # Load image paths and labels for training, validation, and test data
    def get_image_paths_and_labels(data_dir):
        image_paths = []
        labels = []
        for label_name, label_id in LABEL_MAP.items():
            label_dir = os.path.join(data_dir, label_name)
            if os.path.exists(label_dir):
                for img_name in os.listdir(label_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):  # Modify extensions as needed
                        image_paths.append(os.path.join(label_dir, img_name))
                        labels.append(label_id)

        return image_paths, labels

    train_image_paths, train_labels = get_image_paths_and_labels(train_dir)
    val_image_paths, val_labels = get_image_paths_and_labels(val_dir)
    test_image_paths, test_labels = get_image_paths_and_labels(test_dir)

    train_dataset = CustomDataset(train_image_paths, train_labels, transform=preprocess)
    val_dataset = CustomDataset(val_image_paths, val_labels, transform=preprocess)
    test_dataset = CustomDataset(test_image_paths, test_labels, transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Loaded {len(train_dataset)} training images.")
    print(f"Loaded {len(val_dataset)} validation images.")
    print(f"Loaded {len(test_dataset)} test images.")

    return train_loader, val_loader, test_loader

# Example of using the load_data function
if __name__ == "__main__":
    # Define your directories
    train_dir = 'path/to/train/directory'
    val_dir = 'path/to/validation/directory'
    test_dir = 'path/to/test/directory'

    # Example preprocess function (adjust as needed)
    preprocess = None  # Replace with actual preprocessing transforms

    # Load data
    train_loader, val_loader, test_loader = load_data(train_dir, val_dir, test_dir, BATCH_SIZE, preprocess)
