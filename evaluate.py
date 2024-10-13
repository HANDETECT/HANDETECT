import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from models import ResNetWithCurvature  # Ensure this model is designed to accept curvature features
from data_loader import load_data
import cv2
import os
from PIL import Image
from scipy.fftpack import fft
from scipy.stats import entropy, skew, kurtosis
from sklearn.linear_model import LinearRegression

def compute_regression_features(curvature, window_size=10):
    """Compute regression features from curvature data."""
    num_segments = len(curvature) // window_size
    r2_scores = []

    for i in range(num_segments):
        start = i * window_size
        end = start + window_size
        segment = curvature[start:end]
        if len(segment) < window_size:
            break

        # Fit a linear model
        X = np.arange(len(segment)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, segment)
        r2_score = model.score(X, segment)  # R^2 score
        r2_scores.append(r2_score)

    return np.mean(r2_scores) if r2_scores else 0, np.std(r2_scores) if r2_scores else 0

def compute_features(image):
    """Extract features from an image."""
    radius = compute_radius(image)
    
    # Ensure radius is not NaN
    if np.isnan(radius):
        print("NaN found in radius!")

    curvature = compute_curvature_from_image(image)

    # Ensure curvature is valid
    if np.isnan(curvature).any():
        print("NaN found in curvature from image!")

    # Compute new features: Fourier, entropy, skewness, kurtosis, inversions
    fourier_features = compute_fourier_features(curvature)
    
    # Check Fourier features
    if np.isnan(fourier_features).any():
        print("NaN found in Fourier features!")

    curvature_entropy, curvature_skewness, curvature_kurtosis = compute_entropy_skewness_kurtosis(curvature)
    
    # Ensure statistical features are valid
    if np.isnan(curvature_entropy) or np.isnan(curvature_skewness) or np.isnan(curvature_kurtosis):
        print("NaN found in statistical features!")

    num_inversions = compute_inversions(curvature)
    
    # Ensure inversions is valid
    if np.isnan(num_inversions):
        print("NaN found in inversions!")

    # Compute regression-based features
    mean_r2, std_r2 = compute_regression_features(curvature)

    # Check regression features
    if np.isnan(mean_r2) or np.isnan(std_r2):
        print("NaN found in regression features!")

    # Combine all features
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

    fixed_length = 100  # Adjust to the expected feature size
    return np.pad(combined_features, (0, max(0, fixed_length - len(combined_features))), mode='constant')[:fixed_length]


def compute_accuracy(loader, model, criterion, device):
    """Compute accuracy for the given DataLoader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, curvature_features in loader:
            images, labels, curvature_features = images.to(device), labels.to(device), curvature_features.to(device)

            outputs = model(images, curvature_features)
            
            # Check for NaN values in outputs
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


# Fourier and statistical features extraction functions

def compute_fourier_features(curvature):
    valid_curvature = curvature[~np.isnan(curvature)]
    if len(valid_curvature) == 0:
        return np.zeros(5)  # Return a default zero array if no valid curvature
    fft_values = fft(valid_curvature)
    magnitude_spectrum = np.abs(fft_values)[:len(fft_values)//2]
    return magnitude_spectrum[:5]  # Return the first 5 components



def compute_entropy_skewness_kurtosis(curvature):
    """Compute entropy, skewness, and kurtosis of curvature data."""
    valid_curvature = curvature[~np.isnan(curvature)]
    if len(valid_curvature) < 2 or np.all(valid_curvature == valid_curvature[0]):
        # Return 0 or some default value for entropy, skewness, kurtosis if invalid
        return 0, 0, 0
    curvature_entropy = entropy(np.histogram(valid_curvature, bins=10, density=True)[0])
    curvature_skewness = skew(valid_curvature)
    curvature_kurtosis = kurtosis(valid_curvature)

    return curvature_entropy, curvature_skewness, curvature_kurtosis


def compute_inversions(curvature):
    """Compute the number of inversions in curvature data."""
    sign_changes = np.diff(np.sign(curvature))
    return np.count_nonzero(sign_changes)


def compute_radius(image):
    """Compute the radius of the spiral from the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # Binarize the image

    # Ensure the binary image is of type uint8 (CV_8UC1)
    binary = binary.astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0
    
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return w / 2


def compute_curvature(spiral_points):
    """Compute the curvature of spiral points."""
    if len(spiral_points) < 3:
        return np.zeros(len(spiral_points))
    
    dx = np.gradient(spiral_points[:, 0])
    dy = np.gradient(spiral_points[:, 1])

    if np.all(dx == 0) and np.all(dy == 0):
        return np.zeros(len(spiral_points))  # Handle flat case

    # Calculate second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Calculate curvature
    denominator = (dx ** 2 + dy ** 2) ** (3/2)
    if np.all(denominator == 0):  # Handle flat regions where there's no curvature
        return np.zeros(len(spiral_points))
    curvature = (dx * ddy - dy * ddx) / (denominator + 1e-9)

    return curvature


def extract_curve_data(image):
    """Extract curve data from the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # Binarize the image

    # Ensure the binary image is of type uint8 (CV_8UC1)
    binary = binary.astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None  # Return None if no contours found

    # Process the contours as needed for curvature calculations
    # For example, extracting x and y coordinates from the largest contour
    contour = max(contours, key=cv2.contourArea)
    points = contour[:, 0]  # Extract the points
    x = points[:, 0]
    y = points[:, 1]

    return x, y


def compute_curvature_from_image(image):
    """Compute curvature from the image."""
    x, y = extract_curve_data(image)
    if x is None or y is None:
        return np.zeros(100)  # Return a fixed length zero array if no curvature can be computed
    spiral_points = np.vstack((x, y)).T  # Combine x and y into a 2D array
    return compute_curvature(spiral_points)  # Pass the combined points

# Load and preprocess images, including curvature feature extraction
def load_and_preprocess_image(image_path, preprocess, device):
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transformations
    image = preprocess(image)
    image = image.unsqueeze(0).to(device)
    
    # Compute curvature features
    curvature_features = compute_features(np.array(image.cpu().squeeze().permute(1, 2, 0)))  # Convert tensor back to numpy for feature extraction
    curvature_features = torch.tensor(curvature_features, dtype=torch.float32).unsqueeze(0).to(device)

    return image, curvature_features

# Modify your evaluation function to assess all images in the test folder
def evaluate_folder(folder_path, model, preprocess, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                # Determine the label from the folder name
                label = os.path.basename(subdir)
                label_id = 0  # Map label (folder name) to numerical ID

                if label == 'Healthy':
                    label_id = 0
                elif label == 'Parkinsons':
                    label_id = 1
                elif label == 'Alzheimers':
                    label_id = 2

                # Load and preprocess the image
                image_path = os.path.join(subdir, file)
                image, curvature_features = load_and_preprocess_image(image_path, preprocess, device)
                
                # Forward pass through the model
                outputs = model(image, curvature_features)
                _, predicted = torch.max(outputs.data, 1)

                total += 1
                correct += (predicted == label_id).sum().item()

                all_labels.append(label_id)
                all_preds.append(predicted.item())

    accuracy = 100 * correct / total
    return accuracy, np.array(all_labels), np.array(all_preds)

def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TEST_DATA_DIR = r"data/test/"
    MODEL_SAVE_PATH = 'saved_model.pth'
    NUM_CLASSES = 3

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load model
    model = ResNetWithCurvature(num_classes=NUM_CLASSES, num_curvature_features=100)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.to(DEVICE)

    # Evaluate all images in the test folder
    accuracy, all_labels, all_preds = evaluate_folder(TEST_DATA_DIR, model, preprocess, DEVICE)

    print(f"Test Accuracy: {accuracy:.2f}%")

    # Compute and display confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Healthy', 'Parkinson\'s Disease', 'Alzheimer\'s Disease'])
    disp.plot(cmap='Blues')

    print("Confusion matrix:")
    print(conf_matrix)

if __name__ == "__main__":
    main()
