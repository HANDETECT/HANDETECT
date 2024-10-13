import numpy as np
import cv2
from sklearn.metrics import accuracy_score
import os
from scipy.stats import entropy, skew, kurtosis
from scipy.fft import fft
import matplotlib.pyplot as plt

CLASSES = ['Alzheimers Disease', 'Healthy', 'Parkinsons Disease']
TRAIN_DATA_DIR = r"data/train/"
VAL_DATA_DIR = r"data/val/"
TEST_DATA_DIR = r"data/test/"

def get_labels(data_dir):
    """Get labels for the dataset based on subfolder names."""
    labels = []
    image_paths = []
    for class_name in os.listdir(data_dir):
        class_folder = os.path.join(data_dir, class_name)
        if os.path.isdir(class_folder):
            label = CLASSES.index(class_name)
            for img_name in os.listdir(class_folder):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(class_folder, img_name))
                    labels.append(label)
    return image_paths, labels

def extract_curve_data(image):
    """Extract curve data from the image and highlight the curve on the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    binary = binary.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found!")
        return None, None, image

    contour = max(contours, key=cv2.contourArea)
    points = contour[:, 0]
    x = points[:, 0]
    y = points[:, 1]

    # Draw the contour on the image
    highlighted_image = image.copy()
    cv2.drawContours(highlighted_image, [contour], -1, (0, 0, 255), 2)  # Red color for the contour

    print(f"Extracted {len(x)} points from the contour.")
    return x, y, highlighted_image

def compute_features(image):
    """Extract specified features from an image."""
    x, y, _ = extract_curve_data(image)  # Ignore the highlighted_image
    if x is None or y is None:
        return np.zeros(10)  # Return zero vector if no curve is detected
    
    # Compute radius
    center_x, center_y = np.mean(x), np.mean(y)
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    radius = np.mean(distances)
    
    # Compute Fourier features
    fft_values = fft(distances)
    fourier_features = np.abs(fft_values[:3])  # Use first 3 Fourier coefficients
    
    # Compute curvature
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-8)
    
    # Compute curvature statistics
    curvature_entropy = entropy(np.histogram(curvature, bins=20)[0])
    curvature_skewness = skew(curvature)
    curvature_kurtosis = kurtosis(curvature)
    
    # Compute number of inversions
    num_inversions = np.sum(np.diff(curvature) < 0)
    
    # Compute R^2 values
    t = np.arange(len(x))
    x_fit = np.polyfit(t, x, 2)
    y_fit = np.polyfit(t, y, 2)
    x_r2 = 1 - np.sum((x - np.polyval(x_fit, t))**2) / np.sum((x - np.mean(x))**2)
    y_r2 = 1 - np.sum((y - np.polyval(y_fit, t))**2) / np.sum((y - np.mean(y))**2)
    mean_r2 = (x_r2 + y_r2) / 2
    std_r2 = np.std([x_r2, y_r2])
    
    features = np.array([
        radius,
        *fourier_features,
        curvature_entropy,
        curvature_skewness,
        curvature_kurtosis,
        num_inversions,
        mean_r2,
        std_r2
    ])
    
    return features

def extract_features_from_images(image_paths):
    """Extract features from a list of image paths."""
    features = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            features.append(compute_features(img))
    return np.array(features)

def find_best_threshold(feature_values, labels, feature_name):
    """Find the best threshold for a single feature to separate classes."""
    sorted_features = np.sort(feature_values)
    thresholds = (sorted_features[1:] + sorted_features[:-1]) / 2
    
    best_threshold = None
    best_accuracy = 0
    best_direction = None
    
    for threshold in thresholds:
        for direction in ['greater', 'less']:
            if direction == 'greater':
                predictions = (feature_values > threshold).astype(int)
            else:
                predictions = (feature_values <= threshold).astype(int)
            
            accuracy = accuracy_score(labels, predictions)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                best_direction = direction
    
    return best_threshold, best_direction, best_accuracy

def determine_thresholds(features, labels):
    """Determine the best thresholds for each feature."""
    thresholds = {}
    feature_names = [
        'radius', 'fourier_1', 'fourier_2', 'fourier_3',
        'curvature_entropy', 'curvature_skewness', 'curvature_kurtosis',
        'num_inversions', 'mean_r2', 'std_r2'
    ]
    
    for i, feature_name in enumerate(feature_names):
        feature_values = features[:, i]
        threshold, direction, accuracy = find_best_threshold(feature_values, labels, feature_name)
        thresholds[feature_name] = (threshold, direction, accuracy)
    
    return thresholds

def threshold_classify(features, thresholds):
    """Classify based on determined thresholds."""
    scores = np.zeros(3)
    
    for i, (feature_name, (threshold, direction, _)) in enumerate(thresholds.items()):
        feature_value = features[i]
        if direction == 'greater':
            if feature_value > threshold:
                scores[1] += 1  # Healthy
            else:
                scores[0] += 0.5  # Alzheimer's
                scores[2] += 0.5  # Parkinson's
        else:
            if feature_value <= threshold:
                scores[1] += 1  # Healthy
            else:
                scores[0] += 0.5  # Alzheimer's
                scores[2] += 0.5  # Parkinson's
    
    return np.argmax(scores)

def plot_curve(x, y, title="Curve Data", highlight_color='r', highlight_marker='o', highlight_line_style='-', highlight_line_width=2):
    """Plot the curve data with highlighted styling."""
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker=highlight_marker, linestyle=highlight_line_style, color=highlight_color, linewidth=highlight_line_width)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()

def main():
    # Load and prepare data
    train_image_paths, train_labels = get_labels(TRAIN_DATA_DIR)
    val_image_paths, val_labels = get_labels(VAL_DATA_DIR)
    test_image_paths, test_labels = get_labels(TEST_DATA_DIR)

    # Extract features
    train_features = extract_features_from_images(train_image_paths)
    val_features = extract_features_from_images(val_image_paths)
    test_features = extract_features_from_images(test_image_paths)

    # Determine thresholds
    thresholds = determine_thresholds(train_features, train_labels)

    # Classify using determined thresholds
    val_predictions = [threshold_classify(features, thresholds) for features in val_features]
    test_predictions = [threshold_classify(features, thresholds) for features in test_features]

    # Evaluate on validation set
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Evaluate on test set
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Print determined thresholds
    print("\nDetermined Thresholds:")
    for feature_name, (threshold, direction, accuracy) in thresholds.items():
        print(f"{feature_name}: {direction} than {threshold:.4f} (Accuracy: {accuracy:.4f})")

    # Plot the curve data for the first image in the training set
    if train_image_paths:
        img = cv2.imread(train_image_paths[0])
        if img is not None:
            x, y, highlighted_image = extract_curve_data(img)
            if x is not None and y is not None:
                # Display the image with the highlighted curve
                cv2.imshow("Highlighted Curve", highlighted_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()