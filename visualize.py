import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.fftpack import fft
from scipy.stats import entropy, skew, kurtosis
import pickle

def save_curvature_statistics(mean_curvature, std_curvature, epoch, filename='curvature_stats_epoch_{}.pkl'):
    """Save curvature statistics to a pickle file."""
    with open(filename.format(epoch), 'wb') as f:
        pickle.dump({'mean': mean_curvature, 'std': std_curvature}, f)


def compute_regression_features(curvature, window_size=10):
    """Compute regression features from curvature data."""
    if curvature is None or len(curvature) < window_size:
        print("Insufficient curvature data for regression features.")
        return 0, 0

    num_segments = len(curvature) // window_size
    r2_scores = []

    for i in range(num_segments):
        start = i * window_size
        end = start + window_size
        segment = curvature[start:end]
        
        # Avoid fitting if the segment has less than the window size
        if len(segment) < window_size:
            continue

        # Fit a linear model
        X = np.arange(len(segment)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, segment)
        r2_score = model.score(X, segment)  # R^2 score
        r2_scores.append(r2_score)
        print(f"Segment {i}: R2 Score = {r2_score}")  # Debugging line

    mean_r2 = np.mean(r2_scores) if r2_scores else 0
    std_r2 = np.std(r2_scores) if r2_scores else 0
    print(f"Mean R2: {mean_r2}, Std R2: {std_r2}")  # Debugging line
    return mean_r2, std_r2


def compute_features(image):
    """Extract features from an image."""
    radius = compute_radius(image)
    
    if np.isnan(radius):
        print("NaN found in radius!")

    curvature = compute_curvature_from_image(image)

    if np.isnan(curvature).any():
        print("NaN found in curvature from image!")

    fourier_features = compute_fourier_features(curvature)
    
    if np.isnan(fourier_features).any():
        print("NaN found in Fourier features!")

    curvature_entropy, curvature_skewness, curvature_kurtosis = compute_entropy_skewness_kurtosis(curvature)
    
    if np.isnan(curvature_entropy) or np.isnan(curvature_skewness) or np.isnan(curvature_kurtosis):
        print("NaN found in statistical features!")

    num_inversions = compute_inversions(curvature)
    
    if np.isnan(num_inversions):
        print("NaN found in inversions!")

    mean_r2, std_r2 = compute_regression_features(curvature)

    if np.isnan(mean_r2) or np.isnan(std_r2):
        print("NaN found in regression features!")

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


def compute_fourier_features(curvature):
    valid_curvature = curvature[~np.isnan(curvature)]
    if len(valid_curvature) == 0:
        return np.zeros(5)  
    fft_values = fft(valid_curvature)
    magnitude_spectrum = np.abs(fft_values)[:len(fft_values)//2]
    return magnitude_spectrum[:5]  


def compute_entropy_skewness_kurtosis(curvature):
    """Compute entropy, skewness, and kurtosis of curvature data."""
    valid_curvature = curvature[~np.isnan(curvature)]
    if len(valid_curvature) < 2 or np.all(valid_curvature == valid_curvature[0]):
        return 0, 0, 0
    curvature_entropy = entropy(np.histogram(valid_curvature, bins=10, density=True)[0])
    curvature_skewness = skew(valid_curvature)
    curvature_kurtosis = kurtosis(valid_curvature)

    return curvature_entropy, curvature_skewness, curvature_kurtosis


def compute_inversions(curvature):
    """Compute the number of inversions in curvature data."""
    # Ensure curvature is valid
    if curvature is None or len(curvature) < 2:
        print("Curvature data is invalid for inversion calculation.")
        return 0

    sign_changes = np.diff(np.sign(curvature))
    return np.count_nonzero(sign_changes)



def compute_radius(image):
    """Compute the radius of the spiral from the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  

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
        return np.zeros(len(spiral_points))  

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denominator = (dx ** 2 + dy ** 2) ** (3/2)
    if np.all(denominator == 0):  
        return np.zeros(len(spiral_points))
    curvature = (dx * ddy - dy * ddx) / (denominator + 1e-9)

    return curvature


def extract_curve_data(image):
    """Extract curve data from the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  

    binary = binary.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None  

    contour = max(contours, key=cv2.contourArea)
    points = contour[:, 0]  
    x = points[:, 0]
    y = points[:, 1]

    return x, y


def compute_curvature_from_image(image):
    """Compute curvature from the image."""
    x, y = extract_curve_data(image)
    if x is None or y is None:
        return np.zeros(100)  
    spiral_points = np.vstack((x, y)).T  
    return compute_curvature(spiral_points)  


def visualize_features(image, curvature, fourier_features, curvature_entropy, curvature_skewness, curvature_kurtosis, num_inversions, mean_r2, std_r2):
    """Visualize the extracted features from the image."""
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    
    # Plot original image
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Plot curvature
    axs[0, 1].plot(curvature)
    axs[0, 1].set_title('Curvature')

    # Plot Fourier features
    axs[1, 0].bar(np.arange(len(fourier_features)), fourier_features)
    axs[1, 0].set_title('Fourier Features')

    # Plot statistical features: Entropy, Skewness, Kurtosis
    axs[1, 1].bar(['Entropy', 'Skewness', 'Kurtosis'], [curvature_entropy, curvature_skewness, curvature_kurtosis])
    axs[1, 1].set_title('Statistical Features')

    # Plot number of inversions
    axs[2, 0].bar(['Inversions'], [num_inversions])
    axs[2, 0].set_title('Number of Inversions')

    # Plot regression features
    axs[2, 1].bar(['Mean R2', 'Std R2'], [mean_r2, std_r2])
    axs[2, 1].set_title('Regression Features (R2)')

    plt.tight_layout()
    plt.show()


def compute_regression_features(curvature, window_size=2):  # Adjusted window size
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

def compute_and_visualize(image):
    """Extract features and visualize them."""
    curvature = compute_curvature_from_image(image)

    if curvature is None or len(curvature) == 0:
        print("Curvature data is empty or None!")
        return

    print(f"Curvature Length: {len(curvature)}")
    print(f"Curvature Data: {curvature}")

    fixed_length = 10
    if len(curvature) < fixed_length:
        curvature = np.pad(curvature, (0, fixed_length - len(curvature)), mode='constant')

    fourier_features = compute_fourier_features(curvature)
    curvature_entropy, curvature_skewness, curvature_kurtosis = compute_entropy_skewness_kurtosis(curvature)
    num_inversions = compute_inversions(curvature)

    print(f"Number of Inversions: {num_inversions}")

    mean_r2, std_r2 = compute_regression_features(curvature)

    print(f"Mean R2: {mean_r2}, Std R2: {std_r2}")

    visualize_features(image, curvature, fourier_features, curvature_entropy, curvature_skewness, curvature_kurtosis, num_inversions, mean_r2, std_r2)


# Example usage:
# Assuming `image` is an image loaded via OpenCV, we can call `compute_and_visualize` to extract and visualize features:

if __name__ == '__main__':
    image_path = "C:\\Users\\User\\Desktop\\SpiralSense\\data\\train\\Parkinsons Disease\\Parkinsons Disease_train_244.jpg"
    image = cv2.imread(image_path)  # Load the specified image

    if image is None:
        print(f"Error: Image not found at {image_path}. Please check the path and file integrity.")
        exit()

    # Visualize the feature extraction process
    compute_and_visualize(image)

