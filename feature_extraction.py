import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import pickle
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

def extract_features_from_spiral(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply adaptive thresholding to binary image
    binary_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Assuming the largest contour is the spiral
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze()  # Remove redundant dimensions

    if contour.ndim == 1:
        contour = contour.reshape(-1, 2)  # Ensure contour is a 2D array
    
    # Compute curvature, radius, growth rate
    curvature, radius, growth_rate = compute_spiral_features(contour)
    
    return curvature, radius, growth_rate

def compute_curvature_spline(contour):
    x = contour[:, 0]
    y = contour[:, 1]
    spline_x = UnivariateSpline(np.arange(len(x)), x, k=3, s=0)
    spline_y = UnivariateSpline(np.arange(len(y)), y, k=3, s=0)
    d2x = spline_x.derivative(2)(np.arange(len(x)))
    d2y = spline_y.derivative(2)(np.arange(len(y)))
    curvature = np.abs(d2x * spline_y(np.arange(len(x))) - d2y * spline_x(np.arange(len(y)))) / (spline_x(np.arange(len(x)))**2 + spline_y(np.arange(len(y)))**2)**1.5
    return curvature

def compute_spiral_features(contour):
    if contour.shape[1] != 2:
        raise ValueError("Contour should be a 2D array with shape (n, 2)")

    # Compute the distance between successive points
    distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    
    # Compute radius (average distance from the centroid)
    origin = np.mean(contour, axis=0)
    radius = np.sqrt(np.sum((contour - origin) ** 2, axis=1))
    avg_radius = np.mean(radius)
    
    # Compute curvature
    curvature = compute_curvature_spline(contour)  # Using spline-based curvature

    # Smooth curvature to reduce noise
    curvature = gaussian_filter1d(curvature, sigma=2)

    # Compute growth rate (approximate rate of change of radius)
    growth_rate = np.diff(radius, prepend=radius[0])
    growth_rate = gaussian_filter1d(growth_rate, sigma=2)
    
    return curvature, avg_radius, growth_rate

def process_spiral_images(folder_path):
    features = []
    labels = []
    
    valid_classes = {'Alzheimers Disease', 'Parkinsons Disease', 'Healthy'}
    
    image_paths = []
    for disease_folder in os.listdir(folder_path):
        if disease_folder not in valid_classes:
            continue
        
        disease_path = os.path.join(folder_path, disease_folder)
        if os.path.isdir(disease_path):
            for filename in os.listdir(disease_path):
                if filename.endswith(".jpg"):
                    image_path = os.path.join(disease_path, filename)
                    image_paths.append((image_path, disease_folder))

    results = Parallel(n_jobs=-1)(delayed(extract_features_from_spiral)(path) for path, label in image_paths)
    
    for (curvature, radius, growth_rate), (path, label) in zip(results, image_paths):
        if curvature is not None:
            features.append({
                'filename': os.path.basename(path),
                'curvature': curvature,
                'radius': radius,
                'growth_rate': growth_rate
            })
            labels.append(label)
    
    return features, labels

def pad_or_truncate(array, target_length, pad_value=0):
    if len(array) > target_length:
        return array[:target_length]
    else:
        return np.pad(array, (0, target_length - len(array)), 'constant', constant_values=pad_value)

def generate_statistics_from_npz(npz_file, pkl_file):
    # Load data from the .npz file
    data = np.load(npz_file)
    curvatures = data['curvatures']
    labels = data['labels']
    
    # Print the unique labels in the dataset
    unique_labels = set(labels)
    print(f"Unique labels found in the dataset: {unique_labels}")

    # Adjust the dictionary keys based on the unique labels
    curvature_values = {label: [] for label in unique_labels}

    # Calculate mean curvature for each image and group by label
    for i, label in enumerate(labels):
        curvature = np.mean(curvatures[i])  # Adjust this if you need a different statistic
        curvature_values[label].append(curvature)

    # Compute mean and standard deviation for each label
    curvature_mean = {label: np.mean(curvature_values[label]) for label in curvature_values}
    curvature_std = {label: np.std(curvature_values[label]) for label in curvature_values}

    # Save the statistics to a .pkl file
    with open(pkl_file, 'wb') as f:
        pickle.dump({'mean': curvature_mean, 'std': curvature_std}, f)

    print(f"Curvature statistics saved to {pkl_file}")

# Main execution
folder_path = r'data/train'
features, labels = process_spiral_images(folder_path)

# Define a maximum length for padding or truncating
max_curvature_length = min(1024, max(len(feature['curvature']) for feature in features))
max_growth_rate_length = min(1024, max(len(feature['growth_rate']) for feature in features))

# Convert features to numpy arrays with padding/truncation
curvatures = np.array([pad_or_truncate(feature['curvature'], max_curvature_length) for feature in features])
radii = np.array([feature['radius'] for feature in features])
growth_rates = np.array([pad_or_truncate(feature['growth_rate'], max_growth_rate_length) for feature in features])

# Save features and labels to npz file
np.savez('spiral_features.npz', 
         curvatures=curvatures, 
         radii=radii, 
         growth_rates=growth_rates, 
         labels=labels)

print(f"Processed {len(features)} images.")
print(f"Features saved to 'spiral_features.npz'")
print(f"Curvatures shape: {curvatures.shape}")
print(f"Radii shape: {radii.shape}")
print(f"Growth rates shape: {growth_rates.shape}")
print(f"Number of labels: {len(labels)}")

# Generate statistics
generate_statistics_from_npz('spiral_features.npz', 'curvature_stats_epoch_30.pkl')
