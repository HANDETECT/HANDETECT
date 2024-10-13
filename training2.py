import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree instead of Random Forest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE  # For balancing the dataset
import pickle
from data_loader import CustomDataset, load_data  # Ensure this module is correctly defined
from scipy.fftpack import fft
from scipy.stats import kurtosis, skew

# Constants
BATCH_SIZE = 8
RANDOM_SEED = 123
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_DATA_DIR = r"data/train/"
VAL_DATA_DIR = r"data/val/"
TEST_DATA_DIR = r"data/test/"

# Function to ensure r_HT and r_ET are of the same length
def get_radii(curvature_features):
    midpoint = len(curvature_features) // 2
    r_HT = curvature_features[:midpoint]
    r_ET = curvature_features[midpoint:midpoint + len(r_HT)]  # Ensure same length as r_HT
    return r_HT, r_ET

# Add additional statistical features
def extract_advanced_features(r_HT, r_ET):
    r_HT = np.array(r_HT)
    r_ET = np.array(r_ET)
    
    if len(r_HT) != len(r_ET):
        min_len = min(len(r_HT), len(r_ET))
        r_HT = r_HT[:min_len]
        r_ET = r_ET[:min_len]

    # Calculate existing features
    n = len(r_HT)
    rms_difference = np.sqrt(np.sum((r_HT - r_ET) ** 2) / n)
    max_difference = np.max(np.abs(r_HT - r_ET))
    min_difference = np.min(np.abs(r_HT - r_ET))
    std_difference = np.std(r_HT - r_ET)
    relative_tremor = np.mean(np.abs(r_HT - np.roll(r_ET, 9))) if n > 9 else 0
    max_HT_radius = np.max(r_HT)
    min_HT_radius = np.min(r_HT)
    std_HT_radius = np.std(r_HT)
    changes_sign = np.count_nonzero(np.diff(np.sign(r_HT - r_ET)))

    # Additional features
    kurt_HT = kurtosis(r_HT)  # Kurtosis of HT radius
    skew_HT = skew(r_HT)  # Skewness of HT radius

    # FFT (frequency domain analysis)
    fft_HT = np.abs(fft(r_HT))[:10]  # Taking only the first 10 FFT coefficients

    # Combine all features into a single vector
    features = [
        rms_difference, max_difference, min_difference, std_difference,
        relative_tremor, max_HT_radius, min_HT_radius, std_HT_radius, changes_sign,
        kurt_HT, skew_HT
    ]
    
    # Append FFT features
    features.extend(fft_HT)
    
    return features

# Preprocess the dataset to extract features and labels
def preprocess_data(data_loader):
    features = []
    labels = []

    for batch in data_loader:
        images, batch_labels, curvature_features = batch

        for image, label, curvature in zip(images, batch_labels, curvature_features):
            r_HT, r_ET = get_radii(curvature)
            feature_vector = extract_advanced_features(r_HT, r_ET)
            features.append(feature_vector)
            labels.append(label.item())

    return np.array(features), np.array(labels)

# Define preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load and preprocess data
def load_and_preprocess_data():
    return load_data(TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR, preprocess)

# Main Decision Tree training loop with hyperparameter tuning and SMOTE
def main_training_loop():
    train_loader, val_loader, test_loader = load_and_preprocess_data()

    train_features, train_labels = preprocess_data(train_loader)
    val_features, val_labels = preprocess_data(val_loader)

    # Apply SMOTE to balance the dataset if needed
    smote = SMOTE(random_state=42)
    train_features_balanced, train_labels_balanced = smote.fit_resample(train_features, train_labels)

    # Set up Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=32, class_weight='balanced')

    # Hyperparameters to tune
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    dt_random = RandomizedSearchCV(estimator=dt, param_distributions=param_grid, 
                                   n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)

    # Fit the model on balanced data
    dt_random.fit(train_features_balanced, train_labels_balanced)

    # Cross-validate to ensure the model learns well on training data
    cv_scores = cross_val_score(dt_random, train_features_balanced, train_labels_balanced, cv=5)
    print(f"Cross-validation scores: {cv_scores}")

    # Evaluate on validation data
    val_predictions = dt_random.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_predictions)

    print(f'Validation Accuracy after tuning: {val_accuracy * 100:.2f}%')
    print("Classification Report on Validation Data:\n", classification_report(val_labels, val_predictions))

    # Save the trained Decision Tree model
    with open('decision_tree_model.pkl', 'wb') as f:
        pickle.dump(dt_random, f)

    # Getting decision thresholds
    val_probabilities = dt_random.predict_proba(val_features)
    print("Validation Class Probabilities:\n", val_probabilities)

# Evaluate the Decision Tree model on test data
def evaluate_on_test_data():
    _, _, test_loader = load_and_preprocess_data()

    test_features, test_labels = preprocess_data(test_loader)

    # Load the saved Decision Tree model
    with open('decision_tree_model.pkl', 'rb') as f:
        dt_model = pickle.load(f)

    test_predictions = dt_model.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print("Classification Report on Test Data:\n", classification_report(test_labels, test_predictions))

if __name__ == "__main__":
    main_training_loop()
    evaluate_on_test_data()
