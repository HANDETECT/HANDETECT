import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

def load_features(npz_file):
    data = np.load(npz_file)
    curvatures = data['curvatures']
    radii = data['radii']
    growth_rates = data['growth_rates']
    labels = data['labels']
    return curvatures, radii, growth_rates, labels

def extract_features(curvatures, radii, growth_rates):
    curvature_means = np.mean(curvatures, axis=1)
    growth_rate_means = np.mean(growth_rates, axis=1)
    return np.column_stack((curvature_means, radii, growth_rate_means))

def encode_labels(labels):
    label_mapping = {'Healthy': 0, 'Parkinsons Disease': 1, 'Alzheimers Disease': 2}
    return np.array([label_mapping[label] for label in labels])

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def plot_data_distribution(labels):
    sns.countplot(x=labels)
    plt.title('Data Distribution')
    plt.xlabel('Disease Category')
    plt.ylabel('Count')
    plt.show()

def main_analysis(npz_file):
    # Load features
    curvatures, radii, growth_rates, labels = load_features(npz_file)

    # Extract features for the model
    features = extract_features(curvatures, radii, growth_rates)

    # Encode labels to numerical values
    encoded_labels = encode_labels(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.3, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Healthy', 'Parkinsons Disease', 'Alzheimers Disease']))

    # Plot confusion matrix
    print("Confusion Matrix:")
    plot_confusion_matrix(y_test, predictions, labels=['Healthy', 'Parkinsons Disease', 'Alzheimers Disease'])

    # Plot data distribution
    print("Data Distribution:")
    plot_data_distribution(labels)

    # Output the learned threshold values (model coefficients)
    print("\nLearned Coefficients (Thresholds):")
    print(f"Curvature Threshold: {model.coef_[:, 0]}")
    print(f"Radius Threshold: {model.coef_[:, 1]}")
    print(f"Growth Rate Threshold: {model.coef_[:, 2]}")
    print(f"Intercepts: {model.intercept_}")

# Main execution
main_analysis('spiral_features.npz')