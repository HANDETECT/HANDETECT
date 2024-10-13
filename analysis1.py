import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from alibi.explainers import CEM
import tensorflow as tf
from scipy.stats import ttest_ind

# Load features and labels from npz file
def load_features(npz_file):
    data = np.load(npz_file)
    curvatures = data['curvatures']
    radii = data['radii']
    growth_rates = data['growth_rates']
    labels = data['labels']
    return curvatures, radii, growth_rates, labels

# Extract feature means
def extract_features(curvatures, radii, growth_rates):
    curvature_means = np.mean(curvatures, axis=1)
    growth_rate_means = np.mean(growth_rates, axis=1)
    return curvature_means, radii, growth_rate_means

# Encode labels to numeric values
def encode_labels(labels):
    label_mapping = {'Healthy': 0, 'Parkinsons Disease': 1, 'Alzheimers Disease': 2}
    return np.array([label_mapping[label] for label in labels])

# Calculate and plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# Plot ROC curve and AUC
def plot_roc_curve(y_true, y_scores, labels):
    y_true_binarized = label_binarize(y_true, classes=np.arange(len(labels)))
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure()
    for i in range(len(labels)):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {labels[i]} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

# Perform t-test to compare features between classes
def perform_t_tests(features, labels):
    unique_labels = np.unique(labels)
    results = {}
    
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            if i >= j:
                continue
            group1 = [features[k] for k in range(len(labels)) if labels[k] == label1]
            group2 = [features[k] for k in range(len(labels)) if labels[k] == label2]
            
            if len(group1) == 0 or len(group2) == 0:
                continue

            group1 = np.array(group1)
            group2 = np.array(group2)

            # Perform t-test for each feature
            t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
            
            # Ensure the statistics are scalars (mean across features)
            if isinstance(t_stat, np.ndarray):
                t_stat = t_stat.mean()
            if isinstance(p_val, np.ndarray):
                p_val = p_val.mean()
            
            results[f'{label1} vs {label2}'] = (t_stat, p_val)
    
    return results

# Load a pre-trained CNN model (modify this to load your actual model)
def load_cnn_model():
    model = tf.keras.models.load_model('your_cnn_model.h5')
    return model

# Function to generate CEM explanations for the CNN model
def generate_cem_explanations(model, X_scaled):
    # Create an instance of the CEM explainer
    explainer = CEM(model, mode='PN', shape=(X_scaled.shape[1],))
    
    # Set parameters for CEM
    explainer.fit(X_scaled)  # Assuming CNN accepts preprocessed input

    # Choose an instance from the dataset to explain (first sample)
    instance = X_scaled[0].reshape(1, -1)

    # Generate the explanation
    explanation = explainer.explain(instance, verbose=False)

    # Extract the pertinent negative explanation (what features to change to change the prediction)
    print("CEM Explanation - Pertinent Negative:")
    print(explanation.PN)

# Main analysis and explanation workflow
def main_analysis(npz_file):
    # Load features and labels
    curvatures, radii, growth_rates, labels = load_features(npz_file)
    
    # Extract feature means
    curvature_means, radii, growth_rate_means = extract_features(curvatures, radii, growth_rates)
    
    # Prepare features and labels
    features = np.hstack([curvature_means.reshape(-1, 1), np.array(radii).reshape(-1, 1), growth_rate_means.reshape(-1, 1)])
    y = encode_labels(labels)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Initialize and fit SVM model (for classification)
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_scaled, y)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y, predictions, target_names=['Healthy', 'Parkinsons Disease', 'Alzheimers Disease']))
    
    # Plot confusion matrix
    print("Confusion Matrix:")
    plot_confusion_matrix(y, predictions, labels=['Healthy', 'Parkinsons Disease', 'Alzheimers Disease'])
    
    # Calculate and plot ROC curve and AUC
    print("ROC Curve:")
    plot_roc_curve(y, probabilities, labels=['Healthy', 'Parkinsons Disease', 'Alzheimers Disease'])
    
    # Perform t-tests
    print("T-Test Results:")
    t_test_results = perform_t_tests(features, labels)
    for comparison, (t_stat, p_val) in t_test_results.items():
        print(f"{comparison}: t-statistic = {t_stat:.2f}, p-value = {p_val:.2e}")

    # Load CNN model
    cnn_model = load_cnn_model()

    # Generate CEM explanations
    generate_cem_explanations(cnn_model, X_scaled)

# Main execution
main_analysis('spiral_features.npz')
