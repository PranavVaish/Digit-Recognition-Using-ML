import pandas as pd
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import hog
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def load_hog_features(csv_path):
    print(f"Loading features from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Separate features, labels and file paths
    # Last two columns are 'label' and 'file_path'
    features = df.iloc[:, :-2].values
    labels = df['label'].values
    file_paths = df['file_path'].values

    print(f"Loaded {len(features)} samples with {features.shape[1]} features per sample")

    return features, labels, file_paths

def train_svm_model(features, labels, C=1.0, gamma='scale'):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train SVM classifier
    print(f"Training SVM classifier with C={C}, gamma={gamma}...")
    svm = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
    svm.fit(X_train_scaled, y_train)

    # Make predictions on test set
    y_pred = svm.predict(X_test_scaled)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('svm_confusion_matrix.png')
    plt.show()

    # Return the model and scaler for future use
    return svm, scaler, (X_train, X_test, y_train, y_test)

def extract_hog_from_image(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Resize to match training dimensions
    img = cv2.resize(img, (28, 28))

    # Extract HOG features (match parameters with training)
    features, hog_image = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        transform_sqrt=True
    )

    return features, hog_image

def optimize_svm_hyperparameters(X_train, y_train, X_test, y_test):
    """Find the optimal hyperparameters for SVM."""
    print("Optimizing SVM hyperparameters...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

    # Use GridSearchCV for hyperparameter tuning
    svm = SVC(kernel='rbf', probability=True)
    grid_search = GridSearchCV(
        svm, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1
    )

    # Train with grid search
    grid_search.fit(X_train_scaled, y_train)

    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")

    # Evaluate on test set
    best_svm = grid_search.best_estimator_
    test_accuracy = best_svm.score(X_test_scaled, y_test)
    print(f"Test accuracy with best parameters: {test_accuracy:.4f}")

    return best_params

if __name__ == "__main__":
    # Path to the CSV file containing HOG features
    csv_path = "./hog_features.csv"

    # Load features and labels
    features, labels, file_paths = load_hog_features(csv_path)

    # Split data for hyperparameter optimization
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Find optimal hyperparameters
    best_params = optimize_svm_hyperparameters(X_train, y_train, X_test, y_test)

    # Train SVM model with optimal hyperparameters
    svm_model, scaler, _ = train_svm_model(
        features, labels, C=best_params['C'], gamma=best_params['gamma']
    )

    # Save the trained model and scaler
    print("Saving model and scaler...")
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(scaler, 'svm_feature_scaler.pkl')

    print("\nDone! Model and scaler have been saved to 'svm_model.pkl' and 'svm_feature_scaler.pkl'")
