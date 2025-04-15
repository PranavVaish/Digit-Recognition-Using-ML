import pandas as pd
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import hog
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def load_hog_features(csv_path):
    """Load HOG features from CSV file."""
    print(f"Loading features from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Separate features, labels and file paths
    # Last two columns are 'label' and 'file_path'
    features = df.iloc[:, :-2].values
    labels = df['label'].values
    file_paths = df['file_path'].values
    
    print(f"Loaded {len(features)} samples with {features.shape[1]} features per sample")
    
    return features, labels, file_paths

def train_knn_model(features, labels, n_neighbors=5):
    """Train a KNN classifier using the provided features and labels."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train KNN classifier
    print(f"Training KNN classifier with k={n_neighbors}...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions on test set
    y_pred = knn.predict(X_test_scaled)
    
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
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Return the model and scaler for future use
    return knn, scaler, (X_train, X_test, y_train, y_test)

def extract_hog_from_image(image_path):
    """Extract HOG features from a new image for prediction."""
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

def predict_new_image(model, scaler, image_path):
    """Predict the class of a new image using the trained model."""
    # Extract HOG features
    features, hog_image = extract_hog_from_image(image_path)
    if features is None:
        return None
    
    # Scale features using the same scaler
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Display results
    plt.figure(figsize=(12, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # HOG visualization
    plt.subplot(1, 3, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features')
    plt.axis('off')
    
    # Probability distribution
    plt.subplot(1, 3, 3)
    plt.bar(range(10), probabilities)
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title(f'Prediction: {prediction} (Confidence: {probabilities[prediction]:.2f})')
    plt.xticks(range(10))
    
    plt.tight_layout()
    plt.show()
    
    return prediction, probabilities

def find_k_optimal(X_train, y_train, X_test, y_test):
    """Find the optimal value of k for KNN."""
    k_values = range(1, 21)
    accuracies = []
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for k in k_values:
        print(f"Testing k={k}...")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        accuracy = knn.score(X_test_scaled, y_test)
        accuracies.append(accuracy)
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'o-')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. k for KNN')
    plt.grid(True)
    plt.xticks(k_values)
    plt.savefig('knn_optimization.png')
    plt.show()
    
    # Find optimal k
    optimal_k = k_values[np.argmax(accuracies)]
    print(f"Optimal k: {optimal_k} with accuracy: {max(accuracies):.4f}")
    
    return optimal_k

if __name__ == "__main__":
    # Path to the CSV file containing HOG features
    csv_path = "./hog_features.csv"
    
    # Load features and labels
    features, labels, file_paths = load_hog_features(csv_path)
    
    # Split data for k optimization
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Find optimal k value
    optimal_k = find_k_optimal(X_train, y_train, X_test, y_test)
    
    # Train KNN model with optimal k
    knn_model, scaler, _ = train_knn_model(features, labels, n_neighbors=optimal_k)
    
    # Save the trained model and scaler
    print("Saving model and scaler...")
    joblib.dump(knn_model, 'knn_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    # Example of how to use the model for prediction
    # Comment this section if you don't want to test a specific image
    test_img_path = "./Archive/testSet/0/img_10.jpg"  # Change this to your test image path
    if os.path.exists(test_img_path):
        print(f"\nPredicting class for {test_img_path}...")
        prediction, probs = predict_new_image(knn_model, scaler, test_img_path)
        print(f"Predicted digit: {prediction}")
    else:
        print(f"\nTest image not found: {test_img_path}")
        print("To test with a new image, update the test_img_path variable.")
    
    print("\nDone! Model and scaler have been saved to 'knn_model.pkl' and 'feature_scaler.pkl'")