import os
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

def extract_hog_from_image(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None

    # Resize to match training dimensions (28x28)
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

    return features, hog_image, img

def predict_image(svm_model, scaler, image_path, show_visualization=True):
    # Extract HOG features
    features, hog_image, img = extract_hog_from_image(image_path)
    if features is None:
        return None

    # Scale features using the saved scaler
    features_scaled = scaler.transform([features])

    # Make prediction
    prediction = svm_model.predict(features_scaled)[0]
    probabilities = svm_model.predict_proba(features_scaled)[0]

    if show_visualization:
        # Display results
        plt.figure(figsize=(12, 5))

        # Original image
        plt.subplot(1, 3, 1)
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

def main():
    # Load saved model and scaler
    model_path = 'svm_model.pkl'
    scaler_path = 'svm_feature_scaler.pkl'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Model or scaler file not found. Make sure both files exist:")
        print(f"  - {model_path}")
        print(f"  - {scaler_path}")
        return

    print("Loading SVM model and feature scaler...")
    svm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully!")

    while True:
        print("\n==== SVM Digit Recognition - Test Menu ====")
        print("1. Test on a image")
        print("2. Exit")

        choice = input("Enter your choice (1-2): ")

        if choice == '1':
            image_path = input("Enter the path to the image: ")
            if os.path.exists(image_path):
                prediction, probs = predict_image(svm_model, scaler, image_path)
                print(f"Predicted digit: {prediction}")
                print(f"Probability distribution: {probs}")
            else:
                print(f"Error: Image not found at {image_path}")

        elif choice == '2':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 2.")

if __name__ == "__main__":
    main()
