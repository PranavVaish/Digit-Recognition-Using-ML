import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def load_model(model_path="random_forest_hog.pkl"):
    """Load the trained Random Forest model and scaler."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as file:
        model_data = pickle.load(file)
    
    rf_model = model_data['model']
    scaler = model_data['scaler']
    
    print("Model loaded successfully!")
    return rf_model, scaler

def extract_hog_from_image(image_path, img_size=(28, 28)):
    """Extract HOG features from an image."""
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # Resize to match training dimensions
    img = cv2.resize(img, img_size)
    
    # Extract HOG features (use same parameters as during training)
    features, hog_image = hog(
        img, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        visualize=True,
        transform_sqrt=True
    )
    
    return img, features, hog_image

def predict_single_image(model, scaler, image_path):
    """Predict the digit in a single image and visualize the result."""
    # Extract HOG features
    img, features, hog_image = extract_hog_from_image(image_path)
    if features is None:
        return None
    
    # Scale features using the saved scaler
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    # Get probability distribution across all classes
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Display results
    plt.figure(figsize=(15, 5))
    
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
    plt.xticks(range(10))
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title(f'Prediction: {prediction} (Confidence: {probabilities[prediction]:.2f})')
    
    plt.tight_layout()
    plt.show()
    
    return prediction, probabilities

def test_image_folder(model, scaler, test_dir, limit=None):
    """Test the model on all images in a directory structure."""
    true_labels = []
    predicted_labels = []
    image_paths = []
    
    print(f"Testing model on images from {test_dir}...")
    
    # Iterate through class directories
    for label_folder in sorted(os.listdir(test_dir)):
        label_path = os.path.join(test_dir, label_folder)
        
        # Skip if not a directory
        if not os.path.isdir(label_path):
            continue
        
        # Try to convert folder name to integer label
        try:
            true_label = int(label_folder)
        except ValueError:
            print(f"Skipping folder {label_folder}: Not a numeric label")
            continue
        
        print(f"Processing test images for class {true_label}...")
        
        # Process each image in the label folder
        count = 0
        for img_file in os.listdir(label_path):
            if limit is not None and count >= limit:
                break
                
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(label_path, img_file)
                
                # Extract features and predict
                img, features, _ = extract_hog_from_image(img_path)
                if features is None:
                    continue
                
                # Scale features
                features_scaled = scaler.transform([features])
                
                # Predict
                prediction = model.predict(features_scaled)[0]
                
                # Save results
                true_labels.append(true_label)
                predicted_labels.append(prediction)
                image_paths.append(img_path)
                
                count += 1
    
    print(f"Tested {len(true_labels)} images")
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels))
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Test Results')
    plt.tight_layout()
    plt.savefig('test_confusion_matrix.png')
    plt.show()
    
    # Return the results
    return true_labels, predicted_labels, image_paths

def display_misclassified(model, scaler, true_labels, predicted_labels, image_paths, max_images=10):
    """Display misclassified images."""
    # Find misclassified images
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)) if true != pred]
    
    if not misclassified_indices:
        print("No misclassified images found!")
        return
    
    print(f"Found {len(misclassified_indices)} misclassified images")
    
    # Display up to max_images
    num_to_display = min(max_images, len(misclassified_indices))
    
    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=num_to_display, ncols=3, figsize=(15, 5*num_to_display))
    
    for i in range(num_to_display):
        idx = misclassified_indices[i]
        img_path = image_paths[idx]
        true_label = true_labels[idx]
        pred_label = predicted_labels[idx]
        
        # Extract features
        img, features, hog_image = extract_hog_from_image(img_path)
        
        # Get probabilities
        features_scaled = scaler.transform([features])
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Display original image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'True: {true_label}, Predicted: {pred_label}')
        axes[i, 0].axis('off')
        
        # Display HOG image
        axes[i, 1].imshow(hog_image, cmap='gray')
        axes[i, 1].set_title('HOG Features')
        axes[i, 1].axis('off')
        
        # Display probability distribution
        axes[i, 2].bar(range(10), probabilities)
        axes[i, 2].set_xticks(range(10))
        axes[i, 2].set_title(f'Probability Distribution')
        axes[i, 2].set_xlabel('Digit')
        axes[i, 2].set_ylabel('Probability')
    
    plt.tight_layout()
    plt.savefig('misclassified_images.png')
    plt.show()

def main():
    # Path to the trained model
    model_path = "random_forest_hog.pkl"
    
    # Path to test images
    test_dir = "./DataSet/testSet"  # Change this to your test dataset path
    
    try:
        # Load the model
        rf_model, scaler = load_model(model_path)
        print(" Testing for single image")
        # Test single image
        image_path = input("Enter the path to the image file: ")
        if os.path.exists(image_path):
            prediction, probs = predict_single_image(rf_model, scaler, image_path)
            print(f"Predicted digit: {prediction}")
        else:
            print(f"Image not found: {image_path}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have trained the model first.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()