import cv2
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from skimage.feature import hog

def extract_hog_features(img_path):
    """Extract HOG features from an image."""
    # Read the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return None, None
    
    # Store original image for display
    original = img.copy()
    
    # Resize image for consistent feature extraction
    img = cv2.resize(img, (28, 28))
    
    # Extract HOG features
    features, hog_image = hog(
        img, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        visualize=True,
        transform_sqrt=True
    )
    
    return features, hog_image, original, img

def predict_digit(image_path):
    """Predict digit from image using the trained model."""
    # Hardcoded paths for model and scaler
    model_path = "knn_model.pkl"
    scaler_path = "feature_scaler.pkl"
    
    # Load the model and scaler
    try:
        knn_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print(f"Error: Model or scaler file not found. Make sure {model_path} and {scaler_path} exist.")
        return
    
    # Extract HOG features
    features, hog_image, original, resized = extract_hog_features(image_path)
    
    if features is None:
        return
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = knn_model.predict(features_scaled)[0]
    probabilities = knn_model.predict_proba(features_scaled)[0]
    
    # Show results
    plt.figure(figsize=(15, 6))
    
    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Resized image
    plt.subplot(1, 4, 2)
    plt.imshow(resized, cmap='gray')
    plt.title('Resized to 28x28')
    plt.axis('off')
    
    # HOG visualization
    plt.subplot(1, 4, 3)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features')
    plt.axis('off')
    
    # Probability distribution
    plt.subplot(1, 4, 4)
    bars = plt.bar(range(10), probabilities)
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title(f'Prediction: {prediction}')
    plt.xticks(range(10))
    
    # Highlight the predicted digit
    bars[prediction].set_color('red')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()
    
    print(f"Predicted digit: {prediction}")
    print(f"Confidence: {probabilities[prediction]:.4f}")
    
    # Return prediction and confidence
    return prediction, probabilities[prediction]

def batch_predict(directory_path):
    """Predict digits for all images in a directory."""
    # Check if directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory {directory_path} not found.")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(directory_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No image files found in {directory_path}")
        return
    
    results = {}
    
    # Process each image
    for img_file in image_files:
        img_path = os.path.join(directory_path, img_file)
        print(f"\nProcessing {img_path}...")
        
        prediction, confidence = predict_digit(img_path)
        results[img_file] = (prediction, confidence)
    
    # Print summary
    print("\n===== Prediction Summary =====")
    for img_file, (pred, conf) in results.items():
        print(f"{img_file}: Digit {pred} (Confidence: {conf:.4f})")

if __name__ == "__main__":
    # OPTION 1: Predict a single image - UNCOMMENT AND MODIFY PATH
    image_path = "./image.png"  # Change this to your test image path
    predict_digit(image_path)
    
    # OPTION 2: Predict all images in a directory - UNCOMMENT AND MODIFY PATH
    #directory_path = "./Archive/testSet/"  # Change this to your test directory
    #batch_predict(directory_path)