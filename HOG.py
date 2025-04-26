import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import hog
from tqdm import tqdm

def extract_hog_features(img_path):
    """Extract HOG features from an image."""
    # Read the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Warning: Could not read image at {img_path}")
        return None
    
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
    
    return features

def process_directories(base_path, output_csv_path):
    """Process all images in directories 0-9 and extract HOG features."""
    # Lists to store features and labels
    all_features = []
    all_labels = []
    all_file_paths = []
    
    # Process each directory
    for label in range(10):
        dir_path = os.path.join(base_path, str(label))
        
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist. Skipping.")
            continue
        
        print(f"Processing directory {label}...")
        
        # Get all image files in the directory
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Process each image
        for img_file in tqdm(image_files, desc=f"Directory {label}"):
            img_path = os.path.join(dir_path, img_file)
            features = extract_hog_features(img_path)
            
            if features is not None:
                all_features.append(features)
                all_labels.append(label)
                all_file_paths.append(img_path)
    
    # Create a DataFrame
    print("Creating DataFrame...")
    
    # Convert features to DataFrame
    feature_df = pd.DataFrame(all_features)
    
    # Add label and file path columns
    feature_df['label'] = all_labels
    feature_df['file_path'] = all_file_paths
    
    # Save to CSV
    print(f"Saving CSV to {output_csv_path}...")
    feature_df.to_csv(output_csv_path, index=False)
    
    print(f"Completed! Extracted HOG features from {len(all_features)} images.")
    print(f"Feature dimension: {feature_df.shape[1]-2}")
    
    return feature_df

if __name__ == "__main__":
    # Define the base path where directories 0-9 are located
    base_path = "./Archive/trainingSet"  # Update this path as needed
    
    # Define the output CSV file path
    output_csv_path = "./hog_features.csv"
    
    # Process all directories and save features to CSV
    df = process_directories(base_path, output_csv_path)
    
    # Display some stats
    print("\nFeature extraction summary:")
    print(f"Total images processed: {len(df)}")
    print(f"Features per image: {df.shape[1]-2}")
    print(f"Distribution of labels:")
    print(df['label'].value_counts().sort_index())
