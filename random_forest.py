import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_hog_features(csv_path="hog_features.csv"):
    #Load HOG features from CSV file.
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"HOG features file not found: {csv_path}")
    
    print(f"Loading HOG features from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Separate features, labels and file paths
    # Last two columns are 'label' and 'file_path'
    features = df.iloc[:, :-2].values
    labels = df['label'].values
    file_paths = df['file_path'].values
    
    print(f"Loaded {len(features)} samples with {features.shape[1]} HOG features per sample")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    return features, labels, file_paths

def train_random_forest(X, y, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """Train a Random Forest classifier using the provided HOG features and labels."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training Random Forest with {n_estimators} trees, max_depth={max_depth}, "
          f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}...")
    
    # Create and train Random Forest classifier
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    rf.fit(X_train_scaled, y_train)
    
    # Make predictions on test set
    y_pred = rf.predict(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return rf, scaler, (X_train_scaled, X_test_scaled, y_train, y_test, y_pred)

def plot_confusion_matrix(y_test, y_pred):
    #Plot and save confusion matrix.
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get unique class labels
    classes = sorted(np.unique(y_test))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Random Forest with HOG Features')
    plt.tight_layout()
    plt.savefig('rf_hog_confusion_matrix.png')
    print("Confusion matrix saved as 'rf_hog_confusion_matrix.png'")

def plot_feature_importance(rf_model, num_features=20):
    #Plot top feature importances from Random Forest model.
    # Get feature importances
    importances = rf_model.feature_importances_
    
    # Create feature indices
    indices = np.argsort(importances)[::-1]
    
    # Plot top feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Top Feature Importances - Random Forest with HOG Features')
    plt.bar(range(num_features), importances[indices[:num_features]], align='center')
    plt.xticks(range(num_features), indices[:num_features])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('rf_hog_feature_importance.png')
    print(f"Top {num_features} feature importance plot saved as 'rf_hog_feature_importance.png'")

def optimize_hyperparameters(X, y):
    #Find optimal hyperparameters for Random Forest using GridSearchCV.
    # Split data for optimization
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create Random Forest classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search
    print("Optimizing Random Forest hyperparameters...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    # Fit grid search
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    return best_params, scaler

def save_model(model, scaler, filename="random_forest_hog.pkl"):
    # Save the trained model and scaler using pickle 
    # Create a dictionary with both model and scaler
    model_data = {
        'model': model,
        'scaler': scaler
    }
    
    # Save model using pickle
    with open(filename, 'wb') as file:
        pickle.dump(model_data, file)
    
    print(f"Model and scaler saved to {filename}")

def main():
    # Path to the CSV file containing HOG features
    csv_path = "hog_features.csv"
    model_filename = "random_forest_hog.pkl"
    
    try:
        # Load HOG features and labels
        X, y, _ = load_hog_features(csv_path)
        
        # Find optimal hyperparameters
        print("\n1. Optimizing hyperparameters...")
        best_params, _ = optimize_hyperparameters(X, y)
        
        # Train the model with optimal hyperparameters
        print("\n2. Training final Random Forest model with optimal hyperparameters...")
        rf_model, final_scaler, (_, X_test_scaled, _, y_test, y_pred) = train_random_forest(
            X, y, 
            n_estimators=best_params['n_estimators'], 
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf']
        )
        
        # Plot confusion matrix
        print("\n3. Generating evaluation visualizations...")
        plot_confusion_matrix(y_test, y_pred)
        
        # Plot feature importance
        plot_feature_importance(rf_model)
        
        # Save the model
        print("\n4. Saving model...")
        save_model(rf_model, final_scaler, model_filename)
        
        print("\nRandom Forest model training complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run the HOG feature extraction script first to generate the 'hog_features.csv' file.")

if __name__ == "__main__":
    main()
