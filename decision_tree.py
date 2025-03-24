import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

class Model:
    def __init__(self):
        # Set LOKY_MAX_CPU_COUNT to avoid joblib warnings
        os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust this to match your physical core count

        # Load the dataset
        file_path = "jupyter/OriginalDataset.csv"
        df = pd.read_csv(file_path)

        # Check for missing values and fill them with column means if necessary
        if df.isnull().sum().sum() > 0:
            print("Warning: Missing values detected! Filling them with column means.")
            df.fillna(df.mean(numeric_only=True), inplace=True)

        # Drop the 'Id' column if it exists
        if 'Id' in df.columns:
            df = df.drop(columns=['Id'])

        # Filter out careers with very few samples
        min_samples = 2  # Keep careers with at least 2 samples
        career_counts = df['Career'].value_counts()
        valid_careers = career_counts[career_counts >= min_samples].index
        df = df[df['Career'].isin(valid_careers)]

        # Ensure dataset is not empty
        if df.empty:
            raise ValueError("Filtered dataset is empty. Reduce `min_samples` threshold.")

        # Separate features and target variable
        X = df.drop(columns=['Career'])  # Features
        y = df['Career']  # Target

        # Encode the target labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)  # Convert career names to numerical labels
        self.label_encoder = label_encoder

        # Check the number of unique classes
        num_classes = len(np.unique(y_encoded))

        # Apply SMOTE only if conditions allow
        if num_classes > 1 and min(career_counts) >= 2:
            # Applying SMOTE with k_neighbors=1 to avoid errors...
            smote = SMOTE(k_neighbors=1, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
        else:
            # Skipping SMOTE due to low samples. Using original dataset.
            X_resampled, y_resampled = X, y_encoded  # Use the original dataset

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42
        )

        # Train the Random Forest Classifier with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=200,  # More trees for better performance
            max_depth=10,  # Limit depth to prevent overfitting
            class_weight="balanced",  # Handle class imbalance
            random_state=42,
            n_jobs=-1  # Use all available logical cores
        )
        self.model.fit(X_train, y_train)

    # Returns the name of a career based on the input values.
    def predict(self, vals):
        career_id = self.model.predict(vals)
        return self.label_encoder.inverse_transform(career_id)

def model():
    return Model()
