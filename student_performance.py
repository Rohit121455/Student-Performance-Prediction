"""
AI-Based Student Performance Prediction
=======================================

This project predicts student pass/fail status based on study hours, attendance, and previous scores
using machine learning algorithms.

Author: Rohit121455
Date: February 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StudentPerformancePredictor:
    """
    A class to handle student performance prediction using machine learning.
    """

    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the predictor with model parameters.

        Args:
            n_estimators (int): Number of trees in Random Forest
            random_state (int): Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['Study_Hours', 'Attendance', 'Prev_Score']

    def generate_dataset(self, n_samples=1000):
        """
        Generate a synthetic dataset for demonstration.

        Args:
            n_samples (int): Number of samples to generate

        Returns:
            pd.DataFrame: Generated dataset
        """
        np.random.seed(self.random_state)

        # Generate features with realistic distributions
        study_hours = np.random.normal(6, 2, n_samples).clip(0, 12)
        attendance = np.random.normal(75, 15, n_samples).clip(0, 100)
        prev_score = np.random.normal(70, 15, n_samples).clip(0, 100)

        # Create target variable with some correlation
        # Higher study hours, attendance, prev_score -> more likely to pass
        pass_probability = 1 / (1 + np.exp(-(0.3*study_hours + 0.02*attendance + 0.03*prev_score - 5)))
        passed = np.random.binomial(1, pass_probability, n_samples)

        data = {
            'Study_Hours': study_hours,
            'Attendance': attendance,
            'Prev_Score': prev_score,
            'Passed': passed
        }

        return pd.DataFrame(data)

    def load_data(self, data_path=None):
        """
        Load data from file or generate synthetic data.

        Args:
            data_path (str): Path to CSV file (optional)

        Returns:
            pd.DataFrame: Dataset
        """
        if data_path:
            return pd.read_csv(data_path)
        else:
            return self.generate_dataset()

    def explore_data(self, df):
        """
        Perform exploratory data analysis.

        Args:
            df (pd.DataFrame): Dataset to explore
        """
        print("=== DATA EXPLORATION ===")
        print(f"Dataset shape: {df.shape}")
        print("\nDescriptive Statistics:")
        print(df.describe())

        print("\nCorrelation Matrix:")
        corr = df.corr()
        print(corr)

        # Visualize correlations
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for i, col in enumerate(df.columns):
            if col != 'Passed':
                sns.histplot(df[col], ax=axes[i], kde=True)
                axes[i].set_title(f'Distribution of {col}')

        sns.countplot(x='Passed', data=df, ax=axes[3])
        axes[3].set_title('Pass/Fail Distribution')

        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def preprocess_data(self, df):
        """
        Preprocess the data for modeling.

        Args:
            df (pd.DataFrame): Raw dataset

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = df.drop('Passed', axis=1)
        y = df['Passed']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """
        Train the Random Forest model.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model.

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
        """
        y_pred = self.model.predict(X_test)

        print("\n=== MODEL EVALUATION ===")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(8, 4))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, filename='student_performance_model.pkl'):
        """
        Save the trained model to disk.

        Args:
            filename (str): Filename to save the model
        """
        if self.model:
            joblib.dump(self.model, filename)
            joblib.dump(self.scaler, 'scaler.pkl')
            print(f"\nModel saved as {filename}")

    def predict(self, study_hours, attendance, prev_score):
        """
        Make prediction for a single student.

        Args:
            study_hours (float): Study hours per week
            attendance (float): Attendance percentage
            prev_score (float): Previous score

        Returns:
            dict: Prediction results
        """
        if not self.model:
            raise ValueError("Model not trained yet. Call train_model() first.")

        # Scale input
        input_data = np.array([[study_hours, attendance, prev_score]])
        input_scaled = self.scaler.transform(input_data)

        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0]

        return {
            'prediction': 'Pass' if prediction == 1 else 'Fail',
            'probability_fail': probability[0],
            'probability_pass': probability[1]
        }

def main():
    """
    Main function to run the student performance prediction pipeline.
    """
    print("AI-Based Student Performance Prediction")
    print("=" * 50)

    # Initialize predictor
    predictor = StudentPerformancePredictor(n_estimators=200)

    # Load/generate data
    df = predictor.load_data()

    # Explore data
    predictor.explore_data(df)

    # Preprocess data
    X_train, X_test, y_train, y_test = predictor.preprocess_data(df)

    # Train model
    print("\n=== TRAINING MODEL ===")
    predictor.train_model(X_train, y_train)

    # Evaluate model
    predictor.evaluate_model(X_test, y_test)

    # Save model
    predictor.save_model()

    # Example prediction
    print("\n=== EXAMPLE PREDICTION ===")
    example = predictor.predict(study_hours=8, attendance=85, prev_score=78)
    print(f"Student with 8 study hours, 85% attendance, 78 prev score:")
    print(f"Prediction: {example['prediction']}")
    print(".2%")

if __name__ == "__main__":
    main()