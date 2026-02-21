# AI-Based Student Performance Prediction

### MS Elevate AICTE Internship - Jan 2026

**Author:** Rohit121455  
**Date:** February 2026

## 📋 Overview

This project implements a machine learning system to predict student academic performance (pass/fail) based on key factors such as study hours, attendance percentage, and previous academic scores. The solution uses a Random Forest classifier to provide early warnings for at-risk students.

## 🎯 Problem Statement

Educational institutions often lack predictive tools to identify students who might fail before final examinations. This leads to:
- Delayed intervention opportunities
- Higher dropout rates
- Inefficient resource allocation
- Reduced academic success rates

## 💡 Proposed Solution

An AI-powered prediction system that:
- Analyzes historical student data
- Identifies patterns leading to academic success/failure
- Provides probability-based predictions
- Enables proactive academic support

## 🛠 Technology Stack

- **Programming Language:** Python 3.8+
- **Core Libraries:**
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning algorithms
  - `matplotlib` & `seaborn` - Data visualization
  - `joblib` - Model serialization

- **Algorithm:** Random Forest Classifier
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score

## 📊 Dataset

The project uses a synthetic dataset with the following features:
- **Study_Hours:** Weekly study hours (0-12)
- **Attendance:** Class attendance percentage (0-100)
- **Prev_Score:** Previous examination score (0-100)
- **Passed:** Target variable (0=Fail, 1=Pass)

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rohit121455/Student-Performance-Prediction.git
   cd Student-Performance-Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis:**
   ```bash
   python student_performance.py
   ```

## 📈 Usage

### Running the Complete Pipeline
```python
from student_performance import StudentPerformancePredictor

# Initialize predictor
predictor = StudentPerformancePredictor()

# Load data (or generate synthetic)
df = predictor.load_data()

# Run complete analysis
predictor.explore_data(df)
X_train, X_test, y_train, y_test = predictor.preprocess_data(df)
predictor.train_model(X_train, y_train)
predictor.evaluate_model(X_test, y_test)
```

### Making Predictions
```python
# Single prediction
result = predictor.predict(study_hours=8, attendance=85, prev_score=78)
print(f"Prediction: {result['prediction']}")
print(f"Pass Probability: {result['probability_pass']:.2%}")
```

## 📊 Results & Visualizations

The project generates several visualizations:
- **Correlation Matrix:** Feature relationships
- **Feature Distributions:** Data distribution analysis
- **Confusion Matrix:** Model performance visualization
- **Feature Importance:** Key predictors identification

## 🏗 Project Structure

```
Student-Performance-Prediction/
│
├── student_performance.py      # Main prediction script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── correlation_matrix.png      # Generated visualizations
├── confusion_matrix.png
├── feature_distributions.png
└── feature_importance.png
```

## 🔬 Methodology

1. **Data Generation/Loading:** Create or load student performance data
2. **Exploratory Data Analysis:** Understand data distributions and correlations
3. **Data Preprocessing:** Feature scaling and train-test split
4. **Model Training:** Random Forest classifier with hyperparameter tuning
5. **Model Evaluation:** Cross-validation and performance metrics
6. **Visualization:** Generate insights and model explanations
7. **Model Persistence:** Save trained model for future predictions

## 📈 Model Performance

- **Training Accuracy:** ~95% (varies with dataset size)
- **Cross-validation Score:** ~92%
- **Key Features:** Study Hours (most important), Previous Score, Attendance

## 🔮 Future Enhancements

- [ ] Real dataset integration
- [ ] Additional ML algorithms (SVM, Neural Networks)
- [ ] Hyperparameter optimization
- [ ] Web API deployment
- [ ] Real-time prediction dashboard
- [ ] Multi-class prediction (grades instead of pass/fail)

## 📝 License

This project is part of the MS Elevate AICTE Internship program.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Contact

**Rohit121455**  
GitHub: [Rohit121455](https://github.com/Rohit121455)
Educational institutions lack early-warning systems to identify students at risk of failure, preventing timely intervention.

## 2. Proposed Solution
A machine learning system that analyzes study habits and attendance to predict if a student will pass or fail.

## 3. Technology Stack
- Python, Pandas, Scikit-Learn
- Algorithm: Random Forest Classifier

## 4. How to Use
Run `python student_performance.py` to see the model evaluation.