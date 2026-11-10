# Predictive Maintenance of Industrial Equipment Using Machine Learning

## 1. Project Overview

This project focuses on developing a Machine Learning-based Predictive Maintenance system to predict potential industrial machine failures using operational and sensor data.

Predictive maintenance helps industries reduce unexpected equipment breakdowns, minimize downtime, and improve operational efficiency by identifying possible machine failures before they occur.

The project uses the AI4I 2020 Predictive Maintenance Dataset and implements multiple classification algorithms to analyze and predict machine failure.

## 2. Project Objectives

- Analyze industrial machine sensor data.
- Perform Exploratory Data Analysis (EDA).
- Apply data preprocessing and feature engineering.
- Develop multiple Machine Learning classification models.
- Compare model performance using evaluation metrics.
- Optimize model hyperparameters.
- Understand important failure-related features using Explainable AI.

## 3. Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

The dataset contains approximately 10,000 observations with information about industrial machine operating conditions.

### Important Features

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type

**Target Variable:** Machine Failure

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost (optional model, if successfully executed)
- SHAP
- Jupyter Notebook

## 5. Machine Learning Algorithms

The project explores the following classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. XGBoost Classifier (if executed successfully)

## 6. Project Workflow

1. Dataset Collection and Understanding
2. Exploratory Data Analysis
3. Data Preprocessing
4. Feature Engineering
5. Train-Test Splitting
6. Machine Learning Model Implementation
7. Model Evaluation and Comparison
8. Hyperparameter Tuning
9. Explainable AI using SHAP
10. Model Saving

## 7. Model Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Since machine failures represent a minority class, particular attention is given to precision, recall and F1-score rather than accuracy alone.

## 8. Explainable AI

SHAP (SHapley Additive exPlanations) is used to understand how individual features influence model predictions.

The analysis highlights the relative contribution of operating parameters such as torque, rotational speed, tool wear and temperature.

## 9. Repository Structure

```text
Predictive-Maintenance-ML/
│
├── Predictive_Maintenance_ML.ipynb
├── dataset.zip
├── README.md
└── requirements.txt
```

## 10. How to Run the Project

1. Clone or download this repository.
2. Extract the dataset ZIP file.
3. Install the required Python libraries.
4. Open the Jupyter Notebook.
5. Update the dataset path if required.
6. Run the notebook cells sequentially.

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib
```

## 11. Results

The project evaluates multiple classification models and compares their predictive performance.

The final model comparison, classification reports, confusion matrices and SHAP visualizations are generated through the Jupyter Notebook.

Actual evaluation scores should be recorded here after reviewing the executed notebook results.

## 12. Limitations

- The project uses a publicly available dataset rather than live industrial sensor streams.
- The dataset is simulated and may not represent every real-world manufacturing environment.
- Model performance may vary when applied to different industrial machines and operating conditions.

## 13. Future Improvements

- Integrate real-time industrial sensor data.
- Explore additional anomaly detection techniques.
- Develop a web-based monitoring dashboard.
- Investigate remaining useful life prediction.
- Evaluate models on additional industrial datasets.

## 14. Project Status

**Status:** Core Machine Learning implementation completed.

This project was developed as part of an MTech Machine Learning course assignment.

## Author

Machine Learning Academic Project
