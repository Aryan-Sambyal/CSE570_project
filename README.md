# Predictive Maintenance of Industrial Equipment Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

## 1. Project Overview

Unexpected equipment failures can lead to significant financial losses, production delays, and operational downtime in manufacturing industries.

This project focuses on developing a **Machine Learning-based Predictive Maintenance System** that identifies potential industrial machine failures using operational parameters and sensor data.

Using the AI4I 2020 Predictive Maintenance Dataset, multiple classification algorithms are implemented, evaluated, and compared to determine their effectiveness in predicting machine failures.

The project also incorporates Explainable AI (XAI) techniques using SHAP to understand the influence of different operating parameters on model predictions.

## 2. Project Objectives

- Analyze industrial machine sensor and operational data.
- Perform Exploratory Data Analysis (EDA).
- Handle data preprocessing and feature engineering.
- Develop multiple Machine Learning classification models.
- Address class imbalance during model training.
- Compare models using appropriate evaluation metrics.
- Perform hyperparameter optimization.
- Identify important features influencing machine failure predictions.
- Apply Explainable AI techniques using SHAP.

## 3. Dataset Description

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

The dataset contains 10,000 observations representing industrial machine operating conditions.

### Dataset Features

| Feature | Description |
|---|---|
| UDI | Unique observation identifier |
| Product ID | Machine product identifier |
| Type | Product quality/type category |
| Air Temperature | Ambient air temperature |
| Process Temperature | Machine process temperature |
| Rotational Speed | Machine rotational speed |
| Torque | Operating torque |
| Tool Wear | Accumulated tool wear |

**Target Variable:** `Machine failure`

The target variable represents whether a machine experienced a failure:

- 0 – No Failure
- 1 – Machine Failure

### Dataset Statistics

| Parameter | Value |
|---|---:|
| Total Records | 10,000 |
| Total Original Columns | 14 |
| Missing Values | 0 |
| Duplicate Records | 0 |
| Failure Records | 339 |
| Normal Records | 9,661 |
| Failure Rate | 3.39% |

The dataset is highly imbalanced, making precision, recall, F1-score and PR-AUC important evaluation metrics.

## 4. Technologies and Libraries

**Programming Language**
- Python

**Data Analysis and Visualization**
- Pandas
- NumPy
- Matplotlib
- Seaborn

**Machine Learning**
- Scikit-learn
- XGBoost

**Explainable AI**
- SHAP (SHapley Additive exPlanations)

**Development Environment**
- Jupyter Notebook
- Anaconda

## 5. Project Workflow

The project follows a structured Machine Learning workflow:

1. Dataset Collection and Understanding
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Feature Selection and Engineering
5. Train-Test Splitting
6. Machine Learning Model Implementation
7. Model Evaluation and Comparison
8. Hyperparameter Tuning
9. Feature Importance Analysis
10. Explainable AI using SHAP
11. Model Saving and Documentation

## 6. Data Preprocessing

The following preprocessing operations were performed:

- Checked for missing values and duplicate records.
- Removed unnecessary identifier columns such as UDI and Product ID.
- Excluded individual failure-mode indicators (TWF, HDF, PWF, OSF, RNF) to prevent target leakage.
- Applied One-Hot Encoding to categorical features.
- Applied Standard Scaling to numerical features.
- Used a stratified 80:20 train-test split.
- Applied class weighting to supported classifiers to address class imbalance.

The final prediction features included:

- Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

## 7. Machine Learning Algorithms

Four classification algorithms were implemented and evaluated.

### 1. Logistic Regression

Used as a baseline classification algorithm to understand the relationship between operating parameters and machine failure.

### 2. Decision Tree Classifier

Used to identify nonlinear relationships and decision rules within the machine operating data.

### 3. Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to improve predictive performance and reduce overfitting.

### 4. XGBoost Classifier

An optimized gradient boosting algorithm used to improve classification performance, particularly for identifying minority-class machine failures.

## 8. Model Evaluation Metrics

The following evaluation metrics were used:

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Correctness of predicted failures |
| Recall | Ability to identify actual failures |
| F1-Score | Balance between precision and recall |
| ROC-AUC | Ability to distinguish between classes |
| PR-AUC | Performance on the minority failure class |

Since the dataset contains only 3.39% machine failure records, accuracy alone is not sufficient to evaluate model performance.

## 9. Model Performance Comparison

All four models were evaluated on the same held-out test dataset.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 82.45% | 14.18% | 82.35% | 24.19% | 90.70% | 38.18% |
| Decision Tree | 95.35% | 40.88% | 82.35% | 54.63% | 90.06% | 66.91% |
| Random Forest | 97.35% | 58.24% | 77.94% | 66.67% | 96.69% | 75.43% |
| XGBoost | 97.25% | 56.70% | 80.88% | 66.67% | 97.12% | 82.32% |

### Key Observations

- Random Forest achieved the highest accuracy of 97.35%.
- XGBoost achieved the highest ROC-AUC of 0.9712.
- XGBoost achieved the highest PR-AUC of 0.8232.
- XGBoost achieved a recall of 80.88%.
- Logistic Regression demonstrated high recall but comparatively low precision.

### Selected Model

XGBoost was selected as the primary model based on its superior PR-AUC and ROC-AUC performance.

Its ability to identify minority-class failures makes it a strong candidate for this predictive maintenance classification task.

## 10. Confusion Matrix Analysis

The XGBoost classifier produced the following results on the test dataset:

| | Predicted: No Failure | Predicted: Failure |
|---|---:|---:|
| Actual: No Failure | 1,890 | 42 |
| Actual: Failure | 13 | 55 |

### Interpretation

- True Negatives: 1,890
- False Positives: 42
- False Negatives: 13
- True Positives: 55

The model correctly identified 55 out of 68 actual machine failure cases, achieving approximately 80.88% recall.

## 11. Hyperparameter Optimization

RandomizedSearchCV was used to optimize the Random Forest classifier.

Stratified 3-fold cross-validation was applied with Average Precision (PR-AUC) as the optimization metric.

### Best Hyperparameters

| Parameter | Selected Value |
|---|---|
| Number of Estimators | 350 |
| Maximum Depth | None |
| Minimum Samples Split | 2 |
| Minimum Samples Leaf | 2 |

### Optimization Results

- Cross-validation PR-AUC: 0.7253
- Test PR-AUC: 0.7548

The optimized Random Forest achieved a small improvement over the original Random Forest model.

## 12. Feature Importance Analysis

Random Forest feature importance was used to identify the most influential operating parameters.

### Feature Importance Results

| Feature | Importance |
|---|---:|
| Torque | 31.81% |
| Rotational Speed | 29.67% |
| Tool Wear | 20.57% |
| Air Temperature | 9.74% |
| Process Temperature | 6.50% |

### Key Findings

Torque, rotational speed and tool wear were the three most influential features in the fitted Random Forest model.

These features contributed the most to the model's predictive decisions.

## 13. Explainable AI Using SHAP

SHAP (SHapley Additive exPlanations) was implemented to improve model interpretability.

A SHAP Summary Plot was generated to analyze how different feature values influence model predictions.

The analysis helps understand:

- Which features have the greatest impact on predictions.
- How higher and lower feature values influence prediction direction.
- The contribution of individual operating parameters to model outputs.

This provides additional interpretability beyond traditional feature importance.

## 14. Model Saving

The trained classification model was saved using Joblib.

Generated files:

- `predictive_maintenance_model.joblib`
- `model_comparison_results.csv`

These files allow the trained model and experimental results to be reused without retraining from scratch.

## 15. Conclusion

This project demonstrates the application of supervised Machine Learning techniques for predictive maintenance in industrial environments.

Four classification algorithms were implemented and evaluated using a structured preprocessing and model evaluation pipeline.

Among the evaluated models, XGBoost achieved the highest PR-AUC of 0.8232, while Random Forest achieved the highest overall accuracy of 97.35%.

The feature importance and SHAP analyses further demonstrated the influence of torque, rotational speed and tool wear on model predictions.

The project highlights how Machine Learning and Explainable AI can support data-driven maintenance decisions, potentially helping industries reduce unexpected equipment downtime.

## 16. Limitations

- The project uses a publicly available synthetic dataset rather than live industrial sensor streams.
- The dataset may not fully represent the operating conditions of real manufacturing environments.
- Model performance may vary across different industrial machines and operating conditions.
- Real-world deployment would require validation on actual equipment data.

## 17. Future Improvements

Potential future enhancements include:

- Integration with real-time industrial sensor data.
- Development of a web-based predictive maintenance dashboard.
- Implementation of anomaly detection techniques.
- Remaining Useful Life (RUL) prediction.
- Evaluation on additional industrial datasets.
- Deployment through a monitoring interface using Streamlit.

## 18. Repository Structure

```text
Predictive-Maintenance-ML/
│
├── Predictive_Maintenance_ML.ipynb
├── dataset.zip
├── requirements.txt
└── README.md
```

*Update filenames in this structure if your actual repository uses different names.*

## 19. How to Run the Project

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib
```

### Step 3: Extract the Dataset

Extract the AI4I dataset ZIP file and ensure that `ai4i2020.csv` is accessible to the notebook.

### Step 4: Open the Notebook

```bash
jupyter notebook
```

Open `Predictive_Maintenance_ML.ipynb`.

### Step 5: Execute the Notebook

Run all cells sequentially to reproduce the data analysis, model training, evaluation and explainability results.

---

## Author

**MTech Machine Learning Academic Project**

Developed using Python and Machine Learning techniques.

**Project Domain:** Industrial Predictive Maintenance

**Dataset Source:** UCI Machine Learning Repository
