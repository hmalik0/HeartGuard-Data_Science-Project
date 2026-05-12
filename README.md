# HeartGuard — Heart Disease Risk Stratification
**CS439 Final Project**

## Overview
HeartGuard is a hybrid machine learning pipeline for clinical heart disease risk stratification. The pipeline integrates unsupervised K-Means clustering with supervised classification to predict the presence of heart disease from routine clinical measurements. SHAP analysis provides interpretable explanations for model predictions.

## Dataset
- **Source:** UCI Machine Learning Repository — Cleveland Heart Disease Dataset
- **Link:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **Size:** 295 patients, 13 clinical features
- **Target:** Binary — 0 (no disease), 1 (disease present)

## Pipeline
1. **Data Cleaning** — Binarize target variable, reset index
2. **Exploratory Data Analysis** — Feature distributions, correlation analysis
3. **Preprocessing** — One-hot encoding, StandardScaler normalization, stratified 80/20 train/test split
4. **K-Means Clustering** — Discover patient risk profiles (k=3 via elbow method), append cluster label as engineered feature
5. **Classification** — Train and compare Logistic Regression, Random Forest, and XGBoost
6. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, SHAP values, PCA visualization

## Results
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.780 | 0.783 | 0.692 | 0.735 | 0.902 |
| Random Forest | 0.831 | 0.864 | 0.731 | 0.792 | 0.939 |
| XGBoost | **0.864** | 0.846 | **0.846** | **0.846** | 0.930 |

## Project Structure
```text
HeartGuard - Intro to DS Project/
├── data/
│   ├── heart_disease_raw.csv       # Original UCI dataset
│   ├── heart_disease_clean.csv     # Cleaned dataset
│   ├── X_train.csv                 # Training features
│   ├── X_test.csv                  # Test features
│   ├── y_train.csv                 # Training labels
│   ├── y_test.csv                  # Test labels
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── model_lr.pkl                # Trained Logistic Regression
│   ├── model_rf.pkl                # Trained Random Forest
│   ├── model_xgb.pkl               # Trained XGBoost
│   └── model_kmeans.pkl            # Fitted K-Means
├── notebooks/
│   ├── data_cleaning.ipynb         # Data cleaning
│   ├── eda.ipynb                   # Exploratory data analysis
│   ├── preprocessing.ipynb         # Encoding, scaling, splitting
│   ├── modeling.ipynb              # Clustering and classification
│   └── evaluation.ipynb            # Metrics and visualizations
├── figures/                        # All generated plots         
│   ├── confusion_matrices.png                 
│   ├── correlation.png         
│   ├── elbow.png              
│   ├── feature_overview.png
│   ├── pca_clusters.png                   
│   ├── roc_curves.png         
│   ├── shap.png                          
├── report/                         # Final paper (LaTeX)
├── requirements.txt                # Python dependencies
└── README.md
```

## Setup & Reproducibility
```bash
# Clone the repository
git clone https://github.com/hmalik0/HeartGuard-Data_Science-Project.git
cd HeartGuard-Data_Science-Project

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
# 1. data_cleaning.ipynb
# 2. phase1_eda.ipynb
# 3. phase2_preprocessing.ipynb
# 4. phase3_modeling.ipynb
# 5. phase4_evaluation.ipynb
```

## Key Findings
- XGBoost achieves the best overall performance with 86.4% accuracy and 0.846 F1 score
- Random Forest achieves the highest ROC-AUC of 0.939
- SHAP analysis identifies chest pain type, ST depression, and vessel blockage as the strongest predictors
- K-Means clustering reveals three distinct patient risk groups with disease rates of 73.1%, 27.8%, and 18.9%

## Technologies
Python 3.13 | pandas | numpy | scikit-learn | XGBoost | SHAP | matplotlib | seaborn
