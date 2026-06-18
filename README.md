# Particle Physics Event Classification

## Project Overview

This project applies machine learning techniques to classify particle collision events as either **signal** (`s`) or **background** (`b`) using a dataset inspired by Higgs Boson detection experiments.

The objective is to build a robust classifier capable of identifying rare signal events from a much larger set of background events using engineered physics features.

---

## Dataset

The dataset contains:

* ~250,000 collision events
* 30+ physics-based features
* Event weights for evaluation
* Binary target labels:

  * `s` = signal
  * `b` = background

Several features contain missing values encoded as `-999`, which represent unavailable measurements rather than true numeric values.

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

Performed:

* Dataset inspection
* Missing value analysis
* Class distribution analysis
* Weight distribution analysis
* Feature distribution comparison
* Correlation analysis

Key findings:

* Signal events represent approximately 34% of the dataset.
* Jet-related variables contain structured missing values.
* Mass-related variables are among the strongest predictors.

---

### 2. Data Preparation

Steps performed:

* Replaced `-999` sentinel values with `NaN`
* Preserved missing-value information
* Prepared weighted training data
* Stratified validation strategy

---

### 3. Model Development

Model used:

**LightGBM Classifier**

Reasons for selection:

* Excellent performance on tabular data
* Native handling of missing values
* Fast training and inference
* Strong performance with weighted datasets

Training strategy:

* 5-Fold Stratified Cross Validation
* Sample weights included during training

---

### 4. Results

| Metric           | Score  |
| ---------------- | ------ |
| Weighted ROC-AUC | 0.9288 |

Cross-validation results:

* Mean ROC-AUC: 0.9288
* Standard Deviation: 0.0009

The low variance across folds indicates stable model performance.

---

## Feature Importance

The most influential features include:

* DER_mass_MMC
* DER_mass_transverse_met_lep
* PRI_met
* DER_deltar_tau_lep
* DER_mass_vis

These variables are closely related to the underlying physics processes and contribute strongly to signal-background separation.

![Feature Importance](images/feature_importance.png)

---

## Model Interpretation

SHAP (SHapley Additive exPlanations) was used to interpret model predictions and understand feature contributions.

![SHAP Summary](images/shap_summary.png)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM
* SHAP
* Matplotlib
* Seaborn
* Joblib

---

## Repository Structure

```text
particle-physics-classification/
│
├── notebooks/
│   └── EDA_Modeling.ipynb
│
├── models/
│   └── higgs_classifier.pkl
│
├── images/
│   ├── class_distribution.png
│   ├── missing_values.png
│   ├── feature_importance.png
│   └── shap_summary.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Future Improvements

Potential future work:

* Hyperparameter optimization
* Ensemble methods
* AMS optimization
* Additional feature engineering
* Automated experiment tracking

---

## Author

Built as a machine learning portfolio project focused on classification, model evaluation, and explainable AI techniques.
