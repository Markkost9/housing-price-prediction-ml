# 🏠 Housing Price Prediction – Machine Learning Project

## 📌 Project Overview

This project focuses on predicting housing prices using machine learning models trained on structured real-estate data.

The objective was to:

* Perform full exploratory data analysis (EDA)
* Clean and preprocess the dataset
* Compare multiple regression models
* Evaluate model performance
* Analyze feature importance

The project demonstrates practical understanding of regression modeling, preprocessing techniques, and model evaluation.

---

## 📊 Dataset

The dataset contains housing-related features such as:

* Living Space
* Beds
* Baths
* Median Household Income
* ZIP code statistics
* Geographic coordinates
* Location-based categorical variables (City, State, County)

Target variable:

```
Price
```

Dataset size: ~40,000 records.

Dataset source: Kaggle – publicly available housing dataset used for educational and modeling purposes.

---

## 🔎 Exploratory Data Analysis

Performed:

* Missing value detection and median/mode imputation
* Duplicate detection and removal
* Outlier detection using IQR method
* Distribution analysis (with and without 99th percentile trimming)
* Pearson correlation matrix
* Scatter plots for key feature relationships

---

## ⚙️ Data Preprocessing

* Separation of numerical and categorical features
* Manual selection of relevant categorical variables
* One-hot encoding for categorical features
* 80/20 train-test split (`random_state=42`)
* Log transformation of target variable (for one model)

No data leakage was introduced during preprocessing.

---

## 🤖 Models Implemented

### 1️⃣ Linear Regression (Numerical + Categorical Features)

Baseline regression model using all relevant features.

---

### 2️⃣ Linear Regression (Numerical Features Only)

Used to evaluate the impact of categorical variables.

---

### 3️⃣ Linear Regression with Log-Transformed Target

Log transformation applied to reduce skewness and stabilize variance.

---

### 4️⃣ Random Forest Regressor

* 200 trees
* Parallel processing (`n_jobs=-1`)
* Captures nonlinear relationships
* Feature importance analysis performed

---

## 📈 Model Evaluation Metrics

Models were evaluated using:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score
* Actual vs Predicted visual comparison

Random Forest achieved the strongest predictive performance.

---

### Performance Summary

- Linear Regression (numerical + selected categorical features): R² = 0.3872
- Linear Regression (numerical features only): R² = 0.3879
- Linear Regression model with numerical features and log-transformed target variable: R² = 0.6217
- Random Forest: R² = 0.69

---

## 📊 Model Visualization

![Random Forest Prediction](random_forest_predictions.png)

---

## 🧠 Key Insights

* Housing prices exhibit right-skewed distribution.
* Log transformation improves linear model stability.
* Location-based features significantly impact predictions.
* Random Forest captures nonlinear relationships better than linear models.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/Markkost9/housing-price-prediction-ml.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python American_Housing_Price_Prediction.py
```

---

## 👨‍💻 Author

Marko Kostic
Data Science & Artificial Intelligence Student
2026

---
