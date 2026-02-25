"""
Housing Price Prediction Project

This project includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Linear Regression modeling
- Random Forest regression
- Model evaluation and comparison

Author: Marko Kostic
Year: 2026
"""
# Libraries for data processing and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import libraries for model training and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configure visualization settings
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Load dataset and perform initial data inspection
df = pd.read_csv("American_Housing_Data.csv")
print("\nDataset successfully loaded.")

print("\nPreview of the first 5 rows:")
print(df.head())

print("\nDataset structure and data types:")
print(df.info())

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nDescriptive statistics of numerical features:")
print(df.describe())

# Separate numerical and categorical features
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# Output selected feature lists
print("\nNumerical features:", numerical_cols)
print("\nCategorical features:", categorical_cols)

# Define the target variable and exclude it from numerical feature list
target = "Price"   # Target variable definition
numerical_cols.remove(target)

# Select relevant categorical features for modeling
important_cat_cols = ['City', 'State', 'County']

# Missing value handling and verification
print("\nChecking for missing values by column:")
print(df.isna().sum())

# numerical → median
for col in numerical_cols + [target]:
    df[col] = df[col].fillna(df[col].median())

# categorical → mode
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print(df.isna().sum())
print("\nAll missing values have been successfully imputed.")

# Duplicate detection and removal
print("\nNumber of duplicate rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("\nNumber of duplicate rows after removal:", df.duplicated().sum())
print("\nDuplicates have been successfully removed.")

# Outlier detection using IQR method
Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df[target] < lower_bound) | (df[target] > upper_bound)]

# Outlier visualization
plt.figure(figsize=(6,4))
sns.boxplot(y=df[target])
plt.title("Boxplot: Price")
plt.ylabel("Price")
plt.show()

# Display outlier boundaries and statistics
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")
print(f"Number of detected outliers: {outliers.shape[0]}")

# Descriptive statistics and visualizations
print(df[numerical_cols + [target]].describe().round(2))

# Distribution of categorical features
for col in categorical_cols:
    print(f"\nDistribution of column {col}:")
    print(df[col].value_counts())

# Visualization of selected numerical features

# Columns selected for distribution analysis
dist_numerical_cols = ["Living Space", "Median Household Income"]

for col in dist_numerical_cols + [target]:
    # Define upper percentile threshold (99th percentile) for visualization without extreme values
    upper = df[col].quantile(0.99)

    plt.figure(figsize=(12, 4))

    # Histogram including outliers
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"{col} – Distribution (with outliers)")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    # Histogram without outliers
    plt.subplot(1, 2, 2)
    sns.histplot(df[col][df[col] <= upper], bins=30, kde=True)
    plt.title(f"{col} – Distribution (without outliers)")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    print(df[col].describe().round(2))



for col in ["Beds", "Baths"]:
    # Define upper percentile threshold (99th percentile) for visualization without extreme values
    upper = df[col].quantile(0.99)

    # Define bin edges for discrete integer features
    min_v = int(df[col].min())
    max_v = int(df[col].max())
    max_v_trim = int(upper)

    bins_full = np.arange(min_v - 0.5, max_v + 1.5, 1)
    bins_trim = np.arange(min_v - 0.5, max_v_trim + 1.5, 1)

    plt.figure(figsize=(12, 4))

    # Histogram including outliers
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], bins=bins_full, kde=False)
    plt.xticks(range(min_v, max_v + 1))
    plt.title(f"{col} – Distribution (with outliers)")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    # Histogram without outliers
    plt.subplot(1, 2, 2)
    sns.histplot(df[col][df[col] <= upper], bins=bins_trim, kde=False)
    plt.xticks(range(min_v, max_v_trim + 1))
    plt.title(f"{col} – Distribution (without outliers)")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    print(df[col].describe().round(2))


# Correlation matrix using Pearson correlation coefficient
corr_matrix = df[numerical_cols + [target]].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Pearson Correlation Matrix of Numerical Features")
plt.show()

plt.figure(figsize=(12, 8))

#Living Space vs Price – including outliers
plt.subplot(2, 2, 1)
sns.scatterplot(x=df["Living Space"], y=df["Price"], alpha=0.4)
plt.title("Living Space vs Price – including outliers")
plt.xlabel("Living Space")
plt.ylabel("Price")


#Living Space vs Price – without outliers
# Define upper percentile threshold (99th percentile) for visualization without extreme values
upper_ls = df["Living Space"].quantile(0.99)
upper_price = df["Price"].quantile(0.99)

#
df_ls_trim = df[
    (df["Living Space"] <= upper_ls) &
    (df["Price"] <= upper_price)
]

plt.subplot(2, 2, 2)
sns.scatterplot(
    x=df_ls_trim["Living Space"],
    y=df_ls_trim["Price"],
    alpha=0.5
)
plt.title("Living Space vs Price – without outliers")
plt.xlabel("Living Space")
plt.ylabel("Price")
plt.xlim(df_ls_trim["Living Space"].min(), upper_ls)
plt.ylim(df_ls_trim["Price"].min(), upper_price)

#Median Household Income vs Price – including outliers
plt.subplot(2, 2, 3)
sns.scatterplot(
    x=df["Median Household Income"],
    y=df["Price"],
    alpha=0.4
)
plt.title("Median Household Income vs Price – including outliers")
plt.xlabel("Median Household Income")
plt.ylabel("Price")

#Median Household Income vs Price – without outliers
upper_income = df["Median Household Income"].quantile(0.99)

df_inc_trim = df[
    (df["Median Household Income"] <= upper_income) &
    (df["Price"] <= upper_price)
]

plt.subplot(2, 2, 4)
sns.scatterplot(
    x=df_inc_trim["Median Household Income"],
    y=df_inc_trim["Price"],
    alpha=0.5
)
plt.title("Median Household Income vs Price – without outliers")
plt.xlabel("Median Household Income")
plt.ylabel("Price")
plt.xlim(df_inc_trim["Median Household Income"].min(), upper_income)
plt.ylim(df_inc_trim["Price"].min(), upper_price)

plt.tight_layout()
plt.show()


# Prepare feature matrix (X) using selected numerical and relevant categorical features
X = df[numerical_cols + important_cat_cols]
y = df[target]

# Perform one-hot encoding to convert categorical variables into numerical format
X = pd.get_dummies(X, drop_first=True)


# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model, predict, and evaluate
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression (numerical + selected categorical features):")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² score: {r2:.4f}")

low_y = y_test.quantile(0.01)
high_y = y_test.quantile(0.99)

low_pred = np.quantile(y_pred, 0.01)
high_pred = np.quantile(y_pred, 0.99)
mask = (
    (y_test >= low_y) & (y_test <= high_y) &
    (y_pred >= low_pred) & (y_pred <= high_pred)
)

plt.figure(figsize=(12, 5))

#Scatterplot including outliers
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--"
)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Linear Regression – with outliers")

#Scatterplot bez outliera
plt.subplot(1, 2, 2)
plt.scatter(
    y_test[mask],
    y_pred[mask],
    alpha=0.5,
    s=20
)

min_lim = min(y_test[mask].min(), y_pred[mask].min())
max_lim = max(y_test[mask].max(), y_pred[mask].max())

plt.plot([min_lim, max_lim], [min_lim, max_lim], "r--")

plt.xlim(min_lim, max_lim)
plt.ylim(min_lim, max_lim)

plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Linear Regression – without outliers")


plt.tight_layout()
plt.show()


# Additional models for performance comparison

# Linear Regression model with numerical features only
num_cols_lr = [
    "Living Space",
    "Beds",
    "Baths",
    "Median Household Income",
    "Zip Code Population",
    "Zip Code Density",
    "Latitude",
    "Longitude"
]

X1 = df[num_cols_lr]
y1 = df[target]


X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

lin_reg1 = LinearRegression()
lin_reg1.fit(X1_train, y1_train)

y1_pred = lin_reg1.predict(X1_test)

mse1 = mean_squared_error(y1_test, y1_pred)
rmse1 = np.sqrt(mse1)
r2_1 = r2_score(y1_test, y1_pred)

print("\nLinear Regression (numerical features only):")
print(f"MSE: {mse1:.2f}")
print(f"RMSE: {rmse1:.2f}")
print(f"R² score: {r2_1:.4f}")

upper_y = y1_test.quantile(0.99)
upper_pred = np.quantile(y1_pred, 0.99)

mask = (y1_test <= upper_y) & (y1_pred <= upper_pred)

plt.figure(figsize=(12, 5))

#Scatterplot with outliers
plt.subplot(1, 2, 1)
plt.scatter(y1_test, y1_pred, alpha=0.4)
plt.plot(
    [y1_test.min(), y1_test.max()],
    [y1_test.min(), y1_test.max()],
    color="red",
    linestyle="--"
)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Linear Regression (numerical features only) – with outliers")

#Scatterplot without outliers
plt.subplot(1, 2, 2)
plt.scatter(y1_test[mask], y1_pred[mask], alpha=0.5)

min_lim = min(y1_test[mask].min(), y1_pred[mask].min())
max_lim = max(upper_y, upper_pred)

plt.plot([min_lim, max_lim], [min_lim, max_lim], "r--")
plt.xlim(min_lim, max_lim)
plt.ylim(min_lim, max_lim)

plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Linear Regression (numerical features only) – without outliers")

plt.tight_layout()
plt.show()


# Linear Regression model with numerical features and log-transformed target variable
X2 = df[num_cols_lr]        # select numerical features
y2 = np.log1p(df[target])   # apply log transformation to stabilize variance


X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

lin_reg2 = LinearRegression()
lin_reg2.fit(X2_train, y2_train)

y2_pred = lin_reg2.predict(X2_test)

mse2 = mean_squared_error(y2_test, y2_pred)
rmse2 = np.sqrt(mse2)
r2_2 = r2_score(y2_test, y2_pred)

print(f"\nLinear Regression model with numerical features and log-transformed target variable:")
print(f"MSE: {mse2:.2f}")
print(f"RMSE: {rmse2:.2f}")
print(f"R² score: {r2_2:.4f}")

upper_y = y2_test.quantile(0.99)
upper_pred = np.quantile(y2_pred, 0.99)

mask = (y2_test <= upper_y) & (y2_pred <= upper_pred)

plt.figure(figsize=(12, 5))

#Scatterplot with outliers
plt.subplot(1, 2, 1)
plt.scatter(y2_test, y2_pred, alpha=0.4)
plt.plot(
    [y2_test.min(), y2_test.max()],
    [y2_test.min(), y2_test.max()],
    color="red",
    linestyle="--"
)
plt.xlabel("Actual log price")
plt.ylabel("Predicted log price")
plt.title("Linear Regression (numerical features + log-target) – with outliers")

#Scatterplot without outliers
plt.subplot(1, 2, 2)
plt.scatter(y2_test[mask], y2_pred[mask], alpha=0.5)

min_lim = min(y2_test[mask].min(), y2_pred[mask].min())
max_lim = max(upper_y, upper_pred)

plt.plot([min_lim, max_lim], [min_lim, max_lim], "r--")
plt.xlim(min_lim, max_lim)
plt.ylim(min_lim, max_lim)

plt.xlabel("Actual log price")
plt.ylabel("Predicted log price")
plt.title("Linear Regression (numerical features + log-target) – without outliers")

plt.tight_layout()
plt.show()

# Analyze feature impact based on absolute coefficient magnitude
coef_df2 = pd.DataFrame({
    "Feature": X2.columns,
    "Coefficient": lin_reg2.coef_
}).sort_values(by="Coefficient", key=lambda x: abs(x), ascending=False)

print("\nCoefficients of the model with only numerical features and log-transformed target variable:")
print(coef_df2.head(10))

# Train Random Forest Regressor on numerical and relevant categorical features
X3 = df[numerical_cols + important_cat_cols]
y3 = df[target]
X3 = pd.get_dummies(X3, drop_first=True)
X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y3, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=None, n_jobs=-1)
rf_reg.fit(X3_train, y3_train)
y3_pred = rf_reg.predict(X3_test)
mse3 = mean_squared_error(y3_test, y3_pred)
rmse3 = np.sqrt(mse3)
r2_3 = r2_score(y3_test, y3_pred)
print("\nRandom Forest Regressor (numerical and selected categorical features):")
print(f"MSE: {mse3:.2f}")
print(f"RMSE: {rmse3:.2f}")
print(f"R² score: {r2_3:.4f}")

# Define 99th percentile threshold for zoomed-in visualization (outlier trimming for display)
upper_y3 = y3_test.quantile(0.99)
upper_pred3 = np.quantile(y3_pred, 0.99)

mask = (y3_test <= upper_y3) & (y3_pred <= upper_pred3)

plt.figure(figsize=(12, 5))

# With outliers
plt.subplot(1, 2, 1)
plt.scatter(y3_test, y3_pred, alpha=0.3, color="darkgreen")
plt.plot(
    [y3_test.min(), y3_test.max()],
    [y3_test.min(), y3_test.max()],
    "r--"
)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Random Forest – with outliers")

# Without outliers
plt.subplot(1, 2, 2)
plt.scatter(y3_test[mask], y3_pred[mask], alpha=0.4, color="darkgreen")
# Plot diagonal reference line adjusted to the zoomed visualization range
min_lim = min(y3_test[mask].min(), y3_pred[mask].min())
max_lim = max(upper_y3, upper_pred3)

plt.plot([min_lim, max_lim], [min_lim, max_lim], "r--")

plt.xlim(min_lim, max_lim)
plt.ylim(min_lim, max_lim)

plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Random Forest – without outliers")

plt.tight_layout()
plt.show()


# Analyze feature importance based on Random Forest impurity reduction
importances = rf_reg.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X3.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop features ranked by importance (Random Forest):")
print(importance_df.head(10))