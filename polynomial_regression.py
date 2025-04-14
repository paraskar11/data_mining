import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load dataset
data = pd.read_csv('/content/polynomial_regression.csv')

# Encode categorical columns
non_numeric_cols = data.select_dtypes(exclude=['number']).columns
le = LabelEncoder()
for col in non_numeric_cols:
    data[col] = le.fit_transform(data[col])

# Define input features and target variable
X = data.drop('CyclistCount', axis=1)
y = data['CyclistCount']

# Handle missing values in features
X.fillna(X.mean(), inplace=True)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Transform features to polynomial (degree = 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Fit a linear model to polynomial features
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predict on test set
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate performance
r2 = r2_score(y_test, y_pred_poly)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_poly))
mae = mean_absolute_error(y_test, y_pred_poly)

# Print results
print("Polynomial Regression Results:")
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")
