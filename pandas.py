import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
try:
    data = pd.read_csv("C:\\Users\\nirwa\\OneDrive\\Documents\\placement_data.csv")
except FileNotFoundError:
    print("The file 'placement_data.csv' was not found.")
    exit()

# Handle missing values if any
if data.isnull().sum().any():
    print("Warning: Missing values detected. Consider handling them appropriately.")
    data = data.dropna()  # Simple approach: removing rows with missing values

# One-Hot Encoding
if "Course" in data.columns:
    data = pd.get_dummies(data, columns=["Course"], drop_first=True)
else:
    print("Column 'Course' not found in data.")

# Define Features (X) and Target (y)
required_columns = ['CGPA', 'Course', 'Placement_Package']
if all([column in data.columns for column in required_columns]):
    X = data[['CGPA', 'Course']]
    y = data['Placement_Package']
else:
    print("One or more features/target columns are missing.")
    exit()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-Squared Value:", r2)

# Plot Predictions vs Actual
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Packages")
plt.ylabel("Predicted Packages")
plt.title("Actual vs Predicted Packages")
plt.show()

# Print Model Parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)