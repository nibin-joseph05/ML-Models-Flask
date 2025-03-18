import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load Dataset
dataset_path = 'data/house_prices_dataset_mlr.csv'
data = pd.read_csv(dataset_path)

# Remove negative house prices
data = data[data['House_Price'] > 0]

# Define Independent (X) and Dependent (y) Variables
X = data[['Square_Footage', 'Bedrooms', 'Bathrooms', 'Age_of_House']]
y = data['House_Price']

# Split into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train MLR Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model trained successfully! Mean Squared Error: {mse:.2f}")
print(f"Model Accuracy (R² Score): {r2:.2f}")

# Save the Model
model_path = 'mlr_model.pkl'
joblib.dump(model, model_path)
print(f"Trained model saved at {model_path}")
