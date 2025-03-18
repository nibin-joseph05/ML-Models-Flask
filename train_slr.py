import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Define the path to your dataset
dataset_path = os.path.join(os.path.dirname(__file__), 'data/house_prices_dataset_slr.csv')

# Load the dataset
data = pd.read_csv(dataset_path)

# Remove negative house prices
data = data[data['House_Price'] > 0]

# Extract independent (X) and dependent (y) variables
X = data[['Square_Footage']]
y = data['House_Price']  

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  # Calculate R² score

print(f"Model trained successfully! Mean Squared Error: {mse:.2f}")
print(f"Model Accuracy (R² Score): {r2:.2f}")  # Print R² score

# Save the trained model
model_path = os.path.join(os.path.dirname(__file__), 'slr_model.pkl')
joblib.dump(model, model_path)
print(f"Trained model saved at {model_path}")
