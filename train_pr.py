import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/house_prices_dataset_pr.csv")

# Remove negative prices
df = df[df["Price"] > 0]

# Log transformation for stability
df["Price"] = np.log1p(df["Price"])

# Remove extreme outliers (top 1%)
df = df[df["Price"] < df["Price"].quantile(0.99)]

# Prepare data
X = df[["SquareFootage"]]
y = df["Price"]

# Train-test split (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial transformation 
poly = PolynomialFeatures(degree=6)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict
y_pred = model.predict(X_test_poly)

# Calculate accuracy
accuracy = r2_score(y_test, y_pred)
print(f"Polynomial Regression Model Accuracy (R² Score): {accuracy:.4f}")

# Save model along with scaler and polynomial features
with open("pr_model.pkl", "wb") as f:
    pickle.dump((poly, scaler, model), f)

print("Polynomial Regression model saved as pr_model.pkl")

