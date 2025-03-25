import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the model
with open("lr_model.pkl", "rb") as f:
    model_data = pickle.load(f)

# Check if it's a dictionary (contains features) or direct model
if isinstance(model_data, dict):
    model = model_data["model"]
    features = model_data["features"]
else:
    model = model_data
    features = ["SquareFootage", "Bedrooms", "Bathrooms"]  # Manually set feature names

# Load dataset
df = pd.read_csv("data/house_prices_dataset_lr.csv")

# Prepare data
X = df[features]
y = df["Expensive"]

# Split data
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Predict Example
input_data = pd.DataFrame([[1500, 3, 2]], columns=features)
prediction = model.predict(input_data)
print("Prediction:", prediction[0])  # Expensive or not
