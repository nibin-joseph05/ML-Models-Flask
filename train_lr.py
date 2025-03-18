import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/house_prices_dataset_lr.csv")

# Prepare data
X = df[["SquareFootage", "Bedrooms", "Bathrooms"]]
y = df["Expensive"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")

# Save model
with open("lr_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Logistic Regression model saved as lr_model.pkl")
