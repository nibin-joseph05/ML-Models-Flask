import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/house_prices_dataset_knn.csv")

# Prepare data
X = df[["SquareFootage", "Bedrooms", "Bathrooms"]]
y = df["PriceCategory"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.4f}")

# Save model
with open("knn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("KNN model saved as knn_model.pkl")
