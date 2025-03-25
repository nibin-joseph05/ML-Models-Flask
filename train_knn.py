import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/house_prices_dataset_knn.csv")

# Ensure correct data types
df["PriceCategory"] = df["PriceCategory"].astype(int)

# Prepare data
X = df[["SquareFootage", "Bedrooms", "Bathrooms"]]
y = df["PriceCategory"]

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model with optimized k
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.4f}")

# Save both the scaler and model
with open("knn_model.pkl", "wb") as f:
    pickle.dump((scaler, model), f)


print("KNN model saved as knn_model.pkl")
