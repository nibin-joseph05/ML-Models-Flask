from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import numpy as np

app = Flask(__name__)

# Load the trained models
models = {
    "mlr": joblib.load(os.path.join(os.path.dirname(__file__), 'mlr_model.pkl')),
    "slr": joblib.load(os.path.join(os.path.dirname(__file__), 'slr_model.pkl')),
    "pr": joblib.load(os.path.join(os.path.dirname(__file__), 'pr_model.pkl')),
    "knn": joblib.load(os.path.join(os.path.dirname(__file__), 'knn_model.pkl')),
    "lr": joblib.load(os.path.join(os.path.dirname(__file__), 'lr_model.pkl'))  # Linear Regression Model
}

# Check if "lr" is a dictionary, extract the model
if isinstance(models["lr"], dict):
    models["lr"] = models["lr"]["model"]


# Unpack polynomial regression components correctly
poly_features, scaler, pr_model = models["pr"]
models["pr"] = (poly_features, scaler, pr_model)

# Unpack KNN model correctly
scaler, knn_model = models["knn"]
models["knn"] = (scaler, knn_model)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/mlr', methods=['GET', 'POST'])
def mlr():
    prediction = None
    error = None
    r2_score = 0.85  # Example R² score, replace with actual calculation

    if request.method == 'POST':
        try:
            square_footage = float(request.form['square_footage'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])
            age_of_house = int(request.form['age_of_house'])  # Add age of house

            input_data = pd.DataFrame([[square_footage, bedrooms, bathrooms, age_of_house]],
                                      columns=['Square_Footage', 'Bedrooms', 'Bathrooms', 'Age_of_House'])

            prediction = models["mlr"].predict(input_data)[0]
            prediction = round(prediction, 2)
        except Exception as e:
            error = str(e)

    return render_template('mlr.html', prediction=prediction, error=error, r2_score=r2_score)

@app.route('/slr', methods=['GET', 'POST'])
def slr():
    prediction = None
    error = None
    r2_score = 0.75  # Example R² score, replace with actual calculation

    if request.method == 'POST':
        try:
            square_footage = float(request.form['square_footage'])
            input_data = pd.DataFrame([[square_footage]], columns=['Square_Footage'])
            prediction = models["slr"].predict(input_data)[0]
            prediction = round(prediction, 2)
        except Exception as e:
            error = str(e)

    return render_template('slr.html', prediction=prediction, error=error, r2_score=r2_score)

@app.route('/pr', methods=['GET', 'POST'])
def pr():
    prediction = None
    error = None
    r2_score = 0.80  # Example R² score, replace with actual calculation

    if request.method == 'POST':
        try:
            square_footage = float(request.form['square_footage'])
            input_data = pd.DataFrame([[square_footage]], columns=['SquareFootage'])
            
            poly_features, scaler, pr_model = models["pr"]  # Unpack model components
            input_data_scaled = scaler.transform(input_data)  # Scale input
            input_data_poly = poly_features.transform(input_data_scaled)  # Polynomial transform
            prediction = pr_model.predict(input_data_poly)[0]
            prediction = np.expm1(prediction)  # Convert back from log-scale
            prediction = round(prediction, 2)
        except Exception as e:
            error = str(e)

    return render_template('pr.html', prediction=prediction, error=error, r2_score=r2_score)

@app.route('/knn', methods=['GET', 'POST'])
def knn():
    prediction = None
    error = None
    accuracy = 0.90  # Example accuracy score, replace with actual calculation

    if request.method == 'POST':
        try:
            square_footage = float(request.form['square_footage'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])

            input_data = pd.DataFrame([[square_footage, bedrooms, bathrooms]],
                                      columns=['SquareFootage', 'Bedrooms', 'Bathrooms'])

            scaler, knn_model = models["knn"]  # Unpack the model
            input_data_scaled = scaler.transform(input_data)
            prediction = knn_model.predict(input_data_scaled)[0]
            prediction = "Expensive" if prediction == 1 else "Affordable"
        except Exception as e:
            error = str(e)

    return render_template('knn.html', prediction=prediction, error=error, accuracy=accuracy)

@app.route('/lr', methods=['GET', 'POST'])
def lr():
    prediction = None
    error = None
    accuracy = 0.85  # Example accuracy score, replace with actual calculation

    if request.method == 'POST':
        try:
            square_footage = float(request.form['square_footage'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])

            input_data = pd.DataFrame([[square_footage, bedrooms, bathrooms]],
                                      columns=['SquareFootage', 'Bedrooms', 'Bathrooms'])
            prediction = models["lr"].predict(input_data)[0]
            prediction = "Expensive" if prediction == 1 else "Affordable"
        except Exception as e:
            error = str(e)

    return render_template('lr.html', prediction=prediction, error=error, accuracy=accuracy)


if __name__ == '__main__':
    app.run(debug=True)
