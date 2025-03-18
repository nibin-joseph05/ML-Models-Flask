from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained models
mlr_model_path = os.path.join(os.path.dirname(__file__), 'mlr_model.pkl')
slr_model_path = os.path.join(os.path.dirname(__file__), 'slr_model.pkl')

mlr_model = joblib.load(mlr_model_path)
slr_model = joblib.load(slr_model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/mlr', methods=['GET', 'POST'])
def mlr():
    predicted_price = None
    error = None
    
    if request.method == 'POST':
        try:
            square_footage = float(request.form['square_footage'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])
            age_of_house = int(request.form['age_of_house'])

            # Convert input to DataFrame
            input_data = pd.DataFrame([[square_footage, bedrooms, bathrooms, age_of_house]], 
                                      columns=['Square_Footage', 'Bedrooms', 'Bathrooms', 'Age_of_House'])

            # Predict price using MLR model
            predicted_price = mlr_model.predict(input_data)[0]
            predicted_price = round(predicted_price, 2)
        
        except Exception as e:
            error = str(e)

    return render_template('mlr.html', predicted_price=predicted_price, error=error)

@app.route('/slr', methods=['GET', 'POST'])
def slr():
    predicted_price = None
    error = None
    
    if request.method == 'POST':
        try:
            square_footage = float(request.form['square_footage'])

            # Convert input to DataFrame
            input_data = pd.DataFrame([[square_footage]], columns=['Square_Footage'])

            # Predict price using SLR model
            predicted_price = slr_model.predict(input_data)[0]
            predicted_price = round(predicted_price, 2)
        
        except Exception as e:
            error = str(e)

    return render_template('slr.html', predicted_price=predicted_price, error=error)

if __name__ == '__main__':
    app.run(debug=True)
