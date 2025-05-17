import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Correct absolute paths
BASE_DIR = r"N:\Jupyter_Lab\Car-Price-Predictor-Clone"
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "CarPricePredictorLR.pkl")
CSV_PATH = os.path.join(BASE_DIR, "Cleaned Car.csv")

# Load model and data
model = pickle.load(open(MODEL_PATH, 'rb'))
car = pd.read_csv(CSV_PATH)

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    # Build the company-model mapping
    company_model_map = {}
    for company in companies:
        models = car[car['company'] == company]['name'].unique().tolist()
        company_model_map[company] = models

    return render_template('index.html',
                           companies=companies,
                           car_models=car_models,
                           years=years,
                           fuel_types=fuel_types,
                           company_model_map=company_model_map)  # <-- Added this

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    driven = int(request.form.get('kilo_driven'))

    input_df = pd.DataFrame([[car_model, company, year, driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    prediction = model.predict(input_df)[0]
    return str(round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
