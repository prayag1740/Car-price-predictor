from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

def square_transform(x):
    return np.square(x)

# Function for log transformation (note: avoid log(0) issues)
def log_transform(x):
    return np.log1p(x)

app = Flask(__name__)
car = pd.read_csv('quikr_car_cleaned.csv')
model = pickle.load(open('LinearRegressionQuickr.pkl', 'rb'))


@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())
    companies.insert(0, "Select Company")
    years.insert(0, "Select year")
    fuel_type.insert(0, "Select Fuel Type")
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_type=fuel_type)


@app.route('/predict', methods=['POST'])
def predict():
    
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    pred = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    
    pred = str(round(pred[0])) if pred else "Not Available"
    
    return pred

if __name__ =="__main__":
    app.run(debug=True)