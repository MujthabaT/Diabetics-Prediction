from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained classifier and scaler
with open('./classifier_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)
with open('./scaler_model.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['blood_pressure']),
        float(request.form['skin_thickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['diabetes_pedigree_function']),
        float(request.form['age'])
    ]
    input_data = np.asarray(input_data).reshape(1, -1)
    standardized_data = scaler.transform(input_data)
    prediction = classifier.predict(standardized_data)
    result = {'prediction': 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)