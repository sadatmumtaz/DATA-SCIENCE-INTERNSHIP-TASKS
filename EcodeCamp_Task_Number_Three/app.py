from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from waitress import serve
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load('titanic_model.pkl')

# Feature names expected by the model
feature_names = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'family_size', 'sex_male', 'embarked_S']

@app.route('/')
def home():
    return "Welcome to the Titanic Survival Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        # Create DataFrame to match feature names
        input_data = pd.DataFrame([data], columns=feature_names)
        prediction = model.predict(input_data)
        return jsonify({'survived': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
