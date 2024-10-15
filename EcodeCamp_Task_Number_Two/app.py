from flask import Flask, request, jsonify
import yfinance as yf
import numpy as np
from waitress import serve
from keras.models import load_model
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)  

lstm_model = load_model('lstm_model.h5')

with open('arima_model.pkl', 'rb') as file:
    arima_model = pickle.load(file)

scaler = MinMaxScaler(feature_range=(0, 1))

@app.route('/predict', methods=['GET'])
def predict_stock():
    stock_symbol = request.args.get('symbol', 'AAPL')
    model_type = request.args.get('model', 'lstm')
    
    try:
        stock_data = yf.download(stock_symbol, period='1mo', interval='1d')
        
        if stock_data.empty:
            return jsonify({'error': 'No data found for the symbol'}), 404

        close_prices = stock_data['Close'].values.reshape(-1, 1)

        # Normalize data for LSTM
        scaled_data = scaler.fit_transform(close_prices)
        
        # Choose which model to use
        if model_type == 'lstm':

            # LSTM model prediction logic
            X_input = scaled_data[-60:]  

            # Reshape for LSTM input
            X_input = np.reshape(X_input, (1, X_input.shape[0], 1))  
            prediction_scaled = lstm_model.predict(X_input)

            prediction = scaler.inverse_transform(prediction_scaled)[0][0]  
            
            prediction = float(prediction)
            
        elif model_type == 'arima':
            # ARIMA model prediction logic
            arima_prediction = arima_model.forecast(steps=1)  
            prediction = arima_prediction[0]
            
            prediction = float(prediction)
        else:
            return jsonify({'error': 'Unknown model type'}), 400
        
        return jsonify({'symbol': stock_symbol, 'predicted_price': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8000)
