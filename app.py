import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import requests
from flask import Flask, Response, json
import logging
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

app = Flask(__name__)

# Configure basic logging without JSON formatting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# Function to fetch historical data from Binance
def get_binance_url(symbol="ETHUSDT", interval="1m", limit=1000):
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

# Function to preprocess data
def preprocess_data(data):
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
    df = df[["close_time", "close"]]
    df.columns = ["date", "price"]
    df["price"] = df["price"].astype(float)
    df.set_index("date", inplace=True)
    return df

# Function to fit ARIMA model
def fit_arima_model(data):
    # Use auto_arima to automatically select the best parameters
    model = auto_arima(data, start_p=1, start_q=1, max_p=5, max_q=5, m=1,
                       start_P=0, seasonal=False, d=1, D=1, trace=True,
                       error_action='ignore', suppress_warnings=True, stepwise=True)
    return model

# Function to evaluate model performance
def evaluate_model(model, test_data):
    predictions = model.predict(n_periods=len(test_data))
    mse = mean_squared_error(test_data, predictions)
    rmse = sqrt(mse)
    return rmse

@app.route("/inference/<string:token>")
def get_inference(token):
    symbol_map = {
        'ETH': 'ETHUSDT',
        'BTC': 'BTCUSDT',
        'BNB': 'BNBUSDT',
        'SOL': 'SOLUSDT',
        'ARB': 'ARBUSDT'
    }

    token = token.upper()
    if token in symbol_map:
        symbol = symbol_map[token]
    else:
        return Response(json.dumps({"error": "Unsupported token"}), status=400, mimetype='application/json')

    url = get_binance_url(symbol=symbol)
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        df = preprocess_data(data)

        # Log the current price and the timestamp
        current_price = df.iloc[-1]["price"]
        current_time = df.index[-1]
        logger.info(f"Current Price: {current_price} at {current_time}")

        # Split data into train and test sets
        train_size = int(len(df) * 0.8)
        train, test = df[:train_size], df[train_size:]

        # Fit ARIMA model
        model = fit_arima_model(train['price'])

        # Evaluate model performance
        rmse = evaluate_model(model, test['price'])
        logger.info(f"Model RMSE: {rmse}")

        # Make prediction
        if symbol in ['BTCUSDT', 'SOLUSDT']:
            forecast_steps = 10  # 10-minute prediction
        else:
            forecast_steps = 20  # 20-minute prediction

        forecast = model.predict(n_periods=forecast_steps)
        predicted_price = round(float(forecast.iloc[-1]), 2)

        # Log the prediction
        logger.info(f"Prediction: {predicted_price}")

        # Return prediction and model performance in JSON response
        response_data = {
            "predicted_price": predicted_price,
            "model_rmse": round(rmse, 4)
        }
        return Response(json.dumps(response_data), status=200, mimetype='application/json')
    else:
        return Response(json.dumps({"error": "Failed to retrieve data from Binance API", "details": response.text}), 
                        status=response.status_code, 
                        mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
