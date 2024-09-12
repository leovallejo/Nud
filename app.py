import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import requests
from flask import Flask, jsonify
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
        return jsonify({"error": "Unsupported token"}), 400

    url = get_binance_url(symbol=symbol)
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
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

        # Log the current price and the timestamp
        current_price = df.iloc[-1]["price"]
        current_time = df.index[-1]
        logger.info(f"Current Price: {current_price} at {current_time}")

        # Split data into train and test sets
        train_size = int(len(df) * 0.8)
        train, test = df[:train_size], df[train_size:]

        # Fit ARIMA model
        model = ARIMA(train['price'], order=(5,1,0))
        model_fit = model.fit()

        # Make prediction
        if symbol in ['BTCUSDT', 'SOLUSDT']:
            forecast_steps = 10  # 10-minute prediction
        else:
            forecast_steps = 20  # 20-minute prediction

        forecast = model_fit.forecast(steps=forecast_steps)
        predicted_price = round(float(forecast.iloc[-1]), 2)

        # Calculate RMSE
        test_forecast = model_fit.forecast(steps=len(test))
        rmse = sqrt(mean_squared_error(test['price'], test_forecast))

        # Log the prediction and RMSE
        logger.info(f"Prediction: {predicted_price}, RMSE: {rmse}")

        # Return prediction and RMSE in JSON response
        return jsonify({
            "predicted_price": predicted_price,
            "model_rmse": round(rmse, 4)
        }), 200
    else:
        return jsonify({
            "error": "Failed to retrieve data from Binance API",
            "details": response.text
        }), response.status_code

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
