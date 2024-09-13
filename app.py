import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import requests
from flask import Flask, Response, json
import logging
from datetime import datetime

app = Flask(__name__)

# Configure basic logging without JSON formatting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# Function to fetch historical data from Binance
def get_binance_url(symbol="ETHUSDT", interval="1m", limit=1000):
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    return result[1] <= 0.05

# Function to find optimal ARIMA parameters
def find_optimal_arima_params(timeseries):
    model = auto_arima(timeseries, start_p=1, start_q=1, max_p=5, max_q=5, m=1,
                       start_P=0, seasonal=False, d=1, D=1, trace=True,
                       error_action='ignore', suppress_warnings=True, stepwise=True)
    return model.order

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

        # Check stationarity
        is_stationary = check_stationarity(df['price'])
        logger.info(f"Is the time series stationary? {is_stationary}")

        # Find optimal ARIMA parameters
        optimal_order = find_optimal_arima_params(df['price'])
        logger.info(f"Optimal ARIMA order: {optimal_order}")

        # Fit ARIMA model with optimal parameters
        model = ARIMA(df['price'], order=optimal_order)
        model_fit = model.fit()

        # Make prediction
        if symbol in ['BTCUSDT', 'SOLUSDT']:
            forecast_steps = 10  # 10-minute prediction
        else:
            forecast_steps = 20  # 20-minute prediction

        forecast = model_fit.forecast(steps=forecast_steps)
        predicted_price = round(float(forecast.iloc[-1]), 2)

        # Calculate confidence intervals
        conf_int = model_fit.get_forecast(steps=forecast_steps).conf_int()
        lower_bound = round(float(conf_int.iloc[-1]['lower price']), 2)
        upper_bound = round(float(conf_int.iloc[-1]['upper price']), 2)

        # Log the prediction and confidence interval
        logger.info(f"Prediction: {predicted_price}")
        logger.info(f"95% Confidence Interval: ({lower_bound}, {upper_bound})")

        # Return prediction and confidence interval in JSON response
        response_data = {
            "predicted_price": predicted_price,
            "confidence_interval": {
                "lower": lower_bound,
                "upper": upper_bound
            }
        }
        return Response(json.dumps(response_data), status=200, mimetype='application/json')
    else:
        return Response(json.dumps({"error": "Failed to retrieve data from Binance API", "details": response.text}), 
                        status=response.status_code, 
                        mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
