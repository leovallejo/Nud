import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import requests
from flask import Flask, Response, json
import logging
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Configure basic logging without JSON formatting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# Function to fetch historical data from Binance
def get_binance_url(symbol="ETHUSDT", interval="1m", limit=1000):
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

def preprocess_data(df):
    # Convert to datetime and set as index
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    
    # Remove any NaN or inf values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df

def train_model(df):
    model = ARIMA(df['price'], order=(5,1,0))
    return model.fit()

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

        df["date"] = pd.to_datetime(df["close_time"], unit='ms')
        df = df[["date", "close"]]
        df.columns = ["date", "price"]
        df["price"] = df["price"].astype(float)

        # Preprocess data
        df = preprocess_data(df)

        # Log the current price and the timestamp
        current_price = df.iloc[-1]["price"]
        current_time = df.index[-1]
        logger.info(f"Current Price: {current_price} at {current_time}")

        # Train model
        model_fit = train_model(df)

        # Make prediction
        if symbol in ['BTCUSDT', 'SOLUSDT']:
            forecast_steps = 10  # 10-minute prediction
        else:
            forecast_steps = 20  # 20-minute prediction

        forecast = model_fit.forecast(steps=forecast_steps)
        predicted_price = round(float(forecast.iloc[-1]), 2)

        # Log the prediction
        logger.info(f"Prediction: {predicted_price}")

        # Return only the predicted price in JSON response
        return Response(json.dumps(predicted_price), status=200, mimetype='application/json')
    else:
        return Response(json.dumps({"error": "Failed to retrieve data from Binance API", "details": response.text}), 
                        status=response.status_code, 
                        mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
