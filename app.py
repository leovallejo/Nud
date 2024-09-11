import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import requests
from flask import Flask, Response, json
import logging
from datetime import datetime, timedelta
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
    
    # Add time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    # Add technical indicators
    df['SMA_5'] = df['price'].rolling(window=5).mean()
    df['SMA_20'] = df['price'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['price'])
    
    # Forward fill NaN values
    df.fillna(method='ffill', inplace=True)
    
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(df, order=(1,1,1), seasonal_order=(1,1,1,12)):
    model = SARIMAX(df['price'], 
                    exog=df[['hour', 'day_of_week', 'SMA_5', 'SMA_20', 'RSI']],
                    order=order, 
                    seasonal_order=seasonal_order)
    return model.fit(disp=False)

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

        # Prepare exogenous variables for forecasting
        future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps+1, freq='1min')[1:]
        future_exog = pd.DataFrame({
            'hour': future_dates.hour,
            'day_of_week': future_dates.dayofweek,
            'SMA_5': [df['SMA_5'].iloc[-1]] * forecast_steps,
            'SMA_20': [df['SMA_20'].iloc[-1]] * forecast_steps,
            'RSI': [df['RSI'].iloc[-1]] * forecast_steps
        })

        forecast = model_fit.forecast(steps=forecast_steps, exog=future_exog)
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
