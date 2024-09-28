from flask import Flask, Response
import requests
import json
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta

app = Flask(__name__)

CMC_API_KEY = "YOUR_CMC_API_KEY_HERE"  # Replace with your CoinMarketCap API key
CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1"

def get_cmc_historical_data(symbol, days=30):
    url = f"{CMC_BASE_URL}/cryptocurrency/quotes/historical"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    parameters = {
        "symbol": symbol,
        "time_start": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "time_end": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "interval": "hourly"
    }
    
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": CMC_API_KEY
    }
    
    response = requests.get(url, headers=headers, params=parameters)
    
    if response.status_code == 200:
        data = response.json()
        quotes = data["data"][symbol]["quotes"]
        df = pd.DataFrame(quotes)
        df["ds"] = pd.to_datetime(df["timestamp"])
        df["y"] = df["quote.USD.price"]
        return df[["ds", "y"]]
    else:
        raise Exception(f"Failed to retrieve data: {response.text}")

def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def create_gru_model(input_shape):
    model = Sequential([
        GRU(units=50, return_sequences=True, input_shape=input_shape),
        GRU(units=50),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

@app.route("/inference/<string:token>")
def get_inference(token):
    """Generate inference for given token using multiple models."""
    try:
        df = get_cmc_historical_data(token.upper())
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')

    # Prepare data for LSTM and GRU models
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['y'].values.reshape(-1, 1))

    time_steps = 24  # Use 24 hours of historical data
    X, y = prepare_data(scaled_data, time_steps)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # LSTM Model
    lstm_model = create_lstm_model((X.shape[1], 1))
    lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    # GRU Model
    gru_model = create_gru_model((X.shape[1], 1))
    gru_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    # Prophet Model
    prophet_model = Prophet()
    prophet_model.fit(df)

    # Make predictions
    last_sequence = scaled_data[-time_steps:]
    lstm_input = np.reshape(last_sequence, (1, time_steps, 1))
    gru_input = np.reshape(last_sequence, (1, time_steps, 1))

    lstm_pred = lstm_model.predict(lstm_input)
    gru_pred = gru_model.predict(gru_input)

    future = prophet_model.make_future_dataframe(periods=1, freq='H')
    prophet_forecast = prophet_model.predict(future)
    prophet_pred = prophet_forecast.iloc[-1]["yhat"]

    # Inverse transform LSTM and GRU predictions
    lstm_pred = scaler.inverse_transform(lstm_pred)[0][0]
    gru_pred = scaler.inverse_transform(gru_pred)[0][0]

    # Ensemble prediction (simple average)
    ensemble_pred = (lstm_pred + gru_pred + prophet_pred) / 3

    result = {
        "ensemble_prediction": ensemble_pred,
        "lstm_prediction": lstm_pred,
        "gru_prediction": gru_pred,
        "prophet_prediction": prophet_pred
    }

    return Response(json.dumps(result), status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
