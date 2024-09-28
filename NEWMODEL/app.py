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

def get_cmc_historical_data(symbol, hours=24):
    url = f"{CMC_BASE_URL}/cryptocurrency/quotes/historical"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours)
    
    parameters = {
        "symbol": symbol,
        "time_start": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "time_end": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "interval": "5m"  # 5-minute intervals
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

def predict_next_n_minutes(model, last_sequence, n_minutes, time_steps):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_minutes):
        input_data = np.reshape(current_sequence, (1, time_steps, 1))
        predicted_price = model.predict(input_data)
        predictions.append(predicted_price[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = predicted_price
    
    return np.array(predictions)

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

    time_steps = 12  # Use 1 hour of historical data (12 * 5 minutes)
    X, y = prepare_data(scaled_data, time_steps)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # LSTM Model
    lstm_model = create_lstm_model((X.shape[1], 1))
    lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    # GRU Model
    gru_model = create_gru_model((X.shape[1], 1))
    gru_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    # Prophet Model
    prophet_model = Prophet(interval_width=0.95)
    prophet_model.fit(df)

    # Make predictions
    last_sequence = scaled_data[-time_steps:]

    # LSTM and GRU predictions
    lstm_pred_10 = predict_next_n_minutes(lstm_model, last_sequence, 2, time_steps)  # 2 * 5 minutes = 10 minutes
    lstm_pred_20 = predict_next_n_minutes(lstm_model, last_sequence, 4, time_steps)  # 4 * 5 minutes = 20 minutes

    gru_pred_10 = predict_next_n_minutes(gru_model, last_sequence, 2, time_steps)
    gru_pred_20 = predict_next_n_minutes(gru_model, last_sequence, 4, time_steps)

    # Prophet predictions
    future = prophet_model.make_future_dataframe(periods=4, freq='5T')
    prophet_forecast = prophet_model.predict(future)
    prophet_pred_10 = prophet_forecast.iloc[-2]["yhat"]
    prophet_pred_20 = prophet_forecast.iloc[-1]["yhat"]

    # Inverse transform LSTM and GRU predictions
    lstm_pred_10 = scaler.inverse_transform(lstm_pred_10.reshape(-1, 1))[-1][0]
    lstm_pred_20 = scaler.inverse_transform(lstm_pred_20.reshape(-1, 1))[-1][0]
    gru_pred_10 = scaler.inverse_transform(gru_pred_10.reshape(-1, 1))[-1][0]
    gru_pred_20 = scaler.inverse_transform(gru_pred_20.reshape(-1, 1))[-1][0]

    # Ensemble predictions (simple average)
    ensemble_pred_10 = (lstm_pred_10 + gru_pred_10 + prophet_pred_10) / 3
    ensemble_pred_20 = (lstm_pred_20 + gru_pred_20 + prophet_pred_20) / 3

    result = {
        "10_minute_predictions": {
            "ensemble_prediction": ensemble_pred_10,
            "lstm_prediction": lstm_pred_10,
            "gru_prediction": gru_pred_10,
            "prophet_prediction": prophet_pred_10
        },
        "20_minute_predictions": {
            "ensemble_prediction": ensemble_pred_20,
            "lstm_prediction": lstm_pred_20,
            "gru_prediction": gru_pred_20,
            "prophet_prediction": prophet_pred_20
        }
    }

    return Response(json.dumps(result), status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
