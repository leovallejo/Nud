from flask import Flask, Response
import requests
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta

# create our Flask app
app = Flask(__name__)

# CoinMarketCap API settings
CMC_API_KEY = 'YOUR_COINMARKETCAP_API_KEY'  # Replace with your actual API key
CMC_BASE_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'

def get_coinmarketcap_data(token):
    params = {
        'symbol': token,
        'convert': 'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': CMC_API_KEY,
    }
    response = requests.get(CMC_BASE_URL, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data['data'][token]['quote']['USD']['price']
    else:
        raise Exception(f"Failed to retrieve data: {response.text}")

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

@app.route("/inference/<string:token>/<int:minutes>")
def get_inference(token, minutes):
    """Generate inference for given token and time frame."""
    if minutes not in [10, 20]:
        return Response(json.dumps({"error": "Only 10 or 20 minutes predictions are supported"}), 
                        status=400, mimetype='application/json')

    try:
        # Collect data every minute for the last hour
        data = []
        for _ in range(60):
            price = get_coinmarketcap_data(token)
            data.append(price)
            time.sleep(60)  # Wait for 1 minute

        df = pd.DataFrame(data, columns=['price'])
        
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')

    # Prepare data for models
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['price']].values)

    seq_length = 30  # Use last 30 minutes to predict next 10 or 20 minutes
    X, y = create_sequences(data_scaled, seq_length)

    # Prepare data for non-sequential models
    X_2D = X.reshape((X.shape[0], -1))

    # LSTM model
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    # Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_2D, y)

    # XGBoost model
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_2D, y)

    # Support Vector Regression model
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_2D, y)

    # Prepare data for prediction
    last_sequence = data_scaled[-seq_length:].reshape(1, seq_length, 1)
    last_sequence_2D = last_sequence.reshape(1, -1)

    # Make predictions
    predictions = []
    for _ in range(minutes):
        lstm_pred = lstm_model.predict(last_sequence)
        rf_pred = rf_model.predict(last_sequence_2D)
        xgb_pred = xgb_model.predict(last_sequence_2D)
        svr_pred = svr_model.predict(last_sequence_2D)

        # Ensemble predictions (simple average)
        ensemble_pred = (lstm_pred + rf_pred + xgb_pred + svr_pred) / 4

        predictions.append(ensemble_pred[0][0])

        # Update sequences for next prediction
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = ensemble_pred
        last_sequence_2D = last_sequence.reshape(1, -1)

    # Inverse transform predictions
    forecasted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    result = {
        "token": token,
        "prediction_minutes": minutes,
        "predicted_prices": forecasted_values.flatten().tolist()
    }

    return Response(json.dumps(result), status=200, mimetype='application/json')

# run our Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
