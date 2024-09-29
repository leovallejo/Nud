from flask import Flask, Response
import requests
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from datetime import datetime, timedelta
import time

app = Flask(__name__)

# CoinMarketCap API setup
CMC_API_KEY = "1f99c2fb-9adb-4347-82cd-a8097bead9df"  # Replace with your actual API key
CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"

# Ensemble model setup
def create_ensemble_models():
    return {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

def ensemble_predict(models, X):
    predictions = np.column_stack([model.predict(X) for model in models.values()])
    return np.mean(predictions, axis=1)

def get_historical_data(symbol, interval_minutes=1, limit=100):
    url = f"https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
    parameters = {
        'symbol': symbol,
        'interval': f'{interval_minutes}m',
        'count': limit
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': CMC_API_KEY
    }
    response = requests.get(url, headers=headers, params=parameters)
    if response.status_code == 200:
        data = response.json()
        prices = data['data'][symbol]['quotes']
        df = pd.DataFrame(prices)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        return df
    else:
        raise Exception(f"Failed to retrieve data: {response.text}")

@app.route("/inference/<string:token>")
def get_inference(token):
    try:
        # Fetch historical data
        df = get_historical_data(token.upper())
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

    # Prepare data for ensemble models
    X = df[['timestamp']].values
    y = df['price'].values
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Train ensemble models
    ensemble_models = create_ensemble_models()
    for model in ensemble_models.values():
        model.fit(X_scaled, y_scaled)

    # Make 10 and 20-minute predictions
    last_timestamp = df['timestamp'].iloc[-1]
    next_10min = last_timestamp + timedelta(minutes=10)
    next_20min = last_timestamp + timedelta(minutes=20)
    
    X_pred_10 = scaler_X.transform([[next_10min.value]])
    X_pred_20 = scaler_X.transform([[next_20min.value]])
    
    ensemble_pred_10_scaled = ensemble_predict(ensemble_models, X_pred_10)
    ensemble_pred_20_scaled = ensemble_predict(ensemble_models, X_pred_20)
    
    ensemble_pred_10 = scaler_y.inverse_transform(ensemble_pred_10_scaled.reshape(-1, 1)).ravel()[0]
    ensemble_pred_20 = scaler_y.inverse_transform(ensemble_pred_20_scaled.reshape(-1, 1)).ravel()[0]

    result = {
        "10_minute_prediction": ensemble_pred_10,
        "20_minute_prediction": ensemble_pred_20,
        "current_price": df['price'].iloc[-1],
        "last_updated": df['timestamp'].iloc[-1].isoformat()
    }

    return Response(json.dumps(result), status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
