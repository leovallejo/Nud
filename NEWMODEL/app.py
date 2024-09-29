from flask import Flask, Response
import requests
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from chronos import ChronosPipeline
import torch

app = Flask(__name__)

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

# Chronos model setup
model_name = "amazon/chronos-t5-tiny"

def get_coingecko_url(token):
    base_url = "https://api.coingecko.com/api/v3/coins/"
    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum'
    }
    
    token = token.upper()
    if token in token_map:
        url = f"{base_url}{token_map[token]}/market_chart?vs_currency=usd&days=30&interval=daily"
        return url
    else:
        raise ValueError("Unsupported token")

@app.route("/inference/<string:token>")
def get_inference(token):
    try:
        # Chronos pipeline setup
        chronos_pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    except Exception as e:
        return Response(json.dumps({"pipeline error": str(e)}), status=500, mimetype='application/json')

    try:
        url = get_coingecko_url(token)
    except ValueError as e:
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')

    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "<Your Coingecko API key>"  # replace with your API key
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["prices"])
        df.columns = ["date", "price"]
        df["date"] = pd.to_datetime(df["date"], unit='ms')
        df = df[:-1]  # removing today's price
    else:
        return Response(json.dumps({"Failed to retrieve data from the API": str(response.text)}), 
                        status=response.status_code, 
                        mimetype='application/json')

    # Prepare data for ensemble models
    X = df[['date']].values
    y = df['price'].values
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Train ensemble models
    ensemble_models = create_ensemble_models()
    for model in ensemble_models.values():
        model.fit(X_scaled, y_scaled)

    # Make ensemble prediction
    last_date = df['date'].iloc[-1]
    next_date = last_date + pd.Timedelta(days=1)
    X_pred = scaler_X.transform([[next_date.value]])
    ensemble_pred_scaled = ensemble_predict(ensemble_models, X_pred)
    ensemble_pred = scaler_y.inverse_transform(ensemble_pred_scaled.reshape(-1, 1)).ravel()[0]

    # Chronos prediction
    context = torch.tensor(df["price"])
    prediction_length = 1
    try:
        chronos_forecast = chronos_pipeline.predict(context, prediction_length)
        chronos_pred = chronos_forecast[0].mean().item()
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

    # Combine predictions
    final_prediction = (ensemble_pred + chronos_pred) / 2

    result = {
        "ensemble_prediction": ensemble_pred,
        "chronos_prediction": chronos_pred,
        "final_prediction": final_prediction
    }

    return Response(json.dumps(result), status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
