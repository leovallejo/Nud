import pandas as pd
import numpy as np
import requests
from flask import Flask, Response, json
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Configure basic logging without JSON formatting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# Function to fetch historical data from Binance
def get_binance_url(symbol="ETHUSDT", interval="1m", limit=1000):
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

# Function to create sequences for CNN input
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        sequences.append(seq)
    return np.array(sequences)

# Function to build CNN model
def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

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

        # Normalize the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['price']])

        # Create sequences for CNN input
        sequence_length = 60  # Use 60 minutes of historical data to predict the next minute
        X = create_sequences(scaled_data, sequence_length)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build and train the CNN model
        model = build_cnn_model((sequence_length, 1))
        model.fit(X[:-1], scaled_data[sequence_length:], epochs=50, batch_size=32, verbose=0)

        # Make prediction
        if symbol in ['BTCUSDT', 'SOLUSDT']:
            forecast_steps = 10  # 10-minute prediction
        else:
            forecast_steps = 20  # 20-minute prediction

        last_sequence = X[-1]
        predictions = []

        for _ in range(forecast_steps):
            next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
            predictions.append(next_pred[0, 0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = next_pred

        # Inverse transform the predictions
        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        final_prediction = round(float(predicted_prices[-1][0]), 2)

        # Log the prediction
        logger.info(f"Prediction: {final_prediction}")

        # Return only the predicted price in JSON response
        return Response(json.dumps(final_prediction), status=200, mimetype='application/json')
    else:
        return Response(json.dumps({"error": "Failed to retrieve data from Binance API", "details": response.text}), 
                        status=response.status_code, 
                        mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
