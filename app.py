import pandas as pd
import numpy as np
import requests
from flask import Flask, Response, json
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import traceback

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

def get_binance_url(symbol="ETHUSDT", interval="1h", limit=5000):
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

def add_technical_indicators(df):
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA14'] = df['close'].rolling(window=14).mean()
    df['RSI'] = calculate_rsi(df['close'], window=14)
    return df

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_data(df):
    features = ['open', 'high', 'low', 'close', 'volume', 'MA7', 'MA14', 'RSI']
    for feature in features:
        if df[feature].dtype != 'float64':
            df[feature] = df[feature].astype(float)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    return scaled_data, scaler

def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        target = data[i+sequence_length, 3]  # Assuming 'close' is at index 3
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def build_cnn_lstm_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

@app.route("/inference/<string:token>")
def get_inference(token):
    try:
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
            logger.error(f"Unsupported token: {token}")
            return Response(json.dumps({"error": "Unsupported token"}), status=400, mimetype='application/json')

        url = get_binance_url(symbol=symbol)
        logger.debug(f"Fetching data from URL: {url}")
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])

            # Convert numeric columns to float
            numeric_columns = ["open", "high", "low", "close", "volume"]
            df[numeric_columns] = df[numeric_columns].astype(float)

            df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
            df = df[["close_time", "open", "high", "low", "close", "volume"]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df.set_index("date", inplace=True)

            logger.debug(f"Data types after conversion: {df.dtypes}")
            if not all(df[col].dtype == 'float64' for col in numeric_columns):
                raise ValueError("Failed to convert all numeric columns to float")

            logger.debug(f"Sample of data after initial load:\n{df.head()}")

            try:
                df = add_technical_indicators(df)
            except Exception as e:
                logger.error(f"Error adding technical indicators: {str(e)}")
                logger.error(traceback.format_exc())
                return Response(json.dumps({"error": "Error processing data", "details": str(e)}), 
                                status=500, 
                                mimetype='application/json')

            current_price = df.iloc[-1]["close"]
            current_time = df.index[-1]
            logger.info(f"Current Price: {current_price} at {current_time}")

            scaled_data, scaler = prepare_data(df)

            sequence_length = 60
            X, y = create_sequences(scaled_data, sequence_length)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = build_cnn_lstm_model((sequence_length, X.shape[2]))

            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint('best_model.h5', save_best_only=True)
            ]

            logger.debug("Training model...")
            model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                      epochs=100, batch_size=32, callbacks=callbacks, verbose=0)

            forecast_steps = 10 if symbol in ['BTCUSDT', 'SOLUSDT'] else 20

            last_sequence = X[-1]
            predictions = []

            logger.debug(f"Making predictions for {forecast_steps} steps...")
            for _ in range(forecast_steps):
                next_pred = model.predict(last_sequence.reshape(1, sequence_length, X.shape[2]))
                predictions.append(next_pred[0, 0])
                last_sequence = np.roll(last_sequence, -1, axis=0)
                last_sequence[-1] = next_pred

            predicted_prices = scaler.inverse_transform(np.column_stack((predictions, np.zeros((len(predictions), X.shape[2]-1)))))
            final_prediction = round(float(predicted_prices[-1][0]), 2)

            logger.info(f"Prediction: {final_prediction}")

            return Response(json.dumps(final_prediction), status=200, mimetype='application/json')
        else:
            logger.error(f"Failed to retrieve data from Binance API. Status code: {response.status_code}")
            return Response(json.dumps({"error": "Failed to retrieve data from Binance API", "details": response.text}), 
                            status=response.status_code, 
                            mimetype='application/json')
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(json.dumps({"error": "An internal server error occurred", "details": str(e)}), 
                        status=500, 
                        mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
