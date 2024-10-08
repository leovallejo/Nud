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
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import traceback
import tensorflow as tf
from tensorflow.keras import backend as K
import os
import time

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# Global variables for model and data
model = None
last_train_time = None
data_df = pd.DataFrame()
scaler = None

# Model parameters
sequence_length = 60
forecast_horizon = 20
retrain_interval = 60 * 60  # Retrain every hour (in seconds)

def get_binance_url(symbol="ETHUSDT", interval="1m", limit=5000):
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

def handle_nan_values(df):
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    return df

def add_technical_indicators(df):
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA14'] = df['close'].rolling(window=14).mean()
    df['RSI'] = calculate_rsi(df['close'], window=14)
    df = handle_nan_values(df)
    return df

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / np.maximum(loss, 1e-10)
    return 100 - (100 / (1 + rs))

def prepare_data(df):
    features = ['open', 'high', 'low', 'close', 'volume', 'MA7', 'MA14', 'RSI']
    df = handle_nan_values(df)

    for feature in features:
        df[feature] = df[feature].astype(np.float64)

    if df[features].isna().any().any():
        logger.error("NaN values still present after handling")
        raise ValueError("Unable to handle all NaN values")

    global scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df[features].astype(np.float64))

    logger.debug(f"Scaled data statistics: min={np.min(scaled_data)}, max={np.max(scaled_data)}, mean={np.mean(scaled_data)}")

    return scaled_data, scaler

def create_sequences(data, sequence_length, forecast_horizon=20):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        seq = data[i:i+sequence_length]
        target = data[i+sequence_length:i+sequence_length+forecast_horizon, 3]  # Assuming 'close' is at index 3
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def custom_loss(y_true, y_pred):
    mask = K.not_equal(y_true, 0)
    loss = K.mean(K.square(y_true[mask] - y_pred[mask]))
    return loss

def build_cnn_lstm_model(input_shape, output_size=20):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(output_size)
    ])
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=custom_loss)
    return model

class NanTerminateCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and np.isnan(logs.get('loss')):
            self.model.stop_training = True
            print("NaN loss encountered, terminating training")

def is_valid_prediction(prediction):
    return not (np.isnan(prediction) or np.isinf(prediction))

def fallback_prediction(df):
    return round(df['close'].tail(20).mean(), 2)

def sanity_check_prediction(prediction, current_price):
    max_change = 0.1
    lower_bound = current_price * (1 - max_change)
    upper_bound = current_price * (1 + max_change)
    return max(min(prediction, upper_bound), lower_bound)

def load_or_train_model(df):
    global model, last_train_time, scaler

    model_path = 'best_model.h5'

    if os.path.exists(model_path) and (last_train_time is not None) and (time.time() - last_train_time < retrain_interval):
        logger.info(f"Loading existing model from {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects={'custom_loss': custom_loss})
    else:
        logger.info("Training a new model")
        scaled_data, scaler = prepare_data(df)
        X, y = create_sequences(scaled_data, sequence_length, forecast_horizon)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = build_cnn_lstm_model((sequence_length, X.shape[2]), output_size=forecast_horizon)

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(model_path, save_best_only=True),
            NanTerminateCallback()
        ]

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=100, batch_size=32, callbacks=callbacks, verbose=0)

        logger.info("New model trained and saved")
        last_train_time = time.time()

@app.route("/inference/<string:token>")
def get_inference(token):
    global data_df, scaler
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

            # Update data_df with new data
            new_data = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            numeric_columns = ["open", "high", "low", "close", "volume"]
            new_data[numeric_columns] = new_data[numeric_columns].astype(float)
            new_data["close_time"] = pd.to_datetime(new_data["close_time"], unit='ms')
            new_data = new_data[["close_time", "open", "high", "low", "close", "volume"]]
            new_data.columns = ["date", "open", "high", "low", "close", "volume"]
            new_data.set_index("date", inplace=True)

            data_df = pd.concat([data_df, new_data])
            data_df = data_df[~data_df.index.duplicated(keep='last')] 
            data_df = data_df.tail(5000) 

            # Add technical indicators
            try:
                data_df = add_technical_indicators(data_df)
                logger.debug(f"Data statistics after adding indicators:\n{data_df.describe()}")
                logger.debug(f"NaN count after adding indicators:\n{data_df.isna().sum()}")
            except Exception as e:
                logger.error(f"Error adding technical indicators: {str(e)}")
                logger.error(traceback.format_exc())
                return Response(json.dumps({"error": "Error processing data", "details": str(e)}), 
                                status=500, 
                                mimetype='application/json')

            current_price = data_df.iloc[-1]["close"]
            current_time = data_df.index[-1]
            logger.info(f"Current Price: {current_price} at {current_time}")

            # Load or train the model
            load_or_train_model(data_df.copy())

            # Make predictions
            last_sequence = scaler.transform(data_df.tail(sequence_length)[['open', 'high', 'low', 'close', 'volume', 'MA7', 'MA14', 'RSI']].values.astype(np.float64))
            logger.debug(f"Last sequence statistics: min={np.min(last_sequence)}, max={np.max(last_sequence)}, mean={np.mean(last_sequence)}")

            try:
                predictions = model.predict(last_sequence.reshape(1, sequence_length, -1))
                predicted_prices = scaler.inverse_transform(np.column_stack((predictions.reshape(-1, 1), np.zeros((forecast_horizon, last_sequence.shape[1]-1)))))[:, 0]
                final_prediction = round(float(predicted_prices[-1]), 2)

                if not is_valid_prediction(final_prediction):
                    logger.warning("Main prediction invalid, using fallback method")
                    final_prediction = fallback_prediction(data_df)
            except Exception as e:
                logger.error(f"Error in main prediction: {str(e)}")
                logger.warning("Using fallback prediction method")
                final_prediction = fallback_prediction(data_df)

            final_prediction = sanity_check_prediction(final_prediction, current_price)
            logger.info(f"Final Prediction (20 minutes): {final_prediction}")

            return Response(str(final_prediction), status=200, mimetype='text/plain')

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
    app.run(host='0.0.0.0', port=8000, debug=False)
