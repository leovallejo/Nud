import pandas as pd
import numpy as np
import requests
from flask import Flask, Response, json
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import traceback
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import joblib
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable eager execution for better performance
tf.compat.v1.disable_eager_execution()

# Configure TensorFlow to use CPU only if no GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if not gpus:
    logger.info("No GPUs available. Using CPU.")
    tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

BINANCE_BASE_URL = "https://api.binance.com/api/v3"
MODEL_DIR = "models"
SCALER_DIR = "scalers"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

class AttentionLayer(Layer):
    # ... (keep the existing AttentionLayer implementation)

def get_binance_klines(symbol="ETHUSDT", interval="1h", limit=5000):
    # ... (keep the existing implementation)

def get_binance_ticker(symbol="ETHUSDT"):
    # ... (keep the existing implementation)

def handle_nan_values(df):
    # ... (keep the existing implementation)

def add_technical_indicators(df):
    # ... (keep the existing implementation)

def calculate_rsi(prices, window=14):
    # ... (keep the existing implementation)

def calculate_atr(df, period=14):
    # ... (keep the existing implementation)

def prepare_data(df):
    features = ['open', 'high', 'low', 'close', 'volume', 'MA7', 'MA14', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df[features].astype(np.float64))
    return scaled_data, scaler

def create_sequences(data, sequence_length):
    # ... (keep the existing implementation)

def custom_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    mae = K.mean(K.abs(y_true - y_pred))
    huber_loss = tf.keras.losses.Huber()(y_true, y_pred)
    return 0.4 * mse + 0.3 * mae + 0.3 * huber_loss

def build_advanced_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='elu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='elu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(GRU(64, return_sequences=True)),
        AttentionLayer(),
        Dense(64, activation='elu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(32, activation='elu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mae', 'mse'])
    return model

class NanTerminateCallback(tf.keras.callbacks.Callback):
    # ... (keep the existing implementation)

def is_valid_prediction(prediction):
    # ... (keep the existing implementation)

def fallback_prediction(df):
    # ... (keep the existing implementation)

def sanity_check_prediction(prediction, current_price, historical_volatility):
    # ... (keep the existing implementation)

def calculate_historical_volatility(df, window=30):
    # ... (keep the existing implementation)

def train_and_save_model(symbol, X, y, sequence_length):
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []
    mae_scores = []

    for train_index, val_index in tscv.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = build_advanced_model((sequence_length, X.shape[2]))

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint(f'{MODEL_DIR}/{symbol}_best_model.h5', save_best_only=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001),
            NanTerminateCallback()
        ]

        logger.info(f"Training model for {symbol}...")
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                            epochs=100, batch_size=32, callbacks=callbacks, verbose=0)
        
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        mse_scores.append(mse)
        mae_scores.append(mae)

    logger.info(f"Cross-validation MSE scores for {symbol}: {mse_scores}")
    logger.info(f"Cross-validation MAE scores for {symbol}: {mae_scores}")
    logger.info(f"Average MSE for {symbol}: {np.mean(mse_scores):.6f}")
    logger.info(f"Average MAE for {symbol}: {np.mean(mae_scores):.6f}")

    # Retrain on full dataset
    model = build_advanced_model((sequence_length, X.shape[2]))
    history = model.fit(X, y, epochs=100, batch_size=32, callbacks=callbacks, verbose=0)
    
    logger.info(f"Final model training completed for {symbol}. Final loss: {history.history['loss'][-1]:.6f}")
    
    # Save the model
    model.save(f'{MODEL_DIR}/{symbol}_model.h5')
    logger.info(f"Model saved for {symbol}")

def load_or_train_model(symbol, X, y, sequence_length, force_retrain=False):
    model_path = f'{MODEL_DIR}/{symbol}_model.h5'
    if os.path.exists(model_path) and not force_retrain:
        logger.info(f"Loading existing model for {symbol}")
        return load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer, 'custom_loss': custom_loss})
    else:
        logger.info(f"Training new model for {symbol}")
        train_and_save_model(symbol, X, y, sequence_length)
        return load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer, 'custom_loss': custom_loss})

def save_scaler(scaler, symbol):
    joblib.dump(scaler, f'{SCALER_DIR}/{symbol}_scaler.pkl')

def load_scaler(symbol):
    return joblib.load(f'{SCALER_DIR}/{symbol}_scaler.pkl')

def process_symbol(symbol):
    try:
        klines_data = get_binance_klines(symbol=symbol)
        current_price_data = get_binance_ticker(symbol=symbol)
        
        df = pd.DataFrame(klines_data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])

        numeric_columns = ["open", "high", "low", "close", "volume"]
        df[numeric_columns] = df[numeric_columns].astype(float)

        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
        df = df[["close_time", "open", "high", "low", "close", "volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df.set_index("date", inplace=True)

        df = add_technical_indicators(df)
        historical_volatility = calculate_historical_volatility(df).iloc[-1]

        current_price = float(current_price_data['price'])
        current_time = df.index[-1]
        logger.info(f"Current Price for {symbol}: {current_price} at {current_time}")
        logger.info(f"Historical Volatility for {symbol}: {historical_volatility:.4f}")

        scaled_data, scaler = prepare_data(df)
        save_scaler(scaler, symbol)

        sequence_length = 60
        X, y = create_sequences(scaled_data, sequence_length)

        model = load_or_train_model(symbol, X, y, sequence_length)

        forecast_steps = 20
        last_sequence = X[-1]
        predictions = []

        logger.info(f"Making predictions for {symbol} ({forecast_steps} steps)...")
        for i in range(forecast_steps):
            next_pred = model.predict(last_sequence.reshape(1, sequence_length, X.shape[2]))
            predictions.append(next_pred[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = next_pred

        predicted_prices = scaler.inverse_transform(np.column_stack((predictions, np.zeros((len(predictions), X.shape[2]-1)))))
        final_prediction = round(float(predicted_prices[-1][0]), 2)

        if not is_valid_prediction(final_prediction):
            logger.warning(f"Main prediction invalid for {symbol}, using fallback method")
            final_prediction = fallback_prediction(df)

        final_prediction = sanity_check_prediction(final_prediction, current_price, historical_volatility)

        logger.info(f"Final Prediction for {symbol}: {final_prediction}")

        return {
            "symbol": symbol,
            "current_price": current_price,
            "prediction": final_prediction,
            "historical_volatility": float(historical_volatility),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"An error occurred while processing {symbol}: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "symbol": symbol,
            "error": f"An error occurred while processing {symbol}",
            "details": str(e)
        }

@app.route("/inference")
def get_inference():
    try:
        symbols = ['ETHUSDT', 'BTCUSDT', 'BNBUSDT', 'SOLUSDT', 'ARBUSDT']
        
        with ThreadPoolExecutor(max_workers=len(symbols)) as executor:
            results = list(executor.map(process_symbol, symbols))
        
        return Response(json.dumps(results), status=200, mimetype='application/json')

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(json.dumps({"error": "An internal server error occurred", "details": str(e)}), 
                        status=500, 
                        mimetype='application/json')

if __name__ == "__main__":
    from gunicorn.app.base import BaseApplication

    class StandaloneApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            config = {key: value for key, value in self.options.items()
                      if key in self.cfg.settings and value is not None}
            for key, value in config.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    options = {
        'bind': '0.0.0.0:8000',
        'workers': 4,
        'worker_class': 'gthread',
        'threads': 4,
    }
    StandaloneApplication(app, options).run()
