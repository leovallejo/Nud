import pandas as pd
import numpy as np
import requests
from flask import Flask, Response, json
import logging
from datetime import datetime
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
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def get_binance_klines(symbol="ETHUSDT", interval="1h", limit=5000):
    endpoint = f"{BINANCE_BASE_URL}/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error fetching klines data: {str(e)}")
        raise

def get_binance_ticker(symbol="ETHUSDT"):
    endpoint = f"{BINANCE_BASE_URL}/ticker/price"
    params = {"symbol": symbol}
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error fetching ticker data: {str(e)}")
        raise

def handle_nan_values(df):
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    return df

def add_technical_indicators(df):
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA14'] = df['close'].rolling(window=14).mean()
    df['RSI'] = calculate_rsi(df['close'], window=14)
    df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['ATR'] = calculate_atr(df)
    df = handle_nan_values(df)
    return df

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / np.maximum(loss, 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def prepare_data(df):
    features = ['open', 'high', 'low', 'close', 'volume', 'MA7', 'MA14', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df[features].astype(np.float64))
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
    def on_epoch_end(self, epoch, logs=None):
        if np.isnan(logs.get('loss')):
            self.model.stop_training = True
            logger.warning("NaN loss encountered, terminating training")

def is_valid_prediction(prediction):
    return not (np.isnan(prediction) or np.isinf(prediction))

def fallback_prediction(df):
    return round(df['close'].tail(10).mean(), 2)

def sanity_check_prediction(prediction, current_price, historical_volatility):
    max_change = min(3 * historical_volatility, 0.2)  # Cap at 20% or 3 times historical volatility
    lower_bound = current_price * (1 - max_change)
    upper_bound = current_price * (1 + max_change)
    return max(min(prediction, upper_bound), lower_bound)

def calculate_historical_volatility(df, window=30):
    returns = np.log(df['close'] / df['close'].shift(1))
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility

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
            "historical_volatility": float(historical_
