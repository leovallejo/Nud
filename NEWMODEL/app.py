import os
import threading
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from celery import Celery
from celery.schedules import crontab
from flask import Flask, Response, json
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_healthz import healthz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, LSTM, MaxPooling1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import logging

# Initialize Flask app
app = Flask(__name__)

# Configuration for Celery and Model Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure the models directory exists
MODEL_FILE = os.path.join(MODEL_DIR, 'best_model.h5')

app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0' 
app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# Model Parameters
SEQUENCE_LENGTH = 60
FORECAST_HORIZON = 20

# Global variables for data buffering
last_trained_timestamp = None
new_data_buffer = []

# Thread lock for model operations
model_lock = threading.Lock()

# Global variable to store the loaded model
model = None

# Initialize Rate Limiter
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

# Initialize Health Check
app.register_blueprint(healthz, url_prefix="/health")

# --- All Functions ---

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
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df[features].astype(np.float64))
    
    logger.debug(f"Scaled data statistics: min={np.min(scaled_data)}, max={np.max(scaled_data)}, mean={np.mean(scaled_data)}")
    
    return scaled_data, scaler

def create_sequences(data, sequence_length, forecast_horizon=20):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        seq = data[i:i+sequence_length]
        target = data[i+sequence_length:i+sequence_length+forecast_horizon, 3]  # 'close' is at index 3
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
        if np.isnan(logs.get('loss')):
            self.model.stop_training = True
            print("NaN loss encountered, terminating training")

def is_valid_prediction(prediction):
    return not (np.isnan(prediction) or np.isinf(prediction))

def fallback_prediction(df):
    return round(df['close'].tail(20).mean(), 2)

def sanity_check_prediction(prediction, current_price):
    max_change = 0.1  # 10% change
    lower_bound = current_price * (1 - max_change)
    upper_bound = current_price * (1 + max_change)
    return max(min(prediction, upper_bound), lower_bound)

def load_trained_model():
    global model
    with model_lock:
        if model is None:
            try:
                model = load_model(MODEL_FILE)
                logger.info(f"Loaded existing model from {MODEL_FILE}.")
            except Exception as e:
                logger.warning(f"Failed to load model from {MODEL_FILE}: {e}")
                logger.info("Creating a new model.")
                model = build_cnn_lstm_model((SEQUENCE_LENGTH, 8), output_size=FORECAST_HORIZON)
                model.save(MODEL_FILE)
                logger.info(f"New model created and saved to {MODEL_FILE}.")

def get_model_instance():
    global model
    if model is None:
        load_trained_model()
    return model

def get_updated_data(symbol="ETHUSDT"):
    global last_trained_timestamp, new_data_buffer
    
    # 1. Fetch data from Binance
    url = get_binance_url(symbol=symbol, limit=1000) 
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    # 2. Data Processing
    numeric_columns = ["open", "high", "low", "close", "volume"]
    df[numeric_columns] = df[numeric_columns].astype(float)
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
    df = df[["close_time", "open", "high", "low", "close", "volume"]]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df.set_index("date", inplace=True)

    # 3. Filter for new data (if last_trained_timestamp is available)
    if last_trained_timestamp:
        df = df[df.index > last_trained_timestamp]
    
    # 4. Combine new data from the buffer
    if new_data_buffer:
        buffered_df = pd.DataFrame(new_data_buffer)
        buffered_df.set_index("date", inplace=True)
        df = pd.concat([df, buffered_df], ignore_index=False)
    
    return df

# Celery task for model retraining
@celery.task
def retrain_model():
    global last_trained_timestamp, new_data_buffer, model
    with app.app_context():
        try:
            logger.info("Starting model retraining task.")
            # 1. Fetch new data (symbol can be parameterized if needed)
            df = get_updated_data(symbol='ETHUSDT') 

            if df.empty:
                logger.warning("No new data available for retraining.")
                return

            # 2. Preprocess data
            df = add_technical_indicators(df)
            scaled_data, scaler = prepare_data(df)

            # 3. Load existing model or create a new one
            model = get_model_instance()

            # 4. Retrain the model 
            X, y = create_sequences(scaled_data, SEQUENCE_LENGTH, FORECAST_HORIZON)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(MODEL_FILE, save_best_only=True),
                NanTerminateCallback()
            ]
            logger.info("Starting model training.")
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )

            # 5. Save the retrained model
            model.save(MODEL_FILE)
            logger.info(f"Model retrained and saved to {MODEL_FILE} successfully!")

            # Reload the model into the global variable
            load_trained_model()

            # Update last trained timestamp and clear buffer
            last_trained_timestamp = datetime.utcnow()
            new_data_buffer = []
            logger.info("Model retraining task completed successfully.")

        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")
            logger.error(traceback.format_exc())

# --- Routes and Functions ---

@app.route("/health", methods=["GET"])
def health():
    return healthz()

@app.route("/inference/<string:token>")
@limiter.limit("100 per hour")
def get_inference(token):
    try:
        # Input Validation
        symbol_map = {
            'ETH': 'ETHUSDT',
            'BTC': 'BTCUSDT',
            'BNB': 'BNBUSDT',
            'SOL': 'SOLUSDT',
            'ARB': 'ARBUSDT'
        }
        token = token.upper()
        if token not in symbol_map:
            logger.error(f"Unsupported token: {token}")
            return Response(json.dumps({"error": "Unsupported token"}), status=400, mimetype='application/json')

        symbol = symbol_map[token]

        # Fetch data
        df = get_updated_data(symbol=symbol)

        if df.empty:
            logger.warning(f"No new data available for symbol: {symbol}")
            return Response(json.dumps({"error": "No new data available"}), status=400, mimetype='application/json')

        # --- Prediction Logic ---

        # 1. Add Technical Indicators
        df = add_technical_indicators(df) 

        # 2. Prepare data for prediction
        scaled_data, scaler = prepare_data(df)  
        
        if len(scaled_data) < SEQUENCE_LENGTH:
            logger.warning("Not enough data to create a sequence for prediction.")
            return Response(json.dumps({"error": "Not enough data for prediction"}), status=400, mimetype='application/json')
        
        last_sequence = scaled_data[-SEQUENCE_LENGTH:]  
        last_sequence = last_sequence.reshape(1, SEQUENCE_LENGTH, 8) 

        # 3. Use the pre-loaded model for prediction
        with model_lock:
            if model is None:
                load_trained_model()
            predictions = model.predict(last_sequence)

        # 4. Inverse transform the predictions
        # Reshape predictions from (1, 20) to (20, 1)
        predictions_reshaped = predictions.reshape(-1, 1)  # Shape: (20, 1)
        placeholder = np.zeros((FORECAST_HORIZON, 7))  # Shape: (20, 7)
        predictions_extended = np.hstack((predictions_reshaped, placeholder))  # Shape: (20, 8)

        # Inverse transform
        predicted_prices_scaled = scaler.inverse_transform(predictions_extended)[:, 0]  # 'close' is the first column
        final_prediction = round(float(predicted_prices_scaled[-1]), 2) 

        # 5. Apply sanity checks
        current_price = df['close'].iloc[-1]
        final_prediction = sanity_check_prediction(final_prediction, current_price)

        # --- End of Prediction Logic ---

        logger.info(f"Prediction for {symbol}: {final_prediction}")
        return Response(str(final_prediction), status=200, mimetype='text/plain')

    except Exception as e:
        logger.error(f"An error occurred during inference: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(
            json.dumps({"error": "An internal server error occurred", "details": str(e)}), 
            status=500, 
            mimetype='application/json'
        )

# Schedule periodic retraining (e.g., every hour)
celery.conf.beat_schedule = {
    'retrain-model-hourly': {
        'task': 'app.retrain_model',
        'schedule': crontab(minute=0, hour='*'), 
    },
}

# Load the model at startup
load_trained_model()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False)
