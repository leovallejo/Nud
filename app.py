import pandas as pd
import numpy as np
import requests
from flask import Flask, Response, json
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import traceback
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

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

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - \
               K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

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
    model.compile(optimizer=optimizer, loss=focal_loss(alpha=.25, gamma=2))
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

def sanity_check_prediction(prediction, current_price):
    max_change = 0.1  # 10% maximum change
    lower_bound = current_price * (1 - max_change)
    upper_bound = current_price * (1 + max_change)
    return max(min(prediction, upper_bound), lower_bound)

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

        try:
            klines_data = get_binance_klines(symbol=symbol)
            current_price_data = get_binance_ticker(symbol=symbol)
        except Exception as e:
            logger.error(f"Failed to retrieve data from Binance API: {str(e)}")
            return Response(json.dumps({"error": "Failed to retrieve data from Binance API", "details": str(e)}), 
                            status=500, 
                            mimetype='application/json')

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

        current_price = float(current_price_data['price'])
        current_time = df.index[-1]
        logger.info(f"Current Price: {current_price} at {current_time}")

        scaled_data, scaler = prepare_data(df)

        sequence_length = 60
        X, y = create_sequences(scaled_data, sequence_length)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = build_advanced_model((sequence_length, X.shape[2]))

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001),
            NanTerminateCallback()
        ]

        logger.info("Training model...")
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                            epochs=100, batch_size=32, callbacks=callbacks, verbose=0)
        
        logger.info(f"Model training completed. Final loss: {history.history['loss'][-1]}")

        forecast_steps = 10 if symbol in ['BTCUSDT', 'SOLUSDT'] else 20

        last_sequence = X[-1]
        predictions = []

        logger.info(f"Making predictions for {forecast_steps} steps...")
        try:
            for i in range(forecast_steps):
                next_pred = model.predict(last_sequence.reshape(1, sequence_length, X.shape[2]))
                predictions.append(next_pred[0, 0])
                last_sequence = np.roll(last_sequence, -1, axis=0)
                last_sequence[-1] = next_pred

            predicted_prices = scaler.inverse_transform(np.column_stack((predictions, np.zeros((len(predictions), X.shape[2]-1)))))
            final_prediction = round(float(predicted_prices[-1][0]), 2)

            if not is_valid_prediction(final_prediction):
                logger.warning("Main prediction invalid, using fallback method")
                final_prediction = fallback_prediction(df)
        except Exception as e:
            logger.error(f"Error in main prediction: {str(e)}")
            logger.warning("Using fallback prediction method")
            final_prediction = fallback_prediction(df)

        final_prediction = sanity_check_prediction(final_prediction, current_price)

        logger.info(f"Final Prediction: {final_prediction}")

        return Response(json.dumps(final_prediction), status=200, mimetype='application/json')

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(json.dumps({"error": "An internal server error occurred", "details": str(e)}), 
                        status=500, 
                        mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False)
