import pandas as pd
import numpy as np
from flask import Flask, Response, json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
import traceback
import tensorflow as tf

from api_client import get_binance_klines, get_binance_ticker
from model import build_advanced_model, prepare_data, create_sequences

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class NanTerminateCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if np.isnan(logs.get('loss')):
            self.model.stop_training = True
            print("NaN loss encountered, terminating training")

def is_valid_prediction(prediction):
    return not (np.isnan(prediction) or np.isinf(prediction))

def fallback_prediction(df):
    return round(df['close'].tail(10).mean(), 2)

def sanity_check_prediction(prediction, current_price):
    max_change = 0.1
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
            klines_data = get_binance_klines(symbol=symbol, interval="1h", limit=5000)
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
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001),
            NanTerminateCallback()
        ]

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                            epochs=100, batch_size=32, callbacks=callbacks, verbose=0)

        forecast_steps = 10 if symbol in ['BTCUSDT', 'SOLUSDT'] else 20

        last_sequence = X[-1]
        predictions = []

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
