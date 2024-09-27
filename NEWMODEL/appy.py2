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

# ... (keep all the previous functions and classes)

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
    app.run(host='0.0.0.0', port=8000)
