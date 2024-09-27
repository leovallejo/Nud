import pandas as pd
import numpy as np
import requests
from flask import Flask, Response, json
import logging
from datetime import datetime, timedelta
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

# ... [rest of the code remains the same until the get_inference function] ...

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

        # ... [data preparation and model training remains the same] ...

        forecast_steps = 20  # We'll predict 20 steps for both 10 and 20 minutes

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
            
            # Calculate 10-minute and 20-minute predictions
            prediction_10min = round(float(predicted_prices[9][0]), 2)  # 10th step (index 9) for 10 minutes
            prediction_20min = round(float(predicted_prices[-1][0]), 2)  # Last step for 20 minutes

            if not is_valid_prediction(prediction_10min) or not is_valid_prediction(prediction_20min):
                logger.warning("Main prediction invalid, using fallback method")
                prediction_10min = fallback_prediction(df)
                prediction_20min = fallback_prediction(df)
        except Exception as e:
            logger.error(f"Error in main prediction: {str(e)}")
            logger.warning("Using fallback prediction method")
            prediction_10min = fallback_prediction(df)
            prediction_20min = fallback_prediction(df)

        prediction_10min = sanity_check_prediction(prediction_10min, current_price)
        prediction_20min = sanity_check_prediction(prediction_20min, current_price)

        logger.info(f"10-minute Prediction: {prediction_10min}")
        logger.info(f"20-minute Prediction: {prediction_20min}")

        result = {
            "10min_prediction": prediction_10min,
            "20min_prediction": prediction_20min,
            "current_price": current_price,
            "timestamp": datetime.now().isoformat()
        }

        return Response(json.dumps(result), status=200, mimetype='application/json')

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(json.dumps({"error": "An internal server error occurred", "details": str(e)}), 
                        status=500, 
                        mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
