import pandas as pd
import numpy as np
import requests
from flask import Flask, Response, json
import logging
from datetime import datetime, timedelta
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

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# ... (keep all the existing functions)

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

        url = get_binance_url(symbol=symbol, interval="1m", limit=5000)  # Changed to 1-minute intervals
        logger.debug(f"Fetching data from URL: {url}")
        response = requests.get(url)

        if response.status_code == 200:
            # ... (keep data processing steps)

            current_price = df.iloc[-1]["close"]
            current_time = df.index[-1]
            logger.info(f"Current Price: {current_price} at {current_time}")

            # ... (keep data preparation steps)

            model = build_cnn_lstm_model((sequence_length, X.shape[2]))
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint('best_model.h5', save_best_only=True),
                NanTerminateCallback()
            ]

            logger.debug("Training model...")
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                                epochs=100, batch_size=32, callbacks=callbacks, verbose=0)
            
            logger.debug(f"Model training history: {history.history}")

            # Make predictions for both 10 and 20 minutes
            predictions_10min = make_predictions(model, X, scaler, steps=10)
            predictions_20min = make_predictions(model, X, scaler, steps=20)

            # Sanity check and format predictions
            final_prediction_10min = sanity_check_prediction(predictions_10min[-1], current_price)
            final_prediction_20min = sanity_check_prediction(predictions_20min[-1], current_price)

            prediction_time_10min = current_time + timedelta(minutes=10)
            prediction_time_20min = current_time + timedelta(minutes=20)

            result = {
                "current_price": round(float(current_price), 2),
                "current_time": current_time.isoformat(),
                "prediction_10min": {
                    "price": round(float(final_prediction_10min), 2),
                    "time": prediction_time_10min.isoformat()
                },
                "prediction_20min": {
                    "price": round(float(final_prediction_20min), 2),
                    "time": prediction_time_20min.isoformat()
                }
            }

            logger.info(f"Predictions: {result}")
            return Response(json.dumps(result), status=200, mimetype='application/json')
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

def make_predictions(model, X, scaler, steps):
    last_sequence = X[-1]
    predictions = []
    for i in range(steps):
        next_pred = model.predict(last_sequence.reshape(1, X.shape[1], X.shape[2]))
        predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1] = next_pred
    predicted_prices = scaler.inverse_transform(np.column_stack((predictions, np.zeros((len(predictions), X.shape[2]-1)))))
    return predicted_prices[:, 0]

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
