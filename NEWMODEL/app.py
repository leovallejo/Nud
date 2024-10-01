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

def make_predictions(model, last_sequence, scaler, steps, X_shape):
    predictions = []
    for i in range(steps):
        next_pred = model.predict(last_sequence.reshape(1, last_sequence.shape[0], X_shape[2]))
        predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1] = next_pred
    predicted_prices = scaler.inverse_transform(np.column_stack((predictions, np.zeros((len(predictions), X_shape[2]-1)))))
    return predicted_prices

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
            # ... (keep the data processing part)

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
            
            last_sequence = X[-1]
            
            # Make predictions for 10 minutes, 20 minutes, and 24 hours
            predictions_10min = make_predictions(model, last_sequence, scaler, 10, X.shape)
            predictions_20min = make_predictions(model, last_sequence, scaler, 20, X.shape)
            predictions_24h = make_predictions(model, last_sequence, scaler, 24*60, X.shape)  # 24 hours * 60 minutes

            # Determine which prediction to return based on the symbol
            if symbol in ['BTCUSDT', 'SOLUSDT']:
                predicted_price_10min = round(float(predictions_10min[-1][0]), 2)
                predicted_price_20min = round(float(predictions_20min[-1][0]), 2)
                predicted_price_24h = round(float(predictions_24h[-1][0]), 2)
            else:
                predicted_price_10min = round(float(predictions_10min[-1][0]), 2)
                predicted_price_20min = round(float(predictions_20min[-1][0]), 2)
                predicted_price_24h = round(float(predictions_24h[-1][0]), 2)

            # Sanity check for all predictions
            current_price = df.iloc[-1]["close"]
            predicted_price_10min = sanity_check_prediction(predicted_price_10min, current_price)
            predicted_price_20min = sanity_check_prediction(predicted_price_20min, current_price)
            predicted_price_24h = sanity_check_prediction(predicted_price_24h, current_price)

            logger.info(f"10-minute Prediction: {predicted_price_10min}")
            logger.info(f"20-minute Prediction: {predicted_price_20min}")
            logger.info(f"24-hour Prediction: {predicted_price_24h}")

            result = {
                "symbol": symbol,
                "current_price": current_price,
                "prediction_10min": predicted_price_10min,
                "prediction_20min": predicted_price_20min,
                "prediction_24h": predicted_price_24h
            }

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False)
