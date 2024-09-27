import pandas as pd
import numpy as np
import requests
from flask import Flask, Response, json
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import traceback
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Concatenate, Attention, BatchNormalization, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# ... (keep the existing utility functions like get_binance_url, handle_nan_values, etc.)

def add_advanced_features(df):
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA14'] = df['close'].rolling(window=14).mean()
    df['RSI'] = calculate_rsi(df['close'], window=14)
    df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['Bollinger_upper'] = df['close'].rolling(window=20).mean() + 2 * df['close'].rolling(window=20).std()
    df['Bollinger_lower'] = df['close'].rolling(window=20).mean() - 2 * df['close'].rolling(window=20).std()
    df['Volume_MA'] = df['volume'].rolling(window=10).mean()
    return handle_nan_values(df)

def custom_activation(x):
    return tf.nn.selu(x) + tf.math.sigmoid(x)

def build_advanced_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # CNN branch
    conv1 = Conv1D(64, kernel_size=3, activation=custom_activation, kernel_regularizer=l2(0.01))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(128, kernel_size=3, activation=custom_activation, kernel_regularizer=l2(0.01))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling1D(pool_size=2)(conv2)
    
    # LSTM branch
    lstm1 = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)))(inputs)
    lstm1 = BatchNormalization()(lstm1)
    lstm2 = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    
    # GRU branch
    gru1 = GRU(64, return_sequences=True, kernel_regularizer=l2(0.01))(inputs)
    gru1 = BatchNormalization()(gru1)
    gru2 = GRU(64, return_sequences=True, kernel_regularizer=l2(0.01))(gru1)
    gru2 = BatchNormalization()(gru2)
    
    # Attention mechanism
    attention_lstm = Attention()([lstm2, conv2])
    attention_gru = Attention()([gru2, conv2])
    
    # Merge branches
    merged = Concatenate()([Flatten()(conv2), Flatten()(attention_lstm), Flatten()(attention_gru)])
    
    # Dense layers
    dense1 = Dense(256, activation=custom_activation, kernel_regularizer=l2(0.01))(merged)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.3)(dense1)
    dense2 = Dense(128, activation=custom_activation, kernel_regularizer=l2(0.01))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.2)(dense2)
    
    # Output layer
    output = Dense(1)(dense2)
    
    model = Model(inputs=inputs, outputs=output)
    return model

def build_ensemble_model(input_shape, num_models=5):
    models = [build_advanced_model(input_shape) for _ in range(num_models)]
    
    inputs = Input(shape=input_shape)
    outputs = [model(inputs) for model in models]
    
    ensemble_output = Concatenate()(outputs)
    final_output = Dense(1)(ensemble_output)
    
    ensemble_model = Model(inputs=inputs, outputs=final_output)
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    ensemble_model.compile(optimizer=optimizer, loss=custom_loss)
    
    return ensemble_model

def custom_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    mae = K.mean(K.abs(y_true - y_pred))
    huber_loss = tf.keras.losses.Huber()(y_true, y_pred)
    return 0.4 * mse + 0.3 * mae + 0.3 * huber_loss

class AdaptiveLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, patience=5, factor=0.5, min_lr=1e-6):
        super(AdaptiveLearningRateScheduler, self).__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.wait = 0
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                current_lr = K.get_value(self.model.optimizer.lr)
                new_lr = max(current_lr * self.factor, self.min_lr)
                K.set_value(self.model.optimizer.lr, new_lr)
                print(f'\nEpoch {epoch}: reducing learning rate to {new_lr}.')
                self.wait = 0

@app.route("/inference/<string:token>")
def get_inference(token):
    try:
        # ... (keep the existing code for data retrieval and preprocessing)

        df = add_advanced_features(df)

        scaled_data, scaler = prepare_data(df)

        sequence_length = 60
        X, y = create_sequences(scaled_data, sequence_length)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = build_ensemble_model((sequence_length, X.shape[2]))

        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ModelCheckpoint('best_ensemble_model.h5', save_best_only=True),
            AdaptiveLearningRateScheduler(patience=10, factor=0.7, min_lr=1e-6),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        # Continuous learning
        model.fit(X_test, y_test, epochs=10, batch_size=32, verbose=0)

        forecast_steps = 10 if token in ['BTC', 'SOL'] else 20

        last_sequence = X[-1]
        predictions = []

        try:
            for _ in range(forecast_steps):
                next_pred = model.predict(last_sequence.reshape(1, sequence_length, X.shape[2]))
                predictions.append(next_pred[0, 0])
                last_sequence = np.roll(last_sequence, -1, axis=0)
                last_sequence[-1] = next_pred

            predicted_prices = scaler.inverse_transform(np.column_stack((predictions, np.zeros((len(predictions), X.shape[2]-1)))))
            final_prediction = round(float(predicted_prices[-1][0]), 2)

            if not is_valid_prediction(final_prediction):
                final_prediction = fallback_prediction(df)
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            final_prediction = fallback_prediction(df)

        final_prediction = sanity_check_prediction(final_prediction, df.iloc[-1]["close"])

        logger.info(f"Final Prediction: {final_prediction}")

        return Response(json.dumps(final_prediction), status=200, mimetype='application/json')

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(json.dumps({"error": "An internal server error occurred", "details": str(e)}), 
                        status=500, mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
