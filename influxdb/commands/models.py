import numpy as np
import tensorflow as tf
import os
import datetime
import pickle
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

class LSTMModel:
    """
    Enhanced LSTM Model class for time series forecasting.

    This class implements a configurable LSTM-based neural network for time series forecasting
    with support for:
    - Multiple stacked LSTM layers
    - Bidirectional LSTM layers
    - Dropout regularization
    - Batch normalization
    - Customizable activation functions

    The model is designed to predict future values in time series data based on historical patterns.
    """

    def __init__(self, input_shape, output_steps, activation='relu'):
        """
        Initializes the LSTM model.

        Args:
            input_shape (tuple): Shape of the input (timesteps, features)
            output_steps (int): Number of future timesteps to predict
            activation (str): Activation function for dense layers
        """
        self.output_steps = output_steps
        self.model = self._build_model(
            input_shape, 
            output_steps,
            activation
        )

    def _build_model(self, input_shape, output_steps, activation):
        """
        Build a more complex LSTM model with multiple layers and regularization.

        Args:
            input_shape (tuple): Shape of the input (timesteps, features)
            output_steps (int): Number of future timesteps to predict
            activation (str): Activation function for dense layers

        Returns:
            model: Compiled Keras model
        """
        model = Sequential()
        model.add(Input(shape=input_shape))

        model.add(LSTM(128, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(LSTM(64, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(32, activation=activation))
        model.add(Dropout(0.1))
        model.add(Dense(output_steps))
        model.add(Reshape((output_steps, 1)))  # output: (batch, output_steps, 1)

        model.compile(optimizer='adam', loss=weighted_directional_mae(alpha=5.0), metrics=['mse', directional_accuracy])
        model.summary()
        return model

    def fit(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=128, patience=10):
        """
        Train the LSTM model with early stopping.

        Args:
            X_train (ndarray): Input sequence data
            y_train (ndarray): Target sequence data
            X_val (ndarray): Validation input sequence data
            y_val (ndarray): Validation target sequence data
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size
            patience (int): Number of epochs with no improvement after which training will be stopped.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        checkpoint_cb = ModelCheckpoint(
            filepath='models/best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            save_weights_only=False,
            verbose=1
        )

        log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

        print(f"Training started with EarlyStopping (patience={patience})...")

        history = self.model.fit(
            X_train, 
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, checkpoint_cb, tensorboard_cb],
            verbose=1
        )
        print("Training finished.")
        return history

    def predict(self, data):
        """
        Predict future values using the LSTM model.

        Args:
            data (array-like): Input data of shape (samples, timesteps, features)

        Returns:
            ndarray: Predicted values of shape (samples, output_steps)
        """
        # Get predictions from the model
        predictions = self.model.predict(data)

        acc, cov = confidence_filtered_accuracy(y_true, y_pred_proba, threshold=0.85)
        print(f"Confidence-Filtered Accuracy: {acc:.2f}, Coverage: {cov:.2%}")

        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate the LSTM model on test data.

        Args:
            X_test (ndarray): Test input sequence data
            y_test (ndarray): Test target sequence data

        Returns:
            float: Loss value
        """
        loss, mse, directional_accuracy_test = self.model.evaluate(X_test, y_test, verbose=1)
        print(f"Test Loss: {loss}, MSE: {mse}, Directional Accuracy: {directional_accuracy}")

        acc, cov = confidence_filtered_accuracy(y_true, y_pred_proba, threshold=0.5)
        print(f"Confidence-Filtered Accuracy: {acc:.2f}, Coverage: {cov:.2%}")
        return loss

    def save_model(self, scaler=None):
        """
        Save the trained model to disk.

        Args:
            filepath (str): Path where the model will be saved

        Returns:
            None

        Example:
            model.save_model('path/to/model', save_format='h5')
        """

        try:
            model_path = os.path.join("models", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

            os.makedirs(model_path, exist_ok=True)
            self.model.save(model_path + '.keras')
            print(f"Model saved successfully to {model_path}.keras")

            if scaler is not None:
                scaler_path = os.path.join(model_path, 'scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"Scaler saved successfully to {scaler_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise

    @classmethod
    def load_model(cls, filepath, save_format='.keras'):
        """
        Load a previously saved model from disk.

        Args:
            filepath (str): Path to the saved model
            save_format (str): Format the model was saved in, default is '.keras'

        Returns:
            LSTMModel: A new instance of LSTMModel with the loaded model and scaler

        """

        try:
            model = tf.keras.models.load_model(filepath, custom_objects={'directional_accuracy': directional_accuracy, 'weighted_directional_mae': weighted_directional_mae})
            print(f"Model loaded successfully from {filepath}")

            scaler_path = os.path.join(filepath, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"Scaler loaded successfully from {scaler_path}")
            else:
                scaler = None

            return cls(model.input_shape[1:], model.output_steps), scaler
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


def directional_accuracy(y_true, y_pred):
    """
    If the predicted direction matches the true direction, return 1, otherwise 0.

    Args:
        y_true (tensor): True values
        y_pred (tensor): Predicted values
    Returns:
        float: Directional accuracy
    """
    y_true_direction = tf.sign(y_true)
    y_pred_direction = tf.sign(y_pred)

    matches = tf.cast(tf.equal(y_true_direction, y_pred_direction), tf.float32)

    return tf.reduce_mean(matches)


def weighted_directional_mae(alpha=1.0):
    """
    Weighted Directional Mean Absolute Error (MAE) loss function.
    This loss function penalizes the model more when the predicted direction is wrong.

    Args:
        alpha (float): Weighting factor for the directional penalty. Default is 1.0.

    Returns:
        loss (function): A function that computes the weighted directional MAE.
    """
    mae = tf.keras.losses.MeanAbsoluteError()

    def loss(y_true, y_pred):
        # 1) sima MAE
        base = mae(y_true, y_pred)  # dim=(batch,)

        # 2) igaz és becsült irány
        #   y_true, y_pred shape=(batch, 1)
        diff_true = y_true - tf.roll(y_true, shift=1, axis=0)  # egyszerűsítés: előző batchhez képest
        diff_pred = y_pred - tf.roll(y_pred, shift=1, axis=0)
        sign_true = tf.sign(diff_true)
        sign_pred = tf.sign(diff_pred)

        # 3) irányegyezés vizsgálata
        wrong_dir = tf.cast(tf.not_equal(sign_true, sign_pred), tf.float32)

        # 4) súlyozás: ha wrong_dir=1 → weight=1+alpha, különben weight=1
        weight = 1.0 + alpha * wrong_dir

        return tf.reduce_mean(base * weight)

    return loss


def confidence_filtered_directional_accuracy(y_true, y_pred, threshold=0.5):
    """
    Calculate the directional accuracy of predictions with a confidence threshold.

    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        threshold (float): Confidence threshold for predictions

    Returns:
        tuple: Directional accuracy and coverage.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    confident_mask = np.abs(y_pred) >= threshold
    confident_preds = np.sign(y_pred[confident_mask])
    confident_truth = np.sign(y_true[confident_mask])

    if len(confident_preds) == 0:
        return None, 0.0

    directional_accuracy_confidence = (confident_preds == confident_truth).mean()
    coverage = confident_mask.mean()

    return directional_accuracy_confidence, coverage



