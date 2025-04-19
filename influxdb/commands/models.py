import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
# Importáld az EarlyStopping callback-et
from tensorflow.keras.callbacks import EarlyStopping

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

    def __init__(self, input_shape, output_steps, scaler=None, lstm_units=64, 
                 num_layers=2, dropout_rate=0.2, use_bidirectional=True, 
                 use_batch_norm=True, activation='relu'):
        """
        Initializes the LSTM model.

        Args:
            input_shape (tuple): Shape of the input (timesteps, features)
            output_steps (int): Number of future timesteps to predict
            scaler (optional): Scaler used for inverse transforming predictions
            lstm_units (int): Number of LSTM units
            num_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate for regularization
            use_bidirectional (bool): Whether to use bidirectional LSTM
            use_batch_norm (bool): Whether to use batch normalization
            activation (str): Activation function for dense layers
        """
        self.scaler = scaler
        self.output_steps = output_steps
        self.model = self._build_model(
            input_shape, 
            output_steps, 
            lstm_units, 
            num_layers, 
            dropout_rate, 
            use_bidirectional, 
            use_batch_norm, 
            activation
        )

    def _build_model(self, input_shape, output_steps, lstm_units, num_layers, 
                     dropout_rate, use_bidirectional, use_batch_norm, activation):
        """
        Build a more complex LSTM model with multiple layers and regularization.

        Args:
            input_shape (tuple): Shape of the input (timesteps, features)
            output_steps (int): Number of future timesteps to predict
            lstm_units (int): Number of LSTM units
            num_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate for regularization
            use_bidirectional (bool): Whether to use bidirectional LSTM
            use_batch_norm (bool): Whether to use batch normalization
            activation (str): Activation function for dense layers

        Returns:
            model: Compiled Keras model
        """
        model = Sequential()
        model.add(Input(shape=input_shape))

        # Add LSTM layers
        for i in range(num_layers):
            return_sequences = i < num_layers - 1  # Return sequences for all but the last layer

            # Create LSTM layer
            if use_bidirectional:
                lstm_layer = Bidirectional(
                    LSTM(lstm_units, return_sequences=return_sequences)
                )
            else:
                lstm_layer = LSTM(lstm_units, return_sequences=return_sequences)

            model.add(lstm_layer)

            # Add batch normalization if enabled
            if use_batch_norm:
                model.add(BatchNormalization())

            # Add dropout for regularization
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        # Output layers
        model.add(Dense(128, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate/2))
        model.add(Dense(output_steps))
        model.add(tf.keras.layers.Reshape((output_steps, 1)))  # output: (batch, output_steps, 1)

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def fit(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, patience=10):
        """
        Train the LSTM model with early stopping.

        Args:
            X_train (ndarray): Input sequence data
            y_train (ndarray): Target sequence data
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size
            validation_split (float): Fraction of training data for validation. 
                                      Consider using manual chronological split for time series.
            patience (int): Number of epochs with no improvement after which training will be stopped.
        """
        # Definiáljuk az Early Stopping callback-et
        # Figyeli a validációs veszteséget (val_loss)
        # patience: Hány epoch-ig vár javulás nélkül, mielőtt leállna
        # restore_best_weights: Visszaállítja a legjobb súlyokat a tanítás végén
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Adjuk át a callback-et a fit metódusnak a callbacks listában
        print(f"Training started with EarlyStopping (patience={patience})...")
        history = self.model.fit(
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split, 
            callbacks=[early_stopping], # Itt adjuk hozzá a callback-et
            verbose=1
        )
        print("Training finished.")
        return history # Opcionálisan visszaadhatjuk a history objektumot

    def predict(self, data):
        """
        Predict future values using the LSTM model.

        Args:
            data (array-like): Input data of shape (samples, timesteps, features)

        Returns:
            array-like: Predicted values of shape (samples, output_steps, features)
        """
        preds = self.model.predict(data)

        if self.scaler:
            # Reshape for inverse transform if scaler was fit on 2D
            batch, steps, features = preds.shape
            # Ellenőrizzük, hogy a scaler létezik-e és van-e inverse_transform metódusa
            if hasattr(self.scaler, 'inverse_transform'):
                try:
                    # Feltételezzük, hogy a scaler 1 feature-re lett illesztve, ha a features=1
                    # Ha több feature van, a reshape és inverse_transform logikát lehet, hogy igazítani kell
                    num_features_scaled = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else features
                    
                    # Fontos: A reshape az inverse_transform előtt illeszkedjen ahhoz, ahogy a scaler-t illesztették
                    # Gyakran (samples * steps, num_features_scaled) alakra van szükség
                    reshaped_preds = preds.reshape(-1, features)
                    
                    # Ha a scaler kevesebb feature-re lett illesztve, mint a modell kimenete (ami itt 1),
                    # akkor valószínűleg csak ezt az 1 feature-t kell visszaalakítani.
                    # Ha a scaler több feature-re lett illesztve, akkor dummy adatokat kellhet hozzáadni
                    # a visszaalakításhoz, vagy a scaler-t csak a célváltozóra kellett volna illeszteni.
                    # Az egyszerűség kedvéért feltételezzük, hogy a scaler 1 feature-re (a célváltozóra) lett illesztve.
                    if features == 1 and num_features_scaled == 1:
                         inverted_preds = self.scaler.inverse_transform(reshaped_preds)
                         preds = inverted_preds.reshape(batch, steps, features)
                    elif features == 1 and num_features_scaled > 1:
                        # Ha a scaler több oszlopra lett illesztve, de csak egyet jósolunk,
                        # létre kell hozni egy megfelelő alakú tömböt az inverse_transformhoz.
                        # Ez feltételezi, hogy a jósolt 'close' az ELSŐ oszlop volt a scaler illesztésekor.
                        # EZT A RÉSZT FELÜL KELL VIZSGÁLNI AZ ADATFELDOLGOZÁS ALAPJÁN!
                        print(f"Warning: Inverse transform assumes the predicted feature was the first feature during scaling.")
                        placeholder = np.zeros((reshaped_preds.shape[0], num_features_scaled))
                        placeholder[:, 0] = reshaped_preds[:, 0] # Betöltjük a jóslatokat az első oszlopba
                        inverted_placeholder = self.scaler.inverse_transform(placeholder)
                        preds = inverted_placeholder[:, 0].reshape(batch, steps, 1) # Csak az első (jósolt) oszlopot vesszük vissza
                    else:
                         # Kezelni kell az esetet, ha több feature-t jósol a modell (features > 1)
                         # vagy ha a scaler feature száma nem egyezik.
                         print(f"Warning: Scaler feature count ({num_features_scaled}) and model output feature count ({features}) mismatch requires specific handling.")
                         # Ebben az esetben nem végezzük el az inverz transzformációt, vagy hibát dobunk.
                         # A biztonság kedvéért most nem alakítunk vissza:
                         pass # vagy raise ValueError("Feature mismatch during inverse scaling")

                except Exception as e:
                    print(f"Error during inverse scaling: {e}")
                    # Hiba esetén a skálázott predikciókat adjuk vissza
            else:
                print("Warning: Scaler does not have an inverse_transform method.")

        return preds