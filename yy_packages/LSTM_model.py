import tensorflow
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from data_preparation import PrepareData
from keras.models import load_model
import numpy as np


class LSTMModelGenerator:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # self.model_path = None

    def model_build(self, units, layers, dense):
        model = Sequential()

        for l in range(layers-1):
            model.add(LSTM(units, input_shape=self.X_train.shape[1:], activation='relu', return_sequences=True))
            model.add(Dropout(0.1))

        model.add(LSTM(units, activation='relu'))
        model.add(Dropout(0.1))

        for d in range(dense-1):
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(0.1))

        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mape'])
        return model

    def model_optimization(self, units_package, layers_package, dense_package):
        best_val_loss = float('inf')
        best_model = None
        best_config = {}
        model_path = ""

        for l in layers_package:
            for d in dense_package:
                for u in units_package:

                    filepath = 'Desktop/{epoch:02d}{val_loss:.2f}' + f'Desktop/lstm-{l}dense-{d}units-{u}'
                    self.model_build(u,l,d)
                    checkpoint = ModelCheckpoint(
                        filepath,
                        save_weights_only=False,
                        monitor='val_mape',
                        mode="min",
                        initial_value_threshold=None)
                    model = self.model_build(u, l, d)
                    history = model.fit(self.X_train, self.y_train, batch_size=32, epochs=50, validation_split=0.2, callbacks=checkpoint, shuffle=False)
                    trained_val_loss = history.history['val_loss'][-1]

                    if trained_val_loss < best_val_loss:
                        best_val_loss = trained_val_loss
                        best_config = {'units': u, 'layers': l, 'dense': d}
                        model_path = filepath
                        best_model = model
        # best_model.fit(self.X_train, self.y_train)
        test_evaluation = best_model.evaluate(self.X_test, self.y_test)
        return best_config, model_path, best_val_loss, test_evaluation