import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from data_preparation import PrepareData

class LSTMModelGenerator:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        # self.test_size = test_size
        # self.mem_day = mem_day

    def model_build(self, units, layers, dense):
        model = Sequential()
        model.add(LSTM(units, input_shape=self.X.shape[1:], activation='relu', return_sequences=True))
        model.add(Dropout(0.1))

        for l in range(layers-1):
            model.add(LSTM(units, input_shape=self.X.shape[1:], activation='relu', return_sequences=True))
            model.add(Dropout(0.1))

        model.add(LSTM(units, activation='relu'))
        model.add(Dropout(0.1))

        for d in range(dense):
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(0.1))

        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mape'])
        return model

    def model_optimization(self, units_package, layers_package, dense_package):
        i = 0
        for l in layers_package:
            for d in dense_package:
                for u in units_package:
                    i += 1
                    filepath = '{val_loss:.2f}-{epoch:02d}' + f'lstm-{l}, dense-{d}, units-{u}'+ '.h5'
                    self.model_build(u,l,d)
                    checkpoint = ModelCheckpoint(
                        filepath,
                        save_weights_only=False,
                        monitor='val_mape',
                        mode="min",
                        save_best_only=True,
                        initial_value_threshold=None,)
                    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, shuffle=False)
                    model = self.model_build(u, l, d)
                    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint])

                    val_loss = model.evaluate(X_test, y_test)[0]

                    if i == 1 or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_config = {'units': u, 'layers': l, 'dense': d}
        return best_config

