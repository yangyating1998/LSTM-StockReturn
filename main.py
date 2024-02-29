import pandas_datareader.data as web
import datetime
from yy_packages.LSTM_model import LSTMModelGenerator
from yy_packages.data_preparation import PrepareData

class OptimizedLSTMmodel:
    def __init__(self, ticker, start_day, end_day, mem_days = 5, units_packages = [8,16,32], layers_package=[1,2,3], dense_package=[1,2]):
        self.ticker = ticker
        self.start = start_day
        self.end = end_day
        self.mem_days = mem_days
        self.units_package = units_packages
        self.layers_package = layers_package
        self.dense_package = dense_package

    def preprocessed_data(self): # cleaned raw data; preprocessed for further machine learning
        df = web.DataReader(self.ticker, 'stooq', self.start, self.end)
        cleaner = PrepareData(df, self.mem_days)
        X_train, X_test, y_train, y_test = cleaner.train_test_data()
        return X_train, X_test, y_train, y_test

    def optimize_model(self): # find the best model given several units, layers, and dense; also give the best model's evaluation
        X_train, X_test, y_train, y_test = self.preprocessed_data()
        model_optimization = LSTMModelGenerator(X_train, X_test, y_train, y_test)
        best_config, model_path, best_val_loss, test_evaluation = model_optimization.model_optimization(self.units_package, self.layers_package, self.dense_package)
        return best_config, model_path, best_val_loss, test_evaluation


# Example
ticker = 'GOOGL'
start_day = datetime.datetime(2000,1,1)
end_day = datetime.datetime(2021,9,1)
units_package, layers_package, dense_package = [8], [2], [2]

tester = OptimizedLSTMmodel(ticker, start_day, end_day, 5, units_package, layers_package, dense_package)
best_config, model_path, best_val_loss, test_evaluation = tester.optimize_model() # return best_config, best_model_path, best_model_evaluation
print(test_evaluation)