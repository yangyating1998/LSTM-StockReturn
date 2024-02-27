import pandas as pd
import pandas_datareader.data as web
import datetime
from LSTM_model import LSTMModelGenerator

# created packages
from data_preparation import PrepareData

# Get raw data from stooq
ticker = 'GOOGL'
# start = datetime.datetime(2000,1,1)
# end = datetime.datetime(2021,9,1)
# df = web.DataReader(ticker, 'stooq', start, end)
#
# # store and read data
# df.to_csv(f'{ticker}.csv')
df = pd.read_csv(f'{ticker}.csv')

# Clean data
cleaner = PrepareData(df,5,4)
X, y= cleaner.clean()

units_package, layers_package, dense_package = [8], [1,3], [1]

# Train Model
model_optimization = LSTMModelGenerator(X, y)
model_config = model_optimization.model_optimization(units_package, layers_package, dense_package)
print(model_config)
