from collections import deque
import numpy as np
from sklearn.model_selection import train_test_split

class PrepareData:
    def __init__(self, data, mem_days):
        self.data = data
        self.pre_days = mem_days+1
        self.mem_days = mem_days

    def clean(self):
        # clean raw data-drop empty ones
        self.data.dropna(inplace=True)
        self.data.sort_index(ascending=False, inplace=True)

        # data preprocessing - percentage change
        close = self.data['Close']
        open = self.data['Open']
        high = self.data['High']
        low = self.data['Low']
        volume = self.data['Volume']

        # generate X
        self.data['intra_day_volatility'] = (high-low)/low
        self.data['volume_volatility'] = (volume-volume.shift(-1))/volume.shift(-1)
        self.data['intra_day_return'] = (close-open)/open
        self.data['daily_return'] = (close-close.shift(-1))/close.shift(-1)

        # generate y
        self.data['output'] = self.data.daily_return.shift(-self.pre_days)
        self.data.dropna(inplace=True)

        self.data = self.data[1:]

        x_std = np.array(self.data.iloc[:, -4:])
        x_queue = deque(maxlen=self.mem_days)
        X = []

        for i in x_std:
            x_queue.append(list(i))
            if len(x_queue) == self.mem_days:
                X.append(list(x_queue))

        X = X[:-self.pre_days+self.mem_days-1]
        X = np.array(X)
        y = np.array(self.data['output'][:-self.pre_days])

        return X, y

    def train_test_data(self):
        X, y = self.clean()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test
