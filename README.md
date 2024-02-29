# Objectives
This project implements an LSTM (Long Short-Term Memory) model for stock price prediction using Python, TensorFlow, and Keras. The model is optimized using various configurations of units, LSTM layers, and Dense layers, and it is trained on historical stock price data obtained from website [stooq](https://stooq.com/).

## Packages Used
- TensorFlow
- Keras
- pandas
- numpy
- scikit-learn

## Files

### [data_preparation](yy_packages/data_preparation.py)
Contains the `PrepareData` class, which handles the preprocessing of raw stock price data, including feature engineering and generating input-output pairs for training the LSTM model.

### [LSTM_model](yy_packages/LSTM_model.py)
Contains the `LSTMModelGenerator` class, which is responsible for building, optimizing, and evaluating LSTM models.

### [main](main.py)
The main script that demonstrates how to use the `LSTMModelGenerator` and `PrepareData` classes to build and optimize an LSTM model for stock price prediction.

## Usage

1. Run `main.py` after setting the appropriate parameters such as the stock ticker, start and end dates, and the configuration options for units, LSTM layers, and Dense layers.

2. The script will preprocess the data, optimize the LSTM model with different configurations, and output the best model's evaluation metrics on the test dataset.

3. Adjust the hyperparameters and configurations in `main.py` and other files as needed to experiment with different settings for the LSTM model.

## References
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
