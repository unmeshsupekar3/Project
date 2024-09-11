import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

class TimeSeriesARIMA:
    def __init__(self, file_path, date_col='timestamp', value_col='value', date_format="%Y-%m-%d %H:%M:%S"):
        self.file_path = file_path
        self.date_col = date_col
        self.value_col = value_col
        self.date_format = date_format
        self.data = None
        self.model = None
        self.predictions = []

    def load_data(self):
        data = pd.read_csv(self.file_path)
        data[self.date_col] = pd.to_datetime(data[self.date_col], format=self.date_format)
        data.set_index(self.date_col, inplace=True)
        self.data = data[self.value_col]

    def plot_data(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.data.index, self.data.values)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Time Series Data')
        plt.show()

    def fit_model(self, order=(2, 1, 0)):
        self.model = ARIMA(self.data, order=order).fit()
        print(self.model.summary())

    def plot_residuals(self):
        residuals = pd.DataFrame(self.model.resid)
        residuals.plot()
        plt.title('Residuals')
        plt.show()
        residuals.plot(kind='kde')
        plt.title('Residuals Density')
        plt.show()
        print(residuals.describe())

    def forecast(self, split_ratio=0.66, limit_count=50):
        X = self.data.values
        size = int(len(X) * split_ratio)
        train, test = X[0:size], X[size:size+limit_count]
        history = [x for x in train]
        self.predictions = []

        for t in range(len(test)):
            model = ARIMA(history, order=(2, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            self.predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('pred=%f, exp=%f' % (yhat, obs))

        error = mean_squared_error(test, self.predictions)
        print('Mean Squared Error: %.3f' % error)

        plt.plot(test, label='Actual')
        plt.plot(self.predictions, color='red', label='Predicted')
        plt.legend()
        plt.title('Actual vs Predicted')
        plt.show()

if __name__ == "__main__":
    file_path = r'C:\Users\unmes\Documents\Projects\anomaly_detection\data\realAWSCloudwatch\ec2_cpu_utilization_825cc2.csv'
    ts_arima = TimeSeriesARIMA(file_path)
    ts_arima.load_data()
    ts_arima.plot_data()
    ts_arima.fit_model()
    ts_arima.plot_residuals()
    ts_arima.forecast()
