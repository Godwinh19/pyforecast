import os
import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from .TimeSeries import TimeSeries
from utils.constants import *

DIR = DIR_FROM_ALG


class Sarima(TimeSeries):
    def __init__(self, data, column="value", period=PERIOD):
        super(TimeSeries, self).__init__()
        """
        :param data: DataFrame
        :param column: str
        The name of forecasting column
        :param period: int
        Time series period, the time interval for values record
        :param date_column: str
        Name of date column
        """
        self.column = column
        self.period = period
        self.data = self.make_time_series(df=data)
        # self.feature = column
        self.y = self.data['value']

    @property
    def is_stationary(self) -> bool:
        """
        This property verify if the time series is stationary by using
        Augmented-Dickey Fuller test
        :return:
        """
        ad_fuller_result = adfuller(self.y.dropna(), autolag='AIC')
        p_value = ad_fuller_result[1]
        return p_value <= 0.5

    @property
    def make_dataset_stationary(self):
        """
        Transform a non stationary data points into stationary one with log
        transformation method.
        :return:
        """
        original_y_first_element = self.y.iat[0]
        y_log = np.log(self.y)
        y_log_diff = y_log.diff(periods=1)

        y_log_diff.iat[0] = np.log(original_y_first_element)
        # Here we make data positive
        min_log_value = y_log_diff.min()
        self.y = y_log_diff - min_log_value + 1
        return original_y_first_element, min_log_value

    @staticmethod
    def reverse_transformation(y_forecast, original_y_first_element, min_log_value):
        """
        By applying log transformation, this method is the reverse operation to
        get original values.
        :param y_forecast:
        :param original_y_first_element:
        :param min_log_value:
        :return:
        """
        y_forecast = y_forecast + min_log_value - 1
        y_forecast.iat[0] = np.log(original_y_first_element)
        y_forecast = np.exp(y_forecast.cumsum())
        y_forecast.to_csv(os.path.join(DIR, 'data/sarima_forecast.csv'), index=False)
        return y_forecast

    # Create Training & Testing Datasets
    @property
    def train_test_dataset(self, train_rate=0.8):
        """
        Fot training, we need to split our data into train set and validation set.
        :param train_rate:
        float: rate use to split the data, by default 80% data for training
        :return:
        """
        point_date = int(len(self.y) * train_rate)
        y_to_train = self.y[:point_date]
        y_to_val = self.y[point_date:]
        predict_date = len(self.y) - len(y_to_train)  # the number of data points for the test set
        date_val = self.y.index[point_date]
        return y_to_train, y_to_val, predict_date, date_val

    def _sarima_grid_search(self, y, seasonal_period, max_iteration=200):
        """
        Given a sarima model, we use GridSearch to find p, d, q parameters
        :param y:
        :param seasonal_period:
        :param max_iteration:
        :return:
        """
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], seasonal_period) for x in list(itertools.product(p, d, q))]
        param_mini, param_seasonal_mini = (1, 0, 1), (1, 1, 0, self.period)
        mini = float('+inf')

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(y,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)

                    results = mod.fit(maxiter=max_iteration)

                    if results.aic < mini:
                        mini = results.aic
                        param_mini = param
                        param_seasonal_mini = param_seasonal

                except:
                    continue
        return param_mini, param_seasonal_mini

    @staticmethod
    def _sarima_eva(y, order, seasonal_order, max_iteration=200):
        """
        Evaluate the model
        :param y:
        :param order:
        :param seasonal_order:
        :param max_iteration:
        :return:
        """
        # fit the model
        mod = sm.tsa.statespace.SARIMAX(y,
                                        order=order,
                                        seasonal_order=seasonal_order,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False,)
        return mod.fit(maxiter=max_iteration)

    @staticmethod
    def _forecast(model, predict_steps):
        """
        Give forecasting values given the predict_steps
        :param model:
        :param predict_steps:
        :return:
        """

        pred_uc = model.get_forecast(steps=predict_steps)

        # Produce the forecasted tables
        predicted_mean_df = pred_uc.predicted_mean.reset_index()
        predicted_mean_df.columns = ['Date', 'Predicted_Mean']
        return predicted_mean_df

    def get_forecast(self, time):
        """
        Process to get forecast
        :param time:
        :return:
        """
        param_mini, param_seasonal_mini = self._sarima_grid_search(self.y, self.period)
        model = self._sarima_eva(self.y, param_mini, param_seasonal_mini)
        final_table = self._forecast(model, time)
        return final_table
