import pandas as pd
from abc import ABC


class TimeSeries(ABC):

    def make_time_series(self, df, nbeat=False):
        """
        This method prepare the data as time series data
        :param df:
        :param nbeat:
        :param date_column:
        :param datetime:
        :return:
        DataFrame: Time series data
        """
        data = df[[self.column]]  # .astype('float')
        data.columns = ['value']
        if nbeat:
            data['time_idx'] = data.index
            data['series'] = 0

        return data
