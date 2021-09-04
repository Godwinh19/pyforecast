import pandas as pd
from abc import ABC


class TimeSeries(ABC):

    def make_time_series(self, df, nbeat=False, date_column=None, datetime='s'):
        """
        This method prepare the data as time series data
        :param df:
        :param nbeat:
        :param date_column:
        :param datetime:
        :return:
        DataFrame: Time series data
        """
        data = df[[self.column]].astype('float')
        data.columns = ['value']
        if nbeat:
            data['time_idx'] = data.index
            data['series'] = 0

        # Convert to time series
        if not date_column:
            date = [i for i in range(0, data.shape[0] * self.period, self.period)]
            data = data.join(pd.DataFrame(data=date, columns=['date']))
            data['date'] = pd.to_datetime(data['date'], unit=datetime)
        else:
            data['date'] = df[date_column]

        return data
