import os
import pandas as pd
from src.run import Forecast

DIR = os.path.dirname(__file__)

df = pd.read_csv(os.path.join(DIR, 'data\\GOOG.csv'))

forecast = Forecast(dataframe=df, column='Close', method='nbeats', period=30)
predictions = forecast.run
print(predictions)
