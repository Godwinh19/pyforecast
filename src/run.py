import os
import pandas as pd
from alg.nbeats import PyNbeats
from alg.sarima import Sarima
from utils.dir_helper import remove_lightning_logs_folder

DIR = os.path.dirname(os.path.dirname(__file__))

df = pd.read_csv(os.path.join(DIR, 'data\\GOOG.csv'))

method = 'sarima'

if method == 'nbeats':

    nbeats = PyNbeats(data=df, column='Close', period=30)
    train_dataloader, val_dataloader = nbeats.create_train_time_series_dataset()

    nbeats.train_model(train_dataloader, val_dataloader)
    nbeats.forecast(val_dataloader)

    # Remove the model checkpoints folder
    remove_lightning_logs_folder()

elif method == 'sarima':
    sarima = Sarima(data=df, column='Close', period=30)
    original_y_first_element, min_log_value = 0, 0

    if not sarima.is_stationary:
        original_y_first_element, min_log_value = sarima.make_dataset_stationary

    predicted_values = sarima.get_forecast(25)
    y_forecast = predicted_values['Predicted_Mean']

    y_forecast = sarima.reverse_transformation(
        y_forecast,
        original_y_first_element,
        min_log_value)