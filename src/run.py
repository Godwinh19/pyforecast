import os
import pandas as pd
from alg.nbeats import PyNbeats
from utils.dir_helper import remove_lightning_logs_folder

DIR = os.path.dirname(os.path.dirname(__file__))

df = pd.read_csv(os.path.join(DIR, 'data\\GOOG.csv'))

nbeats = PyNbeats(data=df, column='Close', period=20, date_column='Date')
train_dataloader, val_dataloader = nbeats.create_train_time_series_dataset()

nbeats.train_model(train_dataloader, val_dataloader)
nbeats.forecast(val_dataloader)

# Remove the model checkpoints folder
remove_lightning_logs_folder()
