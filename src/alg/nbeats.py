import os
import shutil
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_lightning.callbacks import EarlyStopping
from .TimeSeries import TimeSeries
from src.utils.constants import *
from src.utils.sub_functions import (seed, )

warnings.filterwarnings('ignore')

DIR = DIR_FROM_ALG


class PyNbeats(TimeSeries):
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
        self.data = self.make_time_series(df=data, nbeat=True)
        self.training = None

    def create_train_time_series_dataset(self, max_encoder_length=50, max_prediction_length=25,
                                         batch_size=128, num_workers=0):
        """
        Create the train et val dataset for the learning phase
        :param max_encoder_length: int
        :param max_prediction_length: int
        :param batch_size: int
        :param num_workers: int
        :return:
        dataloader: train_dataloaser and val_dataloader
        """
        # max_encoder_length = n*max_prediction_length n E [2,7]

        training_cutoff = self.data['time_idx'].max() - max_prediction_length

        context_length = max_encoder_length
        prediction_length = max_prediction_length

        self.training = TimeSeriesDataSet(
            self.data[lambda x: x.time_idx <= training_cutoff],
            time_idx='time_idx',
            target='value',
            categorical_encoders={'series': NaNLabelEncoder().fit(self.data.series)},
            group_ids=['series'],
            time_varying_unknown_reals=["value"],
            max_encoder_length=context_length,
            max_prediction_length=prediction_length,
        )

        validation = TimeSeriesDataSet.from_dataset(self.training, self.data, min_prediction_idx=training_cutoff + 1)
        train_dataloader = self.training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

        return train_dataloader, val_dataloader

    def create_forecast_dataset(self):
        """
        This method prepare the forecast dataset
        :return:
        """
        pass

    def train_model(self, train_dataloader, val_dataloader, epochs=20, gpus=0, options={}):
        """
        This method performs the training step
        :param train_dataloader:
        :param val_dataloader:
        :param epochs:
        :param gpus:
        :param options: dictionary for additional parameters
        :return:
        """
        seed()
        trainer = pl.Trainer(gpus=gpus, gradient_clip_val=0.01)
        widths = options['widths'] if 'widths' in options.keys() else [32, 512]
        limit_train_batches = options['limit_train_batches'] if 'limit_train_batches' in options.keys() else 30
        
        net = NBeats.from_dataset(self.training, learning_rate=3e-2, weight_decay=1e-2, widths=widths,
                                  backcast_loss_ratio=0.1)
        # find optimal learning rate
        res = trainer.tuner.lr_find(net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
                                    min_lr=1e-5)
        net.hparams.learning_rate = res.suggestion()

        # Fit the model
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        trainer = pl.Trainer(
            max_epochs=epochs,
            gpus=gpus,
            weights_summary="top",
            gradient_clip_val=0.01,
            callbacks=[early_stop_callback],
            limit_train_batches=limit_train_batches, # TO-DO : need to be dynamic
        )

        net = NBeats.from_dataset(
            self.training,
            learning_rate=net.hparams.learning_rate,
            log_interval=10,
            log_val_interval=1,
            weight_decay=1e-2,
            widths=widths,
            backcast_loss_ratio=1.0,
        )

        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # Get the best model
        best_model_path = trainer.checkpoint_callback.best_model_path
        model_name = best_model_path.split('\\')[-1]
        shutil.copy(best_model_path, os.path.join(DIR, 'model'))
        try:
            os.remove(os.path.join(DIR, 'model/nbeats.ckpt'))
        except OSError:
            pass
        os.rename(os.path.join(DIR, 'model/' + model_name), os.path.join(DIR, 'model/nbeats.ckpt'))

    @staticmethod
    def forecast(dataloader):
        """
        Given the dataloader, return the forecast dataframe
        :param dataloader:
        :return:
        """
        model = NBeats.load_from_checkpoint(os.path.join(DIR, 'model/nbeats.ckpt'))
        raw_predictions = model.predict(dataloader, mode="raw", return_x=False)
        y = np.array(raw_predictions['prediction'].reshape(-1, 1))
        df = pd.DataFrame(data=y, columns=['prediction'])
        df.to_csv(os.path.join(DIR, 'data/nbeats_forecast.csv'), index=False)
        return y

