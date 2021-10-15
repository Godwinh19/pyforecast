from src.utils.dir_helper import remove_lightning_logs_folder


class Forecast:
    def __init__(self, dataframe, column, method='sarima', period=30):
        self.df = dataframe
        self.column = column
        self.method = method
        self.period = period

    @property
    def run(self):
        if self.method == 'nbeats':
            return self._nbeat_process
        elif self.method == 'sarima':
            return self._sarima_process
        else:
            return NotImplemented

    @property
    def _nbeat_process(self):
        from src.alg.nbeats import PyNbeats
        nbeats = PyNbeats(data=self.df, column=self.column, period=self.period)
        train_dataloader, val_dataloader = nbeats.create_train_time_series_dataset()

        nbeats.train_model(train_dataloader, val_dataloader)
        y = nbeats.forecast(val_dataloader)

        # Remove the model checkpoints folder
        remove_lightning_logs_folder()
        return y

    @property
    def _sarima_process(self):
        from src.alg.sarima import Sarima
        sarima = Sarima(data=self.df, column=self.column, period=self.period)
        original_y_first_element, min_log_value = 0, 0

        if not sarima.is_stationary:
            original_y_first_element, min_log_value = sarima.make_dataset_stationary

        predicted_values = sarima.get_forecast(self.period)
        y_forecast = predicted_values['Predicted_Mean']

        y_forecast = sarima.reverse_transformation(
            y_forecast,
            original_y_first_element,
            min_log_value)

        return y_forecast
