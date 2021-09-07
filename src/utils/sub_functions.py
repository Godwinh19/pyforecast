import pytorch_lightning as pl


def seed(value=42):
    return pl.seed_everything(value)
