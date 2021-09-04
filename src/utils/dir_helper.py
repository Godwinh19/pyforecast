import os
import shutil

DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def remove_lightning_logs_folder():
    try:
        shutil.rmtree(os.path.join(DIR, 'lightning_logs'))
    except OSError:
        return False
    else:
        return True
