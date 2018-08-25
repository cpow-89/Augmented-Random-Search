import os
from datetime import datetime


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_current_date_time():
    return datetime.now().strftime('%Y/%m/%d %H:%M:%S')
