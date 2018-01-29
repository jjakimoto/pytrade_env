from __future__ import print_function

import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import tz
import time
from time import mktime
from copy import deepcopy


def datetime2date(datetime_obj):
    str_time = '%04d-%02d-%02d %02d:%02d:%02d'
    return str_time % (datetime_obj.tm_year, datetime_obj.tm_mon,
                       datetime_obj.tm_mday, datetime_obj.tm_hour,
                       datetime_obj.tm_min, datetime_obj.tm_sec)


def date2seconds(str_time):
    datetime_obj = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
    seconds = time.mktime(datetime_obj.timetuple())
    return seconds


def seconds2date(seconds):
    date_obj = time.localtime(seconds)
    return datetime2date(date_obj)


def seconds2datetime(seconds):
    time_obj = time.localtime(seconds)
    datetime_obj = datetime.fromtimestamp(mktime(time_obj))
    return datetime_obj


def date2datetime(str_time):
    datetime_obj = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
    return datetime_obj


def date2str(date):
    str_date = '%04d-%02d-%02d %02d:%02d:%02d' %\
        (date.year, date.month, date.day, date.hour, date.minute, date.second)
    return str_date


def get_time_now(is_local=True):
    date = datetime.utcnow()
    if is_local:
        to_zone = tz.tzlocal()
        # Convert time zone
        date = date.astimezone(to_zone)
        date = date.now()
    return date2str(date)


def create_sharpe_ratio(returns, periods=252):
    """
    Create the Sharpe ratio for the strategy, based on a
    benchmark of zero (i.e. no risk-free rate information).
    Parameters:
    returns - A pandas Series representing period percentage returns.
    periods - Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    """
    return np.sqrt(periods) * (np.mean(returns)) / np.std(returns)


def create_drawdowns(pnl):
    """
    Calculate the largest peak-to-trough drawdown of the PnL curve
    as well as the duration of the drawdown. Requires that the
    pnl_returns is a pandas Series.
    Parameters:
    pnl - A pandas Series representing period percentage returns.
    Returns:
    drawdown, duration - Highest peak-to-trough drawdown and duration.
    """
    # Calculate the cumulative returns curve
    # and set up the High Water Mark
    hwm = [0]

    # Create the drawdown and duration series
    idx = pnl.index
    drawdown = pd.Series(index=idx)
    duration = pd.Series(index=idx)

    # Loop over the index range
    for t in range(1, len(idx)):
        hwm.append(max(hwm[t - 1], pnl[t]))
        drawdown[t] = (hwm[t] - pnl[t])
        duration[t] = (0 if drawdown[t] == 0 else duration[t - 1] + 1)
    return drawdown, drawdown.max(), duration.max()


def calculate_pv_after_commission(w1, w0, commission_rate):
    """
    @:param w1: target portfolio vector, first element is btc
    @:param w0: rebalanced last period portfolio vector, first element is btc
    @:param commission_rate: rate of commission fee, proportional to the transaction cost
    """
    w1 = deepcopy(w1)
    w0 = deepcopy(w0)
    mu0 = 1
    mu1 = 1 - 2 * commission_rate + commission_rate ** 2
    while abs(mu1 - mu0) > 1e-10:
        mu0 = mu1
        mu1 = (1 - commission_rate * w0[0] -
            (2 * commission_rate - commission_rate ** 2) * \
            np.sum(np.maximum(w0[1:] - mu1*w1[1:], 0))) / \
            (1 - commission_rate * w1[0])
    return mu1
