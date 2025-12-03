from scdata.tools.custom_logger import logger

from pandas import DatetimeIndex, Series, Timedelta, factorize
import numpy as np

def infer_sampling_rate(series: Series) -> int | None:
    '''Infer the sampling rate of the given timeseries, rounded to the
    closest minute.
    '''

    time_differences = series.index.diff().value_counts()
    most_common = time_differences.index[0]

    minutes = most_common / Timedelta("1min")
    integer_minutes = round(minutes)

    if abs(integer_minutes - minutes) > 0.05:
        logger.warning('Rounded a time difference with more than 5% error')
        return None

    return integer_minutes


def mode_ratio(series: Series, ignore_zeroes=True) -> int:
    '''Count the percentage of times the most common value appears in the series,
    ignoring zeroes and NaNs.'''

    if ignore_zeroes:
        # Replace zeroes with random so that they don't impact value count
        series = series.where(series!=0.0, np.random.random(size=series.size))

    mode_count = series.value_counts().iloc[0]

    return mode_count / series.count()


def count_nas(series: Series) -> int:
    '''Count the number of NaN values in the series.'''
    return series.isna().sum()


def rolling_deltas(series: Series) -> Series:
    '''Compute the first derivative of the series at each datapoint.'''

    dys = series.rolling(window=2).apply(lambda ys: ys.iloc[1] - ys.iloc[0])
    dxs = series.index.diff().total_seconds()

    return dys / dxs


def normalize_central(series: Series, pct=0.05) -> Series:
    '''Normalize the series by removing the mean and scaling to unit variance,
    ignroring the top and bottom `pct` percent of values. This should be more
    robust to outliers than standard normalization.'''

    central = series[((series > series.quantile(pct)) | (series > series.quantile(1 - pct)))]
    normalized = (series - central.mean()) / central.std()

    return normalized


def rolling_top_value_ratio_time(values: np.ndarray,
                                  times: DatetimeIndex,
                                  window_td: Timedelta) -> np.ndarray:
    """
    values: 1D numpy array (may contain NaN; NaNs are ignored)
    times:  1D numpy array of datetime64[ns], same length as values
    window_td: pandas Timedelta, e.g. pd.Timedelta("1h")

    Returns:
        1D float array: for each position i, the frequency of the most
        common value in the time window (times in (t_i - window_td, t_i]),
        or NaN if no valid values.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out

    # Factorize values once; NaNs → code -1 (ignored)
    codes, _ = factorize(values, sort=False)
    # codes: -1 for NaN, 0..k-1 for actual values
    k = codes.max() + 1 if codes.size and codes.max() >= 0 else 0
    if k == 0:
        # all NaN
        return out

    counts = np.zeros(k, dtype=np.int64)
    valid_in_window = 0

    left = 0
    win_ns = window_td.value  # nanoseconds int
    times = times.astype("int64").to_numpy()

    for right in range(n):
        # add new point
        c = codes[right]
        if c >= 0:
            counts[c] += 1
            valid_in_window += 1

        t_right = times[right]

        # shrink from left until window is (t_right - window, t_right]
        while left <= right:
            t_left = times[left]
            if t_right - t_left <= win_ns:
                break
            # remove values leaving the window
            c_left = codes[left]
            if c_left >= 0:
                counts[c_left] -= 1
                valid_in_window -= 1
            left += 1

        if valid_in_window > 0:
            out[right] = counts.max() / valid_in_window

    return out
