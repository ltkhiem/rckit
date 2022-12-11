import pandas as pd
import numpy as np
from rckit.utils.checker import inside_th
from rckit.detector._gazepoint_filter import _detect_by_gazepoint_filter
from rckit.detector._wavelet_transform import _detect_by_wavelet_transform


def detect(df=None, signals=None, screen_size=None, method='gazepoint', **kwargs):
    """ Detect fixations, saccades and blinks from eye-tracking data
    
    Parameters
    ----------

    Returns
    -------
    
    """
    if method=='gazepoint':
        assert screen_size is not None, "Please specify screen_size"
        assert df is not None, "Please specify eye-tracking data in dataframe"
        ocular_events = _detect_by_gazepoint_filter(df, screen_size)
    elif method=='wavelet':
        assert signals is not None, "Please specify signals (2 channels)"
        ocular_events = _detect_by_wavelet_transform(signals, **kwargs)
    return ocular_events

