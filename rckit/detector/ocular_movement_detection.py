import pandas as pd
import numpy as np
from rckit.utils.checker import inside_th
from rckit.detector._gazepoint_filter import _detect_by_gazepoint_filter
from rckit.detector._wavelet_transform import _detect_by_wavelet_transform


def detect(df=None, signals=None, screen_size=None, method='gazepoint', **kwargs):
    """ Detect fixations, saccades and blinks from eye-tracking data
    
    Parameters
    ----------
    df : pandas.DataFrame, optional
        Eye-tracking data (from Gazepoint)

    signals : array-like of shape (2, n_samples), optional
        Eye movement data which is represented in two channels: vertical
        and horizontal.

    screen_size : tuple of int, optional
        Screen size (width, height) in pixels.

    method : str, optional, default='gazepoint'
        Method to detect fixations, saccades and blinks. Available methods are:
        'gazepoint' (default), 'wavelet_transform'.

    fs : float, optional
        [Required if signals is given]
        Sampling frequency of eye movement signals. 

    .....

    Returns
    -------

    ocular_events: Tuple of ndarrays
        - Format is not unified yet.
        - If method is 'gazepoint', it returns a tuple of three ndarrays
            (fixations, saccades, blinks).
        - If method is 'wavelet_transform'
            - If mask_only is True 
                it returns one ndarray which is a mask of ocular events.
            - If mask_only is False
                it returns a tuple of four ndarrays
                (fixations, saccades, blinks, ocular_events_mask).
    (This will be unified in the future)
    
    """
    if method=='gazepoint':
        assert screen_size is not None, "Please specify screen_size"
        assert df is not None, "Please specify eye-tracking data in dataframe"
        ocular_events = _detect_by_gazepoint_filter(df, screen_size)
    elif method=='wavelet':
        assert signals is not None, "Please specify signals (2 channels)"
        assert kwargs.get('fs', None) is not None, "Please specify sampling frequency"
        ocular_events = _detect_by_wavelet_transform(signals, **kwargs)
    return ocular_events

