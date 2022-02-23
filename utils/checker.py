def inside_th(thresh, value):
    """
    Check if a value satisfies a given threshold.

    Parameters
    ----------
    thresh : tuple, or list of size 2
        The thrsesholds. The first item specifies the lower limit
        while the second item specifies the upper limit. If either
        is None, the corresponding threshold is ignored.
    value : 
    """
    if thresh is None:
        raise ValueError("Threshold can't be None")

    if len(thresh) != 2: 
        raise ValueError("Threshold should have shape of 2")
    
    if thresh[0] is None:
        return value <= thresh[1]
    elif threshold[1] is None: 
        return value >= thresh[0]
    else:
        return thresh[0] <= value <= thresh[1]

