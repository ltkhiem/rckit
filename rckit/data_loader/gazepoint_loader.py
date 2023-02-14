from pathlib import Path
import pandas as pd

def load(data_path, name_format, delimiter='\t', nblocks=None):
    """
    Load eye tracking data from a directory of file(s).

    Parameters
    ----------
    data_path : str
        Path to the directory containing the data file(s).

    name_format : str
        Format string for the data file(s). If there are multiple data files, 
        the format string should contain a single placeholder for the part number.

    delimiter : str, optional, default='\t'
        Delimiter used in the data files. 

    nblocks : int, optional, default=None
        Number of parts to load. If None, it will treat the data as a single part.

    Returns
    -------
    dfs : list of pandas.DataFrame
        List of eye tracking data sessions (a.k.a parts).
        If nblocks is None, the list contains a single element.
    """

    if nblocks != None:
        parts_path = [Path(data_path) / name_format.format(x) for x in range(nblocks)]
        dfs_raw = [pd.read_csv(path, delimiter=delimiter) for path in parts_path]
    else:
        dfs_raw = [pd.read_csv(path/name_format, delimiter=delimiter)]
    return dfs_raw
    
def epoch(dfs, annotation):
    """
    Divide eye tracking data into smaller segments based on annotations.
    It relies on the 'USER' column which was provided by the Gazepoint eye tracker 
    for annotation purposes. 

    Note: The extracted segments from all sessions (where possible) will be merged
    after this function is executed. The order will be kept as the order of the
    sessions in the input list.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        List of eye tracking data sessions.
        
    annotation : pandas.DataFrame
        Annotation text.

    Returns 
    -------
    dfs : list of pandas.DataFrame
        List of eye tracking data divided into segments based on annotations.
    """

    dfs_events = [df_event for df in dfs for event, df_event in df.groupby((df['USER'].shift() != df['USER']).cumsum()) if annotation in list(df_event['USER'])[0]] 
    return dfs_events

def load_and_epoch(data_path, name_format, annotation, delimiter='\t', nblocks=None):
    """
    Load eye tracking data from a directory of file(s) and divide them into smaller segments
    based on annotations.

    Parameters
    ----------
    data_path : str
        Path to the directory containing the data file(s).
                
    name_format : str
        Format string for the data file(s). If there are multiple data files,
        the format string should contain a single placeholder for the part number.
    
    annotation : pandas.DataFrame
        Annotation text.

    delimiter : str, optional, default='\t'
        Delimiter used in the data files.
            
    nblocks : int, optional, default=None
        Number of parts to load. If None, it will treat the data as a single part.
            
    Returns
    -------
    dfs : list of pandas.DataFrame
        List of eye tracking data divided into segments based on annotations.
    """

    dfs_raw = load(data_path, name_format, delimiter, nblocks)
    dfs_events = epoch(dfs_raw, annotation)
    return dfs_events

