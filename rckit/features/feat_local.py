import pandas as pd
from tsfresh import extract_features as tsf_extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

def gen_window(signal_size, window_size, window_shift, sampling_rate=150, include_last=True):
    """
    Generate windows of a given size and shift from a given event table.
    """
    ret = []
    wsz = int(window_size * sampling_rate)
    wsh = int(window_shift * sampling_rate)
    for i in range(0, signal_size-wsz+1, wsh):
        if i + wsz > signal_size:
            if include_last:
                 ret.append([i, signal_size])
            break
        ret.append([i, i+wsz])
    return ret


def get_window_features(etdata, 
                        window_size, 
                        window_shift, 
                        use_columns,
                        feature_settings=None,
                        sampling_rate=150, 
                        include_last=True):
    """
    Divide eye-tracking samples into smaller windows and extract time-series features for
    each window. 

    Parameters
    ----------
    etdata : list of pandas.DataFrame or pandas.DataFrame
        Dataframe of an eye-tracking sample, or List of eye-tracking samples.

    window_size : float
        Size of the window in seconds.

    window_shift : float
        Shift of the window in seconds.

    use_columns : list of str
        List of columns in eye-tracking sample  to use for feature extraction.

    feature_settings : dict
        Settings for tsfresh feature extraction.

    sampling_rate : int
        Sampling rate of eye-tracking samples.

    include_last : bool
        Whether to include the last window in the eye-tracking sample if it is smaller 
        than the window_size.

    Returns
    -------
    window_feats : pandas.DataFrame
        Dataframe with extracted features for each window for each eye-tracking sample.
        Each row corresponds to a window and has a window_id column that identifies the
        window. 

    """
    def _check_feature_settings():
        if feature_settings is None:
            return 
        tsf_feats = ComprehensiveFCParameters()
        for feat, params in feature_settings.items():
            if feat not in tsf_feats:
                raise ValueError("Unknown feature: {}".format(feat))
            if params is None:
                # Use default parameters if None is given
                feature_settings[feat] = tsf_feats[feat]
              
    _check_feature_settings()
    if isinstance(etdata, pd.DataFrame):
        etdata = [etdata]
    
    windows = []
    for sample_id, et in enumerate(etdata):
        window_pos = gen_window(len(et), window_size, window_shift, sampling_rate, include_last)
        for w_id, [start, end] in enumerate(window_pos):
            window = et.iloc[start:end][use_columns]
            window['window_id'] = f'{sample_id}_{w_id}_{start}_{end}'
            windows.append(window)
    windows = pd.concat(windows)
    feat_windows =  tsf_extract_features(windows, column_id='window_id',
                                         default_fc_parameters=feature_settings,
                                         disable_progressbar=True)
    feat_windows.reset_index(inplace=True)
    feat_windows[['sample_id', 'window_id', 'start', 'end']] = feat_windows['index'].str \
                                                                .split('_', expand=True) \
                                                                .astype(int)
    feat_windows.drop(columns=['index'], inplace=True)
    feat_windows = feat_windows[['sample_id', 'window_id', 'start', 'end'] + list(feat_windows.columns[:-4])]
    return feat_windows

    
            
         
if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('temp/all.csv')

    feature_settings = {
        'abs_energy': None,
        'autocorrelation': None,
    }
    # Generate windows
    windows = get_window_features([df], window_size=1, window_shift=0.5, use_columns=['FPOGX', 'FPOGY'],
                                  feature_settings=feature_settings)


    # df = df.loc[(df['FPOGX']>0) & (df['FPOGX']<1) & (df['FPOGY']>0) & (df['FPOGY']<1), :]
    # df = df.reset_index(drop=True)
    # last_index = len(df)
    
    # # Define the window size and shift
    # # window_size = 5 /2  1# seconds
    # # window_shift = 3 /1 0.75 # seconds
    # # Elbow method fine K

    # windowed_data_list = []
    
    # # Iterate over the groups and extract the windowed data
    # i = 0
    # current_index = 0
    # while i == 0:
    #     if current_index == 0:
    #         window_start = df.at[0, 'TIME_STAMP']
    #     else:
    #         window_start = window_start + window_shift
    #     window_end = window_start + window_size
    #     windowed_data = df[(df['TIME_STAMP'] >= window_start) & (df['TIME_STAMP'] < window_end)]
    #     if windowed_data.empty or (len(windowed_data) == 0 and len(windowed_data.columns) > 0):
    #         continue
    #     else:
    #         current_index = len(df[df['TIME_STAMP'] < window_end])
    #         windowed_data_list.append(windowed_data)
    #     if current_index >= last_index:
    #         i = 1
    
    # # Concatenate the windowed data into a single DataFrame
    # window_index = [f'{i}' for i in range(len(windowed_data_list))]
    # windowed_data_df = pd.concat(windowed_data_list, keys=window_index, names=['WINDOW_INDEX'])
    # windowed_data_df['WINDOW_INDEX'] = windowed_data_df.index.get_level_values('WINDOW_INDEX')
    
    # windowed_data_df.to_csv(Path('window_data')/path/'tracker_data_log_windowed_'f'{window_size}''_'f'{window_shift}''.tsv', sep='\t', index=False)
    
    
