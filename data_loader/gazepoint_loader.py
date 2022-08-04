from pathlib import Path
import pandas as pd

def load(data_path, name_format, delimiter='\t', nblocks=None):
    parts_path = [Path(data_path) / name_format.format(x) for x in range(nblocks)]
    dfs_raw = [pd.read_csv(path, delimiter=delimiter) for path in parts_path]
    return dfs_raw
    
def epoch(dfs, annotation):
    dfs_events = [df_event for df in dfs for event, df_event in df.groupby((df['USER'].shift() != df['USER']).cumsum()) if annotation in list(df_event['USER'])[0]] 
    return dfs_events

def load_and_epoch(data_path, name_format, annotation, delimiter='\t', nblocks=None):
    dfs_raw = load(data_path, name_format, delimiter, nblocks)
    dfs_events = epoch(dfs_raw, annotation)
    return dfs_events

