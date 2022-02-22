import pandas as pd
import numpy as np
from settings import _TH_FIXA_DUR

def _detect_by_gazepoint_filter(df, th_fixa_dur=None, th_sacc_dur=None):
    """ 
    Detect fixations, saccades and blinks using the annotation provided 
    by Gazepoint's internal filter.

    Parameters
    ----------

    Returns
    -------

    """
    def _detect_fixations(df, th_fixa_dur):
        fixa_groups = df.groupby(by=['FPOGID'])
        # Should we ignore the first fixation (as it could be the remaining 
        # of the last fixation in the previous session)?
        fixations = []
        session_time = df['TIME'].values[0]
        for fpogid, fpogs in fixa_groups:
            fpogs_valid = fpogs[fpogs['FPOGV'] == 1]
            fixa_x = fpogs_valid['FPOGX'].mean()
            fixa_y = fpogs_valid['FPOGY'].mean()
            fixa_dur = fpogs_valid['FPOGD'].values[-1]
            fixa_time = fpogs_valid['TIME'].values[0] - session_time

            if th_fixa_dur is not None and fixa_dur * 1000 <= th_fixa_dur: 
                # Fixation duration requirement not meet.
                continue

            if 0 <= fixa_x <= 1 and 0 <= fixa_y <= 1:
                # Only take fixation that is inside the screen.
                fixations.append([fixa_x, fixa_y, fixa_dur, fixa_time])
        return np.array(fixations)


    print(len(fixations))


def detect(df, method='gazepoint'):
    """ Detect fixations, saccades and blinks from eye-tracking data
    
    Parameters
    ----------

    Returns
    -------
    
    """
    if method=='gazepoint':
        ocular_events = _detect_by_gazepoint_filter(df)
    return ocular_events


    
if __name__ == "__main__":
    print(_TH_FIXA_DUR)
    df_data = pd.read_csv('/mnt/DATA/ltkhiem/rcir/dataset/0000/fixed_et/tracker_data_log_0.tsv', delimiter = '\t')
    detect(df_data) 
