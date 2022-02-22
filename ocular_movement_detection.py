import pandas as pd
import numpy as np
from settings import _TH_FIXA_DUR

def _detect_by_gazepoint_filter(df, th_fxdur=None, th_scdur=None):
    """ 
    Detect fixations, saccades and blinks using the annotation provided 
    by Gazepoint's internal filter.

    Parameters
    ----------

    Returns
    -------

    """
    def _detect_fixations():
        fixa_groups = df.groupby(by=['FPOGID'])
        # Should we ignore the first fixation (as it could be the remaining 
        # of the last fixation in the previous session)?
        fixations = []
        session_time = df['TIME'].values[0]
        for fpogid, fpogs in fixa_groups:
            fpogs_valid = fpogs[fpogs['FPOGV'] == 1]
            fxh = fpogs_valid['FPOGX'].mean()
            fxv = fpogs_valid['FPOGY'].mean()
            fxdur = fpogs_valid['FPOGD'].values[-1]
            fxtime = fpogs_valid['TIME'].values[0] - session_time

            if th_fxdur is not None and fxdur * 1000 <= th_fxdur: 
                # Fixation duration requirement not meet.
                continue

            if 0 <= fxh <= 1 and 0 <= fxv <= 1:
                # Only take fixation that is inside the screen.
                fixations.append([fxh, fxv, fxdur, fxtime])
        return np.array(fixations)

    def _detect_saccades():
        print(fixations[0:4])
        fxend = (fixations[:, 2] + fixations[:, 3]).reshape(-1,1)
        print(fxend[:4])
        time_diff = (fixations[1:][:, 3] - fxend[:-1, 0]).reshape(-1,1)
        print(time_diff[:4])
        saccades = np.concatenate([
                fixations[:-1][:, :2],  # start_x, start_y
                fixations[1:][:, :2],   # end_x, end_y
                time_diff,              # saccades duration
                fxend[:-1]               # saccades start will be end of last fixation
            ], axis=1)
        return saccades

    fixations = _detect_fixations()
    saccades = _detect_saccades()
    print(fixations[0:4])
    print(saccades[0:4])
    print(fixations.shape, saccades.shape)
    return fixations, saccades


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
