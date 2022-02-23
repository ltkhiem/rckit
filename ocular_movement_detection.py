import pandas as pd
import numpy as np
from settings import _TH_FIXA_DUR
from utils.checker import inside_th

def _detect_by_gazepoint_filter(df, th_fxdur=None, th_scdur=None, th_bkdur=None):
    """ 
    Detect fixations, saccades and blinks using the annotation provided 
    by Gazepoint's internal filter.

    Parameters
    ----------

    Returns
    -------

    """
    def _detect_fixations():
        """
        Take eye tracking data from outer function and perform detection.
        Only consider valid POG to calculate fixations.
        The fixations that fall outside of the screen area or a given threshold
        (if available) are eliminated.

        Returns
        -------
        fixations : array-like of shape (n_fixations, 4)
            Each element in the array is a fixation described in the format
            [fxh, fxv, fxdur, fxtime], where:
            - fxh : position of fixation on horizontal axis.
            - fxv : position of fixation on vertical axis.
            - fxdur : duration in seconds of fixation.
            - fxtime : starting time in seconds since the start of the session. 
        """
        fx_groups = df.groupby(by=['FPOGID'])
        # Should we ignore the first fixation (as it could be the remaining 
        # of the last fixation in the previous session)?
        fixations = []
        for fpogid, fpogs in fx_groups:
            fpogs_valid = fpogs[fpogs['FPOGV'] == 1]
            if len(fpogs_valid) == 0: 
                continue

            fxh = fpogs_valid['FPOGX'].mean()
            fxv = fpogs_valid['FPOGY'].mean()
            fxdur = fpogs_valid['FPOGD'].values[-1]
            fxtime = fpogs_valid['TIME'].values[0] - session_start_time

            # Checks
            if th_fxdur is not None and fxdur * 1000 <= th_fxdur: 
                # Fixation duration requirement not meet.
                continue
            if 0 <= fxh <= 1 and 0 <= fxv <= 1:
                # Only take fixation that is inside the screen.
                fixations.append([fxh, fxv, fxdur, fxtime])
        return np.array(fixations)

    def _detect_saccades():
        """
        Calculate saccades based on a list of fixations.
        Saccades that fall outside of a given threshold (if available) are eliminated.

        Returns
        -------
        saccades : array-like of shape (n_saccades, 6)
            Each element in the array is a saccade described in the format
            [scsh, scsv, sceh, scev, scdur, sctime], where:
            - scsh : position of starting POG on horizontal axis.
            - scsv : position of starting POG on vertical axis.
            - sceh : position of ending POG on horizontal axis.
            - scev : position of ending POG on vertical axis.
            - scdur : duration in seconds of saccade.
            - sctime : starting time in seconds since the start of the session. 
        """
        fxend = (fixations[:, 2] + fixations[:, 3]).reshape(-1,1)
        time_diff = (fixations[1:][:, 3] - fxend[:-1, 0]).reshape(-1,1)
        saccades = np.concatenate([
                fixations[:-1][:, :2],  # start_x, start_y
                fixations[1:][:, :2],   # end_x, end_y
                time_diff,              # saccades duration
                fxend[:-1]              # saccades start will be end of last fixation
            ], axis=1)
        return saccades
        ### Remove prints, add threshold ... 

    def _detect_blinks():
        """
        Take eye tracking data from outer function and perform detection.
        Blinks that fall outside of a given threshold (if available) are eliminated.

        Returns
        -------
        blinks : array-like of shape (n_fixations, 2)
            Each element in the array is a blink described in the format
            [bkdur, bktime], where:
            - bkdur : duration in seconds of blink.
            - bktime : starting time in seconds since the start of the session. 
        """
        blinks = []
        bk_groups = df[df['BKID'] != 0].groupby(by=['BKID'])
        for bkid, bk in bk_groups:
            bktime = bk['TIME'].values[0] - session_start_time
            # Blink duration is calculated after the blink is over. Hence, the duration
            # is available in the next data entry ...
            next_id = bk.tail(1).index[0]+1
            if next_id >= len(df):
                # Out of bound
                continue
 
            bkdur = df.iloc[next_id]['BKDUR']

            # Checks
            if bkdur == 0:
                # Session is over before blink is over ... eliminate this
                continue
            if th_bkdur is not None and not inside_th(th_bkdur, bkdur):
                continue

            blinks.append([bkdur, bktime])
        blinks = np.array(blinks)
        return blinks

        
    session_start_time = df.iloc[0]['TIME']
    fixations = _detect_fixations()
    saccades = _detect_saccades()
    blinks = _detect_blinks()
    return fixations, saccades, blinks


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
