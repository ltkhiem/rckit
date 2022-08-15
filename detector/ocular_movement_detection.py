import pandas as pd
import numpy as np
from utils.checker import inside_th

def _detect_by_gazepoint_filter(
        df, 
        scrsz, 
        th_fxdur=None,
        th_scdur=None,
        th_bkdur=None
):
    """ 
    Detect fixations, saccades and blinks using the annotation provided 
    by Gazepoint's internal filter.

    Parameters
    ----------
    df : dataframe
        Dataframe of the eye tracking data

    scrsz: tuple or list of shape (2,)
        The monitor size used to collect eye tracking data. The first element 
        is the width and the second element is the height.
    
    th_fxdur: tuple or list of shape (2,) 
        Fixation duration threshold

    th_scdur: tuple or list of shape (2,
        Saccade duration threshold

    th_bkdur: tuple or list of shape (2,)
        Blink duration threshold

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

        # It normally takes 3 invalid samples in order to detect the start of 
        # a new fixation. Hence the duration of the first sample of a fixation 
        # is around 0.02002 to 0.2051 second (larger than the sampling rate which
        # records a sample every 1/150Hz = 0.00667 second), as it includes the 
        # previous 3 invalid samples.
        #
        # As I'm only interested sample marked as valid. The duration of the 
        # fixation will need to recalculate the remove the duration of 3 invalid
        # samples (i.e, not taking the FPOGD directly as the fixation duration).
        
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
            fxdur = fpogs_valid['FPOGD'].values[-1] - \
                    fpogs_valid['FPOGD'].values[0]
            fxtime = fpogs_valid['TIME'].values[0] - session_start_time

            # Checks
            if th_fxdur is not None and fxdur * 1000 <= th_fxdur: 
                # Fixation duration requirement not meet.
                continue
            if 0 <= fxh <= 1 and 0 <= fxv <= 1:
                # Only take fixation that is inside the screen.
                fixations.append(np.array([fxh, fxv, fxdur, fxtime]))
        return np.array(fixations)

    def _detect_saccades():
        """
        Calculate saccades based on a given list of fixations.
        Saccades that fall outside of a given threshold (if available) 
        will be eliminated.

        Returns
        -------
        saccades : array-like of shape (n_saccades, 6)
            Each element in the array is a saccade described in the format
            [scsh, scsv, sceh, scev, scdur, sctime, scmag, scdir], where:
            - scsh : position of starting POG on horizontal axis.
            - scsv : position of starting POG on vertical axis.
            - sceh : position of ending POG on horizontal axis.
            - scev : position of ending POG on vertical axis.
            - scdur : duration in seconds of saccade.
            - sctime : starting time in seconds since the start of the session. 
            - scmag: saccades magnitude in pixels
            - scdir: saccades angle in degree.
        """
        fxend = (fixations[:, 2] + fixations[:, 3]).reshape(-1,1)
        time_diff = (fixations[1:, 3] - fxend[:-1, 0]).reshape(-1,1)
        sxy = fixations[:-1, :2]    
        exy = fixations[1:, :2] 
        sxmag = np.linalg.norm(exy*scrsz - sxy*scrsz, axis=1).reshape(-1,1)
        rad_angles = -np.arctan2(*(exy*scrsz - sxy*scrsz).T[::-1])
        sxdir = ((rad_angles * 180/np.pi + 360) % 360).reshape(-1,1)
        saccades = np.concatenate([
                sxy, 
                exy,
                time_diff,
                fxend[:-1],
                sxmag,
                sxdir,
            ], axis=1)
         
        return saccades

    def _detect_blinks():
        """
        Take eye tracking data from outer function and perform detection.
        Blinks that fall outside of a given threshold (if available) will
        be eliminated.

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
            # Blink duration is calculated after the blink is over. Hence, 
            # the duration is available in the next data entry ...
            next_id = bk.tail(1).index[0]+1 - session_start_id
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
    session_start_id = df.index[0]
    fixations = _detect_fixations()
    cond = len(fixations) > 0
    saccades = _detect_saccades() if cond else np.array([])
    blinks = _detect_blinks() if cond else np.array([])
    return fixations, saccades, blinks


def detect(df, screen_size, method='gazepoint'):
    """ Detect fixations, saccades and blinks from eye-tracking data
    
    Parameters
    ----------

    Returns
    -------
    
    """
    if method=='gazepoint':
        ocular_events = _detect_by_gazepoint_filter(df, screen_size)
    return ocular_events


    
if __name__ == "__main__":
    # df_data = pd.read_csv('/mnt/DATA/ltkhiem/rcir/dataset/0000/fixed_et/tracker_data_log_0.tsv', delimiter = '\t')
    df_data = pd.read_csv('temp/aaaa.csv')
    np.set_printoptions(suppress=True)
    f, s, b = detect(df_data, [1920, 1080]) 
    print("fixations")
    print(f)
    print("saccades")
    print(s)
    
