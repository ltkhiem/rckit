""" 
This is a reproduction of the ocular movement detection using wavelet transform
algorithm that was initially introduced in the paper:

    Bulling, A., Ward, J. A., Gellersen, H., & Tröster, G. (2011). Eye movement
    analysis for activity recognition using electrooculography. IEEE 
    transactions on pattern analysis and machine intelligence, 33(4), 741–753. 
    https://doi.org/10.1109/TPAMI.2010.86
"""

import numpy as np
import pywt
from rckit.detector.constants import OcularEventMask


def _detect_by_wavelet_transform(
        signals,
        mask_only=True,
        fs = 1000,
        th_sm=0.1, 
        th_st_lower=20, 
        th_st_upper=200, 
        th_bm=0.3, 
        th_bt=390, 
        th_ft=200, 
        th_fd=1
):
    """
    Parameters
    ----------
    signals : array-like, shape (2, n_samples)
        The signals in two channels. The first channel is the vertical channel
        and the second channel is the horizontal channel.

    mask_only : bool, default True
        If True, only the ocular event mask will be returned. If False, the
        ocular events (fixations, saccades, blinks) will be returned along 
        with the mask.

    fs : int, default 1000
        The sampling frequency of the signals.

    th_sm : float, default 0.1
        The magnitude threshold for saccades on the wavelet transformed signal.

    th_st_lower : int, default 20
        The lower bound of the duration threshold for saccades

    th_st_upper : int, default 200
        The upper bound of the duration threshold for saccades

    th_bm : float, default 0.3
        The magnitude threshold for blinks on the wavelet transformed signal.

    th_bt : int, default 390
        The duration threshold for blinks

    th_ft : int, default 200
        The duration threshold for fixations

    th_fd : int, default 1
        The dipersion threshold for fixations
        
    Returns
    -------
    fixations : array-like, shape (n_fixations, 4), optional
        [Only returned when mask_only is False]
        List of fixations, where each fixation is represented by a magnitude 
        on each channel, duration and starting time.

    saccades : array-like, shape (n_saccades, 8), optional
        [Only returned when mask_only is False]
        List of saccades, where each saccade is represented by a magnitude on
        each channel (for both starting and ending point), duration, starting
        time, movement magnitude and movement direction.

    blinks : array-like, shape (n_blinks, 2), optional
        [Only returned when mask_only is False]
        List of blinks, where each blink is represented by a blink duration 
        and starting time.

    mask : array-like, shape (n_samples,)

    """

    def _get_segments(condition, split=True):
        d = np.diff(condition)
        idx, = d.nonzero()

        if condition[0]:
            # If the start of condition is True prepend a 0
            idx = np.r_[0, idx]

        if condition[-1]:
            # If the end of condition is True, append the length of the array - 1
            idx = np.r_[idx, condition.size-1] # Edit
        
        if split:
            return np.array(list(zip(*idx.reshape(-1,2))))
        else:
            return idx.reshape(-1,2)

    def _gen_mask(events, mask_value):
        for e in events:
            ocular_mask[e[0]:e[1]+1] = mask_value 
        return ocular_mask

    def _detect_blinks():
        """
        Take vertical eye movement signal to detect blinks.
        Blinks are taken as the wavelet transform of the signal being higher than
        a predefined magnitude threshold (th_bm). 

        Returns
        -------
        blinks : array-like of shape (n_blinks, 2)
            Each row is a blink event in the format [bkdur, bktime], where:
            - bkdur: duration of the blink in seconds
            - bktime: starting time of the 
        """
        wl, freq = pywt.cwt(veog, wavelet='haar', scales=[20])
        wl = wl.flatten()
        condition_b = np.abs(wl) >= th_bm

        # Obtain positions of segments that might contain a blink
        # which exceeds the blink magnitude threshold.
        pstarts, pends = _get_segments(condition_b)
        sign = np.abs(wl[pstarts]) / wl[pstarts]

        # Only the peaks that are positive are considered
        pstarts = np.r_[pstarts, 10000000]
        positive_peaks, = np.where(sign>0)

        # Only picks blinks that fall into the given threshold
        gap_durations = pstarts[positive_peaks+1] - pends[positive_peaks] + 1
        picks = positive_peaks[np.where(gap_durations <= th_bt)[0]]

        # Gather the actual on-set and off-set of the blinks
        # by gradually expanding the segments to capture the whole peak. 

        # Note that the pstarts and pends are just parts of the peak which
        # exceeded the threshold, not the peak itself.
        b_starts = []
        for x in pstarts[picks]:
            while x>0 and th_sm <= np.abs(wl[x]): x-=1;
            b_starts.append(x)

        b_ends = []
        for x in pends[picks+1]:
            while x<wl.size and th_sm <= np.abs(wl[x]): x+=1;
            b_ends.append(x)

        # Calculate the duration and starting time of the blinks
        blinks = np.array(list(map(lambda x,y: ((y-x+1)/1000, x/1000), b_starts, b_ends)))

        # Update ocular mask
        blinks_pos = np.vstack([b_starts, b_ends]).T
        _gen_mask(blinks_pos, OcularEventMask.Blink)

        return blinks


    def _detect_saccades():
        """
        Take eye movement signal on each channel to detect saccades.

        Returns
        -------
        saccades : array-like of shape (n_saccades, 8)
            Each row is a saccade event in the format
            [scsh, scsv, sceh, scev, scdur, sctime, scmag, scdir], where:
            - scsh: starting magnitude on horizontal channel
            - scsv: starting magnitude on vertical channel
            - sceh: ending magnitude on horizontal channel
            - scev: ending magnitude on vertical channel
            - scdur: duration of the saccade in seconds
            - sctime: starting time of the saccade in seconds
            - scmag: magnitude of the saccade movement
            - scdir: direction of the saccade movement in degrees
        """
        for mono_eog in signals:
            wl, freq = pywt.cwt(mono_eog, wavelet='haar', scales=[20])
            wl = wl.flatten()
            condition = (np.abs(wl) >= th_sm) & (ocular_mask == 0)

            # Obtain positions of segments that might contain a saccade
            pstarts, pends = _get_segments(condition) 
            durations = pends - pstarts + 1
            picks, = np.where((durations >= th_st_lower) & (durations <= th_st_upper))
            saccades_pos = np.vstack([pstarts[picks], pends[picks]]).T 

            _gen_mask(saccades_pos, OcularEventMask.Saccade)

        condition = ocular_mask == OcularEventMask.Saccade
        saccades_pos = _get_segments(condition, split=False)
        
        time_diff = (saccades_pos[:,1] - saccades_pos[:,0] + 1).reshape(-1,1) / fs
        sxy = np.vstack([heog[saccades_pos[:,0]], veog[saccades_pos[:,0]]]).T
        exy = np.vstack([heog[saccades_pos[:,1]], veog[saccades_pos[:,1]]]).T
        scmag = np.linalg.norm(exy - sxy, axis=1).reshape(-1,1)
        rad_angles = -np.arctan2(*(exy - sxy).T[::-1])
        scdir = ((rad_angles * 180/np.pi + 360) % 360).reshape(-1,1)
        saccades = np.concatenate([
                sxy, 
                exy,
                time_diff,
                saccades_pos[:,0].reshape(-1,1)/fs,
                scmag,
                scdir,
            ], axis=1)
        return saccades


    def _detect_fixations():
        """
        Take the ocular mask after blinks and saccades are detected to detect fixations.
        It was found that each nonsaccadic segment contains a fixation, so if any segment
        falls into the given threshold, it is considered a fixation.

        Returns
        -------
        fixations : array-like of shape (n_fixations, 4)
            Each row is a fixation event in the format 
            [fxh, fxv, fxdur, fxtime], where:
            - fxh: horizontal magnitude of the fixation
            - fxv: vertical magnitude of the fixation
            - fxdur: duration of the fixation in seconds
            - fxtime: starting time of the fixation in seconds

        """
        condition = ocular_mask == OcularEventMask.NoEvent
        pstarts, pends = _get_segments(condition)
        picks = np.where(pends-pstarts >= th_ft)
        fxstarts, fxdur = [], []
        fxh, fxv = [], []

        def _found_fixation(start, end):
            _gen_mask(np.array([[start, end]]), OcularEventMask.Fixation)
            fxstarts.append(start)
            fxdur.append(end-start+1)
            fxh.append(np.mean(heog[start:end+1]))
            fxv.append(np.mean(veog[start:end+1]))

        for s, e in zip(pstarts[picks], pends[picks]):
            fs = s 
            # Get a window with intial length equals to the fixation threshold.
            fe = fs + th_ft 
            found = False
            while fe <= e:
                dispersion = np.max(veog[fs:fe]) - np.min(veog[fs:fe]) + \
                             np.max(heog[fs:fe]) - np.min(heog[fs:fe])
                if dispersion > th_fd:
                    if found:
                        _found_fixation(fs, fe-1)
                        break
                    else:
                        fs += 1
                        fe = fs + th_ft
                else:
                    if found and fe == e:
                        _found_fixation(fs, fe)
                        break
                    elif not found: 
                        found = True
                fe += 1
        fixations = np.concatenate([
            np.array(fxh).reshape(-1,1),
            np.array(fxv).reshape(-1,1),
            np.array(fxdur).reshape(-1,1)/fs,
            np.array(fxstarts).reshape(-1,1)/fs,
        ], axis=1)
        return fixations

    veog, heog = signals 
    ocular_mask = np.zeros(veog.size)
    blinks = _detect_blinks()
    saccades = _detect_saccades()
    fixations = _detect_fixations()

    if not mask_only:
        # saccades = _get_segments(ocular_mask==OcularEventMask.Saccade, split=False)
        return fixations, saccades, blinks, ocular_mask
    return ocular_mask

#     def extract( signals, fs=1000):
#         veog, heog = signals
#         blinks, saccades, fixations, mask = detect(signals, mask_only=False)

#         signal_length = signals[0].size
#         time_reading = signal_length / fs 

#         num_fixa = fixations.shape[0]
#         num_sacc = saccades.shape[0]
#         num_blink = blinks.shape[0]

#         rate_fixa = num_fixa / time_reading
#         rate_sacc = num_sacc / time_reading
#         rate_blink = num_blink / time_reading

#         fixa_dura_norm = (fixations[:, 1] - fixations[:, 0]) / signal_length
#         sacc_dura_norm = (saccades[:, 1] - saccades[:, 0]) / signal_length
#         blink_dura_norm = (blinks[:, 1] - blinks[:, 0]) / signal_length

#         direction_v, direction_h = [], []
#         for idx in range(1, num_fixa):
#             fs, fe = fixations[idx]
#             pfs, pfe = fixations[idx-1]
#             direction_v.append(np.mean(veog[fs:fe]) - np.mean(veog[pfs:pfe]))
#             direction_h.append(np.mean(heog[fs:fe]) - np.mean(heog[pfs:pfe]))
#         direction_v = np.array(direction_v)
#         direction_h = np.array(direction_h)

#         rate_v_bwd = np.where(direction_v < 0)[0].size / num_fixa
#         rate_h_bwd = np.where(direction_h < 0)[0].size / num_fixa

#         scalar = [num_fixa, rate_fixa, num_sacc, rate_sacc, num_blink, rate_blink, rate_v_bwd, rate_h_bwd]

#         return  {
#             'scalar_data': scalar,
#             'fixa_dura_norm': fixa_dura_norm,
#             'sacc_dura_norm': sacc_dura_norm,
#             'blink_dura_norm': blink_dura_norm
#         }

     


