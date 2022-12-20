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


    Returns
    -------

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

    def _gen_mask( events, ocular_mask, mask_value):
        for e in events:
            ocular_mask[e[0]:e[1]+1] = mask_value 
        return ocular_mask

    def _detect_blinks( veog, ocular_mask):
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
        gap_durations = pstarts[positive_peaks+1] - pends[positive_peaks]
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

        blinks = np.vstack([b_starts, b_ends]).T

        new_mask = _gen_mask(blinks, ocular_mask, OcularEventMask.Blink)
        return blinks, new_mask

    def _detect_saccades( mono_eog, ocular_mask):
        wl, freq = pywt.cwt(mono_eog, wavelet='haar', scales=[20])
        wl = wl.flatten()
        condition = (np.abs(wl) >= th_sm) & (ocular_mask == 0)

        # Obtain positions of segments that might contain a saccade
        pstarts, pends = _get_segments(condition) 
        durations = pends - pstarts
        picks, = np.where((durations >= th_st_lower) & (durations <= th_st_upper))
        saccades = np.vstack([pstarts[picks], pends[picks]]).T 

        new_mask = _gen_mask(saccades, ocular_mask, OcularEventMask.Saccade)
        return saccades, new_mask


    def _detect_fixations( heog, veog, ocular_mask):
        # It was proved that each nonsaccadic segment contains a fixation.
        new_mask = ocular_mask
        condition = ocular_mask == 0
        pstarts, pends = _get_segments(condition)
        picks = np.where(pends-pstarts >= th_ft)
        f_starts, f_ends = [], []

        for s, e in zip(pstarts[picks], pends[picks]):
            fs = s 
            # Get a window with intial length equals to the fixation threshold.
            fe = fs + th_ft 
            found = False
            while fe <= e:
                dispersion = np.max(veog[fs:fe]) - np.min(veog[fs:fe]) + np.max(heog[fs:fe]) - np.min(heog[fs:fe])
                if dispersion > th_fd:
                    if found:
                        new_mask[fs:fe] = OcularEventMask.Fixation
                        f_starts.append(fs)
                        f_ends.append(fe)
                        break
                    else:
                        fs += 1
                        fe = fs + th_ft
                else:
                    if found and fe == e:
                        new_mask[fs:fe] = OcularEventMask.Fixation
                        f_starts.append(fs)
                        f_ends.append(fe)
                        break
                    elif not found: 
                        found = True
                fe += 1
        fixations = np.vstack([f_starts, f_ends]).T
        return fixations, new_mask


    veog, heog = signals 
    ocular_mask = np.zeros(veog.size)
    blinks, ocular_mask = _detect_blinks(veog, ocular_mask)
    saccades_v, ocular_mask = _detect_saccades(veog, ocular_mask)
    saccades_h, ocular_mask = _detect_saccades(heog, ocular_mask)
    fixations, ocular_mask = _detect_fixations(heog, veog, ocular_mask)

    if not mask_only:
        saccades = _get_segments(ocular_mask==OcularEventMask.Saccade, split=False)
        return blinks, saccades, fixations, ocular_mask
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

     


