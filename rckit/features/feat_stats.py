import numpy as np 
import scipy.stats

bof_names = ['tr_max', 'tr_min', 'std', 'mean', 'argmin', 'argmax', 'tr_range', 'iqr', 'kurtosis', 'skewness']
hist_names = [f'bin_{idx}' for idx in range(15+1)]

def get_BOF(a):
    if a.size == 0 or np.isnan(a).any(): a = np.array([0])
    aa = np.sort(a)
    tmax = aa[len(a)*90//100]
    tmin = aa[len(a)*10//100]
    stats = [
        tmax, tmin,  # Trimmed Max, Trimmed Min
        np.std(a), np.average(a),  # Std, Mean
        np.where(a==tmin)[0][0], np.where(a==tmax)[0][0],  # Argmin, Argmax
        tmax-tmin, # Trimmed_Max - Trimmed_Min
        np.subtract(*np.percentile(a, [75, 25])),  # IQR=Q3-Q1
        scipy.stats.kurtosis(a), 
        scipy.stats.skew(a)
    ]
    feats = dict(zip(bof_names, stats)) 
    return feats


def get_HIST(a, bins=10):
    if a.size == 0 or np.isnan(a).any(): a = np.array([0])
    res = np.histogram(a, bins=bins)[0]
    feats = dict(zip(hist_names, res))
    return feats


def get_stats(data, extractor='all', feat_bins=None):
    ret = {}
    if np.isinf(data).any():
        data = np.nan_to_num(data, posinf=0, copy=True)

    if extractor == 'all':
        assert feat_bins is not None, "Feature bins must be specified"
        ret.update(get_BOF(data))
        ret.update(get_HIST(data, bins=feat_bins))
    elif extractor  == 'bof':
        ret.update(get_BOF(data))
    elif extractor == 'hist':
        assert feat_bins is not None, "Feature bins must be specified"
        ret.update(get_HIST(data, bins=feat_bins))

    return ret


def _get_histbins(feats, subjects, nbins):
    feat_all_data = {}

    # Aggregate features
    for sid, sbj in enumerate(subjects):
        for item in feats[sbj]:
            for k, v in item.items():
                if type(v)==list or type(v)==np.ndarray:
                    if k not in feat_all_data:
                        feat_all_data[k] = []
                    if type(v) == np.ndarray:
                        feat_all_data[k] += v.tolist()
                    else:
                        feat_all_data[k] += v
    feat_all_bins = {}

    for k, v in feat_all_data.items():
        if np.isinf(v).any():
            v = np.nan_to_num(v, posinf=0, copy=True)
        # Remove outliers to generate an accurate histogram
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        lower_fence, upper_fence = q1 - 1.5*iqr, q3 + 1.5*iqr

        # Filter outliers
        keep_v, = np.where((v>lower_fence) & (v < upper_fence))
        filter_v = np.array(v)[keep_v]

        # Generate histogram
        _, feat_all_bins[k] = np.histogram(filter_v, bins=nbins)
    return feat_all_bins


def generate(feats, extractor='all', feat_bins=None, ignore_feats = None):
    """
    Generate simple statistics for distribution-like features (list of floats),
    while keeping the single-number features as it is.

    Parameters
    ----------
    feats: dict
        Dictionary of features, dict value can either be a float or a 
        list of floats.

    extractor: str
        Select which extractor will be used. Can be either 'all', 'bof' or 'hist',
        Default is 'all'
    """

    
    if extractor=='all' or extractor=='hist':
        assert feat_bins is not None, "'feat_bins' must be specified"

    _chk_ft = ignore_feats is not None

    new_feats = {}
    for k, v in feats.items():
        if _chk_ft and k in ignore_feats:
            continue
        if type(v) == list or type(v) == np.ndarray:
            if type(v) == list:
                v = np.array(v)
            if extractor=='bof':
                stats_feats = get_stats(v, extractor='bof')
            else:
                stats_feats = get_stats(v, extractor, feat_bins[k])
            new_feats.update({f'{k}_{_k}': _v for _k, _v in stats_feats.items()})
        else:
            new_feats.update({k:v})
    return new_feats


def bulk_generate(feats, 
                  extractor='all', 
                  dataset='ind', 
                  subjects=None, 
                  nbins = 16, 
                  hist_bins=None,
                  ignore_feats=None,
                  return_bins=False):
    """
    Buld generate statistics feature for all trails (of an individual subject or 
    an entire dataset).

    Parameters
    ----------
    feats: list or dict,
        Features generated by rckit.features.feat_global.

    extractor: str
        Select which extractor will be used. Can be either 'all', 'bof' or 'hist',
        Default is 'all'
    
    dataset: str
        Either 'ind' (for individual subject) or 'all' (for the entire dataset)

    subjects: list of str
        List of subject ids corresponding with feats.keys(). Must be specified if
        dataset type is set to 'all'

    nbins: int
        Number of histogram bins to extract. Specify is extractor is set to 'all' 
        or 'hist'. Default is set to 15

    hist_bins: dict
        Dictionary of histogram bins. Must be specified if extractor is set to

    ignore_feats: list of str
        List of features to ignore. Default is None

    return_bins: bool
        If set to True, the histogram bins will be returned. Default is False

    Returns
    -------
    new_feats: dict
        Dictionary of features with statistics extracted

    feat_bins: dict
        Dictionary of histogram bins. Only returned if return_bins is set to True
    """
    if dataset == 'ind': 
        assert type(feats) == list, "'feats' must have type list"
        subjects = ['1234']
        feats = {'1234': feats}
    elif dataset == 'all':
        assert type(feats) == dict, "'feats' must have type dict'"
        assert subjects is not None, "List of 'subjects' must be specified"

    global hist_names
    hist_names = [f'bin_{idx}' for idx in range(nbins+1)]

    if extractor=='all' or extractor=='hist':
        if hist_bins is not None:
            feat_bins = hist_bins
        else:
            feat_bins = _get_histbins(feats, subjects, nbins)
    else:
        feat_bins = None

    new_feats = {}
    
    all_stat_feats = {}
    for sid, sbj in enumerate(subjects):
        stat_feats = []
        for item in feats[sbj]:
            stat_feats.append(generate(item, extractor, feat_bins, ignore_feats))
        all_stat_feats[sbj] = stat_feats

    if dataset == 'ind':
        new_feats = all_stat_feats['1234']
    elif dataset == 'all':
        new_feats = all_stat_feats

    if return_bins:
        return new_feats, feat_bins
    else:
        return new_feats

