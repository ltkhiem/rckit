import numpy as np
"""
This is for generating global features of eye movement
"""

def get_fixations_duration(fx):
    return fx[:, 3]


def get_saccades_duration(sc):
    return sc[:, 4]


def get_saccades_direction(sc):
    return sc[:, 7]


def get_blinks_duration(bk):
    return bk[:, 0]


def get_movement_magnitudes(sc, screen_size=[1920, 1080]):
    """
        
    Returns
    -------
    dist: array-like of shape (n_saccades, 1)
        L2 distances of all saccades

    dist_h: array-like of shape (n_saccades, 1)
        L1 distances of all saccades on the horizontal axis

    dist_v: array-like of shape (n_saccades, 1)
        L1 distances of all saccades on the vertical axis

    """
    dist = sc[:, 6]
    dist_h, dist_v = (np.abs(sc[:, 2:4] - sc[:, 0:2]) * screen_size).T
    return dist, dist_h, dist_v 

     

def get_regressions(sc):
    """

    Returns
    -------
    regr: array-like of shape (n_regressions, 1)
        Indexes of the saccades which are regressive movements.

    nregr: int
        Number of regressions.
    """
    regr = np.where((90<sc[:, 7]) & (sc[:, 7]<270))[0]
    nregr = regr.shape[0]
    return regr, nregr


def _get_velocity(dist, dist_h, dist_v, scdur):
    velo = dist / scdur
    velo_h, velo_v = dist_h/scdur, dist_v/scdur
    return velo, velo_h, velo_v


def get_velocity(sc):
    """
    Calculate the velocity of each saccade

    Returns
    -------
    velo: array-like of shape (n_saccades, 1)
        Velocities of the saccades

    velo_h: array-like of shape (n_saccades, 1)
        Velocities of the saccades on the horizontal axis
    
    velo_v: array-like of shape (n_saccades, 1)
        Velocities of the saccades on the vertical axis
    """
    dist, dist_h, dist_v = get_movement_magnitudes(sc)
    scdur = get_saccades_duration(sc)
    return _get_velocity(dist, dist_h, dist_v, scdur)


def _handle_null(fx, sc, bk):
    if len(fx)==0:
        fx = np.array([[0,0,1,0]])
    if len(sc)==0:
        sc = np.array([[0,0,0,0,1,0,0,0]])
    if len(bk)==0:
        bk = np.array([[0,0]])
    return fx, sc, bk


def generate(fx, sc, bk, norm_value:float=None):
    nfx, nbk = fx.shape[0], bk.shape[0]
    fx, sc, bk = _handle_null(fx, sc, bk)
    regr, nregr = get_regressions(sc)
    dist, dist_h, dist_v = get_movement_magnitudes(sc)
    velo, velo_h, velo_v = _get_velocity(dist, dist_h, 
                                dist_v, sc[:, 4])
    feats = {
        "nfx": nfx,
        "nbk": nbk,
        "fxdur": fx[:, 2],
        "scdur": sc[:, 4],
        "bkdur": bk[:, 0],
        "scdir": sc[:, 7],
        "dist": dist,
        "dist_h": dist_h,
        "dist_v": dist_v,
        "velo": velo, 
        "velo_h": velo_h,
        "velo_v": velo_v,
        "regr_id": regr,
        "nregr": nregr,
        "regr_rate": nregr/fx.shape[0],
    }

    if norm_value is not None:
        feats.update({
            'nfx_norm': nfx/norm_value,
            'nbk_norm': nbk/norm_value,
            'nregr_norm': nregr/norm_value,
        })
            
    return feats 


if __name__ == "__main__":
    fx = np.array([[0.117   , 0.251  , 0.201   , 0.        ],
       [0.276, 0.239, 0.187    , 0.074   ],
       [0.182, 0.203, 0.193   , 0.267   ],
       [0.150, 0.224, 0.259   , 0.468   ]])

    print(get_distance_feats(fx))





