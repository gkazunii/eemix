import numpy as np

def mode_product(x, m, mode):
    x = np.asarray(x)
    m = np.asarray(m)
    if mode <= 0 or mode % 1 != 0:
        raise ValueError('`mode` must be a positive interger')
    if x.ndim < mode:
        raise ValueError('Invalid shape of X for mode = {}: {}'.format(mode, x.shape))
    if m.ndim != 2:
        raise ValueError('Invalid shape of M: {}'.format(m.shape))
    return np.swapaxes(np.swapaxes(x, mode - 1, -1).dot(m.T), mode - 1, -1)
