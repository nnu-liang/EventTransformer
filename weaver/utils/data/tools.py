import numpy as np
import math

import awkward as ak


def _hash(*args):
    return np.array([x.__hash__() for x in zip(*args)])


def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return ak.concatenate(arrays, axis=axis)


def _stack(arrays, axis=1):
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.stack(arrays, axis=axis)
    else:
        s = [slice(None)] * (arrays[0].ndim + 1)
        s[axis] = np.newaxis
        s = tuple(s)
        return ak.concatenate([a.__getitem__(s) for a in arrays], axis=axis)


def _pad(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x


def _repeat_pad(a, maxlen, shuffle=False, dtype='float32'):
    x = ak.to_numpy(ak.flatten(a))
    x = np.tile(x, int(np.ceil(len(a) * maxlen / len(x))))
    if shuffle:
        np.random.shuffle(x)
    x = x[:len(a) * maxlen].reshape((len(a), maxlen))
    mask = _pad(ak.zeros_like(a), maxlen, value=1)
    x = _pad(a, maxlen) + mask * x
    return ak.values_astype(x, dtype)


def _pad_4d(a, maxlen, value=0, dtype='float32'):
    """Pad a 4D tensor (batch, features, p, p) to (batch, features, maxlen, maxlen), supporting Awkward and NumPy."""

    if isinstance(a, ak.Array):
        a = ak.pad_none(a, maxlen, axis=1, clip=True)
        a = ak.pad_none(a, maxlen, axis=2, clip=True)
        a = ak.fill_none(a, value)
        a = ak.values_astype(a, dtype)
        return ak.to_numpy(a)


def _repeat_pad_4d(a, maxlen, shuffle=False, dtype='float32'):
    """Repeat pad a 4D tensor to a specified maximum length.
    
    Args:
        a: Input tensor of shape (batch_size, features, particles, particles)
        maxlen: Target length for the particle dimensions
        shuffle: Whether to shuffle the repeated elements
        dtype: Output dtype
    """
    if isinstance(a, np.ndarray):
        if a.ndim == 4 and a.shape[2] == maxlen and a.shape[3] == maxlen:
            return a
            
        # Flatten the particle dimensions
        flat = a.reshape(a.shape[0], a.shape[1], -1)
        
        # Calculate how many times to repeat
        repeat_times = int(np.ceil(maxlen * maxlen / flat.shape[2]))
        x = np.tile(flat, (1, 1, repeat_times))
        
        # Reshape back to 4D
        x = x[:, :, :maxlen * maxlen].reshape(a.shape[0], a.shape[1], maxlen, maxlen)
        
        # Create mask for original data
        mask = np.zeros((a.shape[0], a.shape[1], maxlen, maxlen), dtype=bool)
        mask[:, :, :a.shape[2], :a.shape[2]] = True
        
        # Combine original and repeated data
        x_orig = _pad_4d(a, maxlen, value=0, dtype=dtype)
        x = x_orig + (~mask) * x
        
        return x.astype(dtype)
    else:
        raise ValueError("Input must be a numpy array")


def auto_pad_with_mode(a, maxlen, params):
    """Pad a 4D tensor using the specified mode.

    Args:
        a: Input tensor, either numpy.ndarray or awkward.Array (converted beforehand)
        maxlen: Target length for P and P
        params: Dictionary with keys:
            - 'pad_mode': 'wrap' or 'constant'
            - 'pad_value': used if pad_mode == 'constant'
    """
    pad_mode = params['pad_mode']

    if pad_mode == 'wrap':
        return _repeat_pad_4d(a, maxlen)
    elif pad_mode == 'constant':
        return _pad_4d(a, maxlen, value=params['pad_value'])
    else:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}")


def _clip(a, a_min, a_max):
    if isinstance(a, np.ndarray) or a.ndim == 1:
        return np.clip(a, a_min, a_max)
    else:
        return ak.unflatten(np.clip(ak.to_numpy(ak.flatten(a)), a_min, a_max), ak.num(a))


def _knn(support, query, k, n_jobs=1):
    from scipy.spatial import cKDTree
    kdtree = cKDTree(support)
    d, idx = kdtree.query(query, k, n_jobs=n_jobs)
    return idx


def _batch_knn(supports, queries, k, maxlen_s, maxlen_q=None, n_jobs=1):
    assert (len(supports) == len(queries))
    if maxlen_q is None:
        maxlen_q = maxlen_s
    batch_knn_idx = np.ones((len(supports), maxlen_q, k), dtype='int32') * (maxlen_s - 1)
    for i, (s, q) in enumerate(zip(supports, queries)):
        batch_knn_idx[i, :len(q[:maxlen_q]), :] = _knn(
            s[:maxlen_s], q[:maxlen_q], k, n_jobs=n_jobs).reshape((-1, k))  # (len(q), k)
    return batch_knn_idx


def _batch_permute_indices(array):
    random_array = ak.unflatten(np.random.rand(ak.count(array)), ak.num(array))
    return ak.argsort(random_array)


def _batch_argsort(array):
    return ak.argsort(array)


def _batch_gather(array, indices):
    return array[indices]


def _p4_from_pxpypze(px, py, pz, energy):
    import vector
    vector.register_awkward()
    return vector.zip({'px': px, 'py': py, 'pz': pz, 'energy': energy})


def _p4_from_ptetaphie(pt, eta, phi, energy):
    import vector
    vector.register_awkward()
    return vector.zip({'pt': pt, 'eta': eta, 'phi': phi, 'energy': energy})


def _p4_from_ptetaphim(pt, eta, phi, mass):
    import vector
    vector.register_awkward()
    return vector.zip({'pt': pt, 'eta': eta, 'phi': phi, 'mass': mass})


def _get_variable_names(expr, exclude=['awkward', 'ak', 'np', 'numpy', 'math', 'len']):
    import ast
    root = ast.parse(expr)
    return sorted({node.id for node in ast.walk(root) if isinstance(
        node, ast.Name) and not node.id.startswith('_')} - set(exclude))


def _eval_expr(expr, table):
    tmp = {k: table[k] for k in _get_variable_names(expr)}
    tmp.update({'math': math, 'np': np, 'numpy': np, 'ak': ak, 'awkward': ak, 'len': len, '_hash': _hash,
                '_concat': _concat, '_stack': _stack, '_pad': _pad, '_repeat_pad': _repeat_pad, '_clip': _clip,
                '_batch_knn': _batch_knn, '_batch_permute_indices': _batch_permute_indices,
                '_batch_argsort': _batch_argsort, '_batch_gather': _batch_gather, '_p4_from_pxpypze': _p4_from_pxpypze,
                '_p4_from_ptetaphie': _p4_from_ptetaphie, '_p4_from_ptetaphim': _p4_from_ptetaphim})
    return eval(expr, tmp)
