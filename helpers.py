from functools import wraps

from numba import njit, prange
import numpy as np


@njit
def distances(X, v, alpha, k):
    N, P = X.shape
    dists = np.zeros((N, k))
    for i in prange(N):
        for p in range(P):
            for j in range(k):
                dists[i, j] += alpha[p] * (X[i, p] - v[j, p]) ** 2
    return dists


@njit
def softmax(x):
    """softmax function with the max trick
    optimization is probably not worth it
    """
    x = x.copy()
    x -= x.max()  # doesn't change result
    return np.exp(x) / x.sum()


@njit
def M_nk(X, v, alpha, k):
    return softmax(-distances(X, v, alpha, k))


@njit
def mean_axis0(X):
    """average of an array of vectors
    assumes N >> k for speed
    equivalent to np.mean(axis=0)
    """
    N, k = X.shape
    ans = np.zeros(k)
    for i in prange(N):
        for j in range(k):
            ans[j] += X[i, j]
    return ans / N


@njit
def squared_error_2d(x, y):
    """fast squared error when N >> P
    """
    N, P = x.shape
    ans = 0.0
    for i in prange(N):
        for j in range(P):
            ans += (x[i, j] - y[i, j]) ** 2
    return ans


@njit
def cross_entropy(y_true, y_pred):
    N = len(y_true)
    ans = 0.0
    eps = np.finfo(y_pred.dtype).eps
    for i in prange(N):
        if y_true[i]:
            ans -= np.log(np.maximum(y_pred[i], eps))
        else:
            ans -= np.log(np.maximum(1 - y_pred[i], eps))
    return ans

def make_bounds(P, k):
    # only the w_k are constrained between 0 and 1
    return [(None, None)] * (2 * P) + [(0, 1)] * k + [(None, None)] * (k * P)


@njit
def unpack_params(params, P, k):
    return (
        params[:P],
        params[P : 2 * P],
        params[2 * P : (2 * P) + k],
        params[(2 * P) + k :].reshape((k, P)),
    )


@njit
def LFR_compute(params, X, is_protected, k):
    N, P = X.shape

    # alpha: weights of the distance function
    # w: classification probabilities
    # v: prototypes
    alpha_p, alpha_np, w, v = unpack_params(params, P, k)

    # probabilistic mapping
    mapping = np.empty((N, k))
    mapping[is_protected] = M_nk(X[is_protected], v, alpha_p, k)
    mapping[~is_protected] = M_nk(X[~is_protected], v, alpha_np, k)

    # reconstructed values
    reconstructed = np.dot(mapping, v)

    # predictions
    pred = np.dot(mapping, w)

    return (mapping, reconstructed, pred)


@njit
def LFR_optim_obj(params, X, is_protected, y, k=10, A_x=0.01, A_y=0.1, A_z=0.5):

    mapping, reconstructed, pred = LFR_compute(params, X, is_protected, k)

    # reconstruction error
    L_x = squared_error_2d(X, reconstructed)

    # prediction error
    L_y = cross_entropy(y, pred)

    # unfairness: absolute error between average mappings
    L_z = np.sum(
        np.abs(mean_axis0(mapping[is_protected]) - mean_axis0(mapping[~is_protected]))
    )

    # total loss
    return A_x * L_x + A_y * L_y + A_z * L_z


def with_logging(print_inteval=250):
    def decorator(f):
        @wraps(f)
        def g(*args, **kwargs):
            g.iters += 1
            ans = f(*args, **kwargs)
            if g.iters % print_inteval == 0:
                print(f"Iter {g.iters}, value {ans}")
            return ans

        g.iters = 0
        return g

    return decorator

