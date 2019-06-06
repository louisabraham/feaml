import numpy as np
import numba
from numba import njit, prange
from sklearn.ensemble import GradientBoostingClassifier


@njit
def dist(x, y):
    ans = 0.0
    for i in range(len(x)):
        ans += (x[i] - y[i]) ** 2
    return ans


@njit(parallel=True)
def abx(encoded, is_protected, n_sample=10 ** 7):
    ans = numba.int64(0)
    pos = np.nonzero(is_protected)[0]
    neg = np.nonzero(~is_protected)[0]
    for _ in prange(n_sample):
        a = encoded[np.random.choice(pos)]
        b = encoded[np.random.choice(neg)]
        x = encoded[np.random.choice(pos)]
        ans += numba.int8(dist(a, x) < dist(b, x))
    return ans / n_sample - 0.5


def adv_generalization(encoded, is_protected, model, p=0.1):
    train = np.random.rand(len(encoded)) < p
    test = ~train
    model.fit(encoded[train], is_protected[train])
    return model.score(encoded[test], is_protected[test])


def demographic_parity(pred, y, is_protected):
    return abs(pred[is_protected].mean() - pred[~is_protected].mean())


def equalized_odds(pred, y, is_protected):
    y = y > 0
    return abs(pred[is_protected & y].mean() - pred[~is_protected & y].mean()) + abs(
        pred[is_protected & ~y].mean() - pred[~is_protected & ~y].mean()
    )


def equal_opportunity(pred, y, is_protected):
    y = y > 0
    return abs(pred[is_protected & y].mean() - pred[~is_protected & y].mean())
