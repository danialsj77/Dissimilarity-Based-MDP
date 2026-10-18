"""Gaussian-process regression of the scalar example: analytic gradient, recovery of a smooth function, chunking."""

import numpy as np

from dmdp_toy.gp import GP

LO, HI = np.zeros(4), np.ones(4)


def _data(n, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 4))
    y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] - X[:, 3] ** 2
    return X, y + noise * rng.standard_normal(n)


def test_nll_gradient_matches_finite_differences():
    X, y = _data(30)
    for v in (None, np.linspace(1e-4, 1e-2, 30)):
        gp = GP(X, y, LO, HI, noise_var=v, theta=np.r_[np.log([0.4, 0.7, 1.1, 0.9]), 0.2, np.log(0.05)])
        th = np.r_[np.log([0.5, 0.8, 1.3, 0.6]), 0.1, np.log(0.03)]
        _, g = gp.nll(th)
        h = 1e-6
        fd = np.array([(gp.nll(th + h * e)[0] - gp.nll(th - h * e)[0]) / (2 * h) for e in np.eye(len(th))])
        assert np.max(np.abs(fd - g) / np.maximum(np.abs(g), 1e-3)) < 1e-4


def test_recovers_smooth_function():
    X, y = _data(200)
    gp = GP(X, y, LO, HI)
    Xt, yt = _data(500, seed=1, noise=0.0)
    assert np.sqrt(np.mean((gp.mean(Xt) - yt) ** 2)) < 0.05 * yt.std()


def test_chunked_prediction_and_state_round_trip():
    X, y = _data(60)
    gp = GP(X, y, LO, HI)
    Xt, _ = _data(1000, seed=2)
    assert np.allclose(gp.mean(Xt, chunk=7), gp.mean(Xt, chunk=10_000), rtol=0, atol=1e-12)
    assert np.allclose(GP.from_state(gp.state("p_"), "p_").mean(Xt), gp.mean(Xt), atol=1e-12)
