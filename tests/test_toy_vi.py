"""Value iteration of the scalar example: convergence, the bounds of Lemma lemma:toy_facts, the closed form on R,
the converged flag, the action grid of the learner (centred at a_hat, never at a_star), the learners without the
true f and a_star, and the refined node actions of the model-free policy."""

import numpy as np
import pytest

from dmdp_toy import modelfree, offline, system as S, vi
from dmdp_toy.certify import test_points as make_test_points
from dmdp_toy.grids import Grid, graded_actions


@pytest.fixture(scope="module")
def medium():
    grid = Grid.graded(0.006, 0.3, 0.12, 0.06)
    actions = graded_actions(S.A_STAR, (0.004, 0.25, 0.04), (0.004, 0.25, 0.1))
    return grid, actions, vi.value_iteration_ratio(grid, actions)


def test_value_iteration_converges_within_lemma_bounds(medium):
    grid, actions, res = medium
    assert res.converged and res.iterations < 50
    U = vi.ValueFunction(grid, res.Z)
    mu, vs = make_test_points(41, 16, n_near=200)
    d = S.dist2(mu, vs)
    ratio = U(mu, vs)[d > 0] / d[d > 0]
    assert ratio.min() >= 1 - 1e-9 and ratio.max() <= S.KAPPA2


def test_closed_form_on_R_and_optimal_action_at_target(medium):
    grid, actions, res = medium
    U = vi.ValueFunction(grid, res.Z)
    mu, vs = make_test_points(81, 31, n_near=500)
    inside = S.deadbeat(mu, vs)[2]
    assert np.max(np.abs(U(mu[inside], vs[inside]) - S.dist2(mu[inside], vs[inside]))) < 2e-3
    i, j = np.argmax(grid.mu == S.MU_STAR), np.argmax(grid.vs == S.VS_STAR)
    assert np.allclose(actions[res.K[i, j]], S.A_STAR)          # node argmin at omega_star
    a, _ = vi.greedy(U, np.array([S.MU_STAR]), np.array([S.VS_STAR]))
    assert np.allclose(a[0], S.A_STAR)


def test_rollout_bound_and_bellman_residual(medium):
    grid, _, res = medium
    U = vi.ValueFunction(grid, res.Z)
    mu, vs = make_test_points(21, 9, n_near=50)
    assert np.all(vi.rollout_upper_bound(U, mu, vs) >= U(mu, vs) - 5e-3)
    assert np.max(vi.bellman_residual(U, mu, vs)) < 5e-3


def test_converged_flag_requires_target_node():
    """Structural check of the flag: the same value iteration with omega_star moved off the grid by 1e-6 reaches the
    tolerance as well, but only the grid that contains omega_star as a node is flagged as converged."""
    acts = graded_actions(S.A_STAR, (0.01, 0.35, 0.03), (0.01, 0.35, 0.1))
    base = Grid.graded(0.02, 0.35, 0.2, 0.1)
    shifted = Grid(np.where(base.mu == S.MU_STAR, S.MU_STAR + 1e-6, base.mu), base.vs)
    on, off = vi.value_iteration_ratio(base, acts), vi.value_iteration_ratio(shifted, acts)
    assert base.has_node(S.OMEGA_STAR) and not shifted.has_node(S.OMEGA_STAR)
    assert on.change < 1e-10 and off.change < 1e-10
    assert on.converged and not off.converged


def test_learned_action_grid_is_centred_at_a_hat():
    a_hat = np.array([0.8031, 1.0457])
    A = offline.learned_actions(a_hat)
    assert np.any(np.all(A == a_hat, axis=1))
    assert not np.any(np.all(np.isclose(A, S.A_STAR), axis=1))
    a1 = np.unique(A[:, 0])
    k = np.argmax(a1 == a_hat[0])
    assert np.isclose(a1[k + 1] - a1[k], 0.01) and np.isclose(a1[k] - a1[k - 1], 0.01)


def _no_oracle(monkeypatch):
    """The true f raises and a_star is NaN: a learner that uses either fails or returns NaN."""
    def f(*args, **kwargs):
        raise AssertionError("the learner called the true f")
    monkeypatch.setattr(S, "f", f)
    monkeypatch.setattr(S, "A_STAR", np.array([np.nan, np.nan]))


def test_offline_learner_uses_only_its_data(monkeypatch):
    """Behavioural leakage check: the offline learner runs with the true f and a_star unavailable."""
    rng = np.random.default_rng(0)
    _no_oracle(monkeypatch)
    learned = offline.learn(4, 30, rng)
    lo, hi = np.array([S.A1_LO, S.A2_LO]), np.array([S.A1_HI, S.A2_HI])
    assert np.all(np.isfinite(learned["a_hat"])) and np.all((learned["a_hat"] >= lo) & (learned["a_hat"] <= hi))
    assert np.all(np.isfinite(learned["actions"])) and np.all(np.isfinite(learned["vi"].Z))
    assert len(learned["actions"]) > 100 and np.any(np.all(learned["actions"] == learned["a_hat"], axis=1))
    assert np.all(np.isfinite(learned["policy"].theta))


def test_model_free_policy_uses_only_the_gp(monkeypatch):
    """Behavioural leakage check: phi_n is computed from the fitted GP W^+_n alone."""
    gp, _ = modelfree.run_learner(np.random.default_rng(1), checkpoints=(20,), M=200)[20]
    _no_oracle(monkeypatch)
    theta = modelfree.greedy_policy(gp).theta
    assert np.all(np.isfinite(theta))
    assert theta[..., 0].min() >= S.A1_LO and theta[..., 0].max() <= S.A1_HI


class _Quadratic:
    """Stand-in for W^+_n with the known off-grid minimizer a_min(omega) = (0.7813 + 0.01 tanh(mu), 1.0071 + 0.02 vs)."""

    @staticmethod
    def a_min(mu, vs):
        return np.stack([0.7813 + 0.01 * np.tanh(mu), 1.0071 + 0.02 * vs], -1)

    def mean(self, X):
        a = self.a_min(X[:, 0], X[:, 1])
        return ((X[:, 2] - a[:, 0]) / 0.25) ** 2 + ((X[:, 3] - a[:, 1]) / 1.0) ** 2


def test_model_free_node_actions_are_refined():
    """The zoom after the 26 x 26 grid moves the node actions to within the final spacing (1/16 step) of the
    minimizer, so they are not tied to the action grid."""
    gp = _Quadratic()
    grid = Grid.graded(*modelfree.POLICY_GRID)
    exact = gp.a_min(grid.M, grid.V)
    coarse = modelfree.greedy_policy(gp, zoom=0).theta
    fine = modelfree.greedy_policy(gp).theta
    h = np.array([(S.A1_HI - S.A1_LO) / 25, (S.A2_HI - S.A2_LO) / 25])
    assert np.all(np.abs(fine - exact) <= h / 32 + 1e-12)
    assert np.max(np.abs(coarse - exact)[..., 1]) > 4 * np.max(np.abs(fine - exact)[..., 1])
