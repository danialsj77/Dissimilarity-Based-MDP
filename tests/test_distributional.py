"""Distributional critics of Distributional PPO and Distributional SAC (quantile regression of the return)."""

import numpy as np
import pytest
import torch

from dmdp_sg.agent import Agent, RunningMeanStd, quantile_huber_loss, quantile_levels
from dmdp_sg.config import make_config
from dmdp_sg.sac import ReturnStatistics, SACAgent, sac_update
from dmdp_sg.train import _bootstrap_truncated, compute_gae, compute_quantile_returns


def test_quantile_levels_are_the_bin_midpoints():
    assert torch.allclose(quantile_levels(4), torch.tensor([0.125, 0.375, 0.625, 0.875]))


def test_quantile_huber_loss_learns_the_quantiles_of_the_targets():
    # minimizing the loss over free parameters theta_i gives the tau_i-quantiles of the target samples
    torch.manual_seed(0)
    samples = torch.distributions.Exponential(1.0).sample((1, 4000))
    theta = torch.zeros(1, 8, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.05)
    for _ in range(3000):
        opt.zero_grad()
        quantile_huber_loss(theta, samples, kappa=0.01).backward()     # small kappa: close to the pure quantile loss
        opt.step()
    expected = torch.quantile(samples[0], quantile_levels(8))
    assert torch.allclose(theta.detach()[0], expected, atol=0.05)


def test_quantile_huber_loss_value_and_target_without_gradient():
    pred = torch.tensor([[0.0, 1.0]], requires_grad=True)
    target = torch.tensor([[0.5]], requires_grad=True)
    # u = (0.5, -0.5); tau = (0.25, 0.75); Huber(0.5) = 0.125: 0.25 * 0.125 + |0.75 - 1| * 0.125 = 0.0625
    loss = quantile_huber_loss(pred, target)
    assert loss.item() == pytest.approx(0.0625)
    loss.backward()
    assert target.grad is None and pred.grad is not None


def test_agent_value_is_the_mean_of_the_quantiles():
    agent = Agent(5, 2, hidden=8, n_quantiles=6)
    x = torch.randn(4, 5)
    assert agent.quantiles(x).shape == (4, 6) and torch.allclose(agent.value(x), agent.quantiles(x).mean(-1))
    with pytest.raises(ValueError):
        Agent(5, 2, hidden=8).quantiles(x)


def test_scalar_agent_is_unchanged():
    # the networks of the other methods are built exactly as before (same parameters for the same seed)
    torch.manual_seed(3)
    a = Agent(5, 2, hidden=8)
    torch.manual_seed(3)
    b = Agent(5, 2, hidden=8, n_quantiles=0)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert a.critic[-1].out_features == 1


def test_mean_of_the_quantile_returns_is_the_lambda_return():
    torch.manual_seed(1)
    T, n, N, gamma, lam = 7, 3, 5, 0.9, 0.8
    rewards, quantiles = torch.randn(T, n), torch.randn(T, n, N)
    offsets = torch.randn(T, n, N)
    offsets -= offsets.mean(-1, keepdim=True)              # the offsets of a truncation have mean zero over the quantiles
    dones = (torch.rand(T, n) < 0.3).float()
    dones[0] = 0.0
    last_q, last_done = torch.randn(n, N), torch.tensor([0.0, 1.0, 0.0])
    y = compute_quantile_returns(rewards, offsets, quantiles, dones, last_q, last_done, gamma, lam)
    _, ret = compute_gae(rewards, quantiles.mean(-1), dones, last_q.mean(-1), last_done, gamma, lam)
    assert y.shape == (T, n, N) and torch.allclose(y.mean(-1), ret, atol=1e-5)
    # lam = 0: one-step distributional targets r_t + gamma * theta_j(s_{t+1})
    y0 = compute_quantile_returns(rewards, torch.zeros_like(offsets), quantiles, torch.zeros(T, n), last_q, torch.zeros(n),
                                  gamma, 0.0)
    assert torch.allclose(y0[:-1], rewards[:-1].unsqueeze(-1) + gamma * quantiles[1:])
    assert torch.allclose(y0[-1], rewards[-1].unsqueeze(-1) + gamma * last_q)


def test_truncation_bootstraps_every_quantile():
    agent = Agent(3, 1, hidden=8, n_quantiles=4)
    rms = RunningMeanStd(3)
    final = np.array([[0.3, -0.2, 0.5]], np.float32)
    infos = {"final_observation": [final[0]]}
    rewards, costs, offsets = torch.zeros(1), torch.zeros(1), torch.zeros(1, 4)
    _bootstrap_truncated(rewards, costs, agent, rms, infos, np.array([False]), np.array([True]), 0.9, offsets)
    with torch.no_grad():
        q = agent.quantiles(torch.as_tensor(rms.normalize(final)))[0]
    # the per-quantile reward r + offset_j equals the bootstrap gamma * theta_j(s_final)
    assert torch.allclose(rewards[0] + offsets[0], 0.9 * q, atol=1e-6)


def test_distributional_sac_target_uses_the_critic_with_the_smaller_mean():
    torch.manual_seed(0)
    cfg = make_config("HalfCheetah", "Dist-SAC", 0, n_quantiles=4)
    agent = SACAgent(3, 2, 8, -np.ones(2), np.ones(2), n_quantiles=4)
    with torch.no_grad():                                   # target critic 2 predicts lower returns everywhere
        agent.q2_target.net[-1].bias.fill_(-5.0)
        agent.q1_target.net[-1].bias.fill_(5.0)
    opts = dict(q=torch.optim.SGD(list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=0.0),
                qc=torch.optim.SGD(agent.qc.parameters(), lr=0.0), actor=torch.optim.SGD(agent.actor.parameters(), lr=0.0),
                alpha=torch.optim.SGD([agent.log_alpha], lr=0.0))
    batch = dict(obs=torch.randn(16, 3), act=torch.rand(16, 2) * 2 - 1, reward=torch.zeros(16), cost=torch.zeros(16),
                 terminated=torch.zeros(16), next_obs=torch.randn(16, 3))
    torch.manual_seed(5)
    with torch.no_grad():                                   # before the update, which moves the target critics (Polyak)
        next_act, next_logp = agent.actor.sample(batch["next_obs"])
        expected = 0.99 * (agent.q2_target(batch["next_obs"], next_act) - next_logp.unsqueeze(-1))   # alpha = 1
    captured = {}
    real = quantile_huber_loss

    def spy(pred, target, kappa=1.0):
        captured.setdefault("target", target)
        return real(pred, target, kappa)
    import dmdp_sg.sac as sac
    sac.quantile_huber_loss, saved = spy, sac.quantile_huber_loss
    try:
        torch.manual_seed(5)
        sac_update(agent, opts, batch, ReturnStatistics(1, 0.99, enabled=False), 0.0, cfg, target_entropy=-2.0)
    finally:
        sac.quantile_huber_loss = saved
    assert captured["target"].shape == (16, 4) and torch.allclose(captured["target"], expected, atol=1e-5)
