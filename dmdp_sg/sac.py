"""Soft actor-critic (SAC) learner and its Lagrangian variant (Appendix E; Section 5.1).

The second learner of the study. Every method maximizes the same training reward as with PPO (``train``),

    r~_t = r_t + r_tD - beta * D(rho_bar_t || rho_star),

and only the learner differs: SAC, SAC-Lag, Dist-SAC, Dist-SAC-Lag, Implicit-SAC and Implicit-SAC-Lag have d = 0 and beta = 0,
and DMDP and DMDP-Lag have d = 10 and beta = BETA (``config.METHODS``); every method has kappa = 0, i.e. r_t is the benchmark reward. The learner is SAC with twin reward critics, target networks updated by Polyak
averaging, a tanh-squashed Gaussian policy and an automatically tuned entropy temperature alpha with target entropy
-dim(a) (Haarnoja et al., 2018). Methods with ``lagrangian=True`` also learn a cost critic Q_c and enforce the episodic
cost constraint J_c <= c_bar through a Lagrange multiplier lambda >= 0, as in SAC-Lagrangian (Ha et al., 2020): the actor
minimizes (alpha * log pi(a|s) - min(Q_1, Q_2)(s, a) + lambda * Q_c(s, a)) / (1 + lambda), i.e. the actor loss of
Ha et al. divided by 1 + lambda as in the PPO learner. The multiplier is updated with
the rule of the PPO learner, one Adam step on -lambda * (J_c_hat - c_bar) every ``sac_lambda_interval`` environment steps
(the number of steps per PPO update), where J_c_hat is the mean cost of the most recently completed training episodes.
The multiplier uses the learning rate of the PPO learner (``lagrangian_lr``). Because the actor weighs Q_c against the reward
critic, whose scale is that of the scaled reward, and not against a standardized advantage as in PPO, the multiplier at which
the constraint binds is several times smaller than with PPO (about 0.2-0.9 instead of 2-5); a learning rate seven times smaller
was tried in pilot runs on seeds 10 and 11 (``run_all.py --lagrangian-lr 0.005``) and enforced the constraint more slowly,
with larger and longer cost overshoots.
The initial standard deviation ``init_log_std`` of the PPO learner does not apply: SAC explores with uniformly random actions
during the first ``sac_learning_starts`` steps and then with the entropy-regularized policy.

The replay buffer stores the unscaled training reward. As in the PPO learner, the reward is divided by the running
standard deviation of its discounted return (``ReturnStatistics``); the division uses the current estimate when a
minibatch is sampled, so that all stored transitions are scaled alike, and the scaled reward is clipped to [-10, 10].
The observations are not normalized, as is common for SAC. Every episode ends at the time limit H by truncation, so the
critic targets bootstrap from the final observation of the episode (``infos["final_observation"]``). During the first
``sac_learning_starts`` steps, the actions are drawn uniformly from the action space and the networks are not updated;
afterwards, every environment step of the ``num_envs`` environments is followed by ``num_envs`` gradient steps.

Distributional SAC and Implicit SAC (``n_quantiles > 0``) have distributional reward critics: each of Q_1 and Q_2 outputs n_quantiles quantiles
of the return, learned with the quantile-regression loss (``agent.quantile_huber_loss``) toward the target samples
r + gamma * (theta_j(s', a') - alpha * log pi(a'|s')), where theta are the quantiles of the target critic whose mean is
smaller at (s', a'); the actor uses the means of the quantiles in place of Q_1 and Q_2. For the means, this is the target
and the actor loss of SAC.

Each run writes the files of the PPO learner (policy.pt, training.csv, episodes.csv, config.json) with the same columns;
training.csv has one row each time a batch of episodes is completed and an additional column ``alpha``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .agent import RunningMeanStd, quantile_huber_loss
from .config import TrainConfig
from .envs import make_vector_env
from .train import EPISODES_PER_ENV, _completed_episodes, _info_at_step, _update_lagrange_multiplier, _write_csv

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def _mlp(n_in: int, n_out: int, hidden: int) -> nn.Sequential:
    """Two hidden ReLU layers of width ``hidden``."""
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


class QNetwork(nn.Module):
    """Action-value network Q(s, a), or, with ``n_quantiles > 0``, the quantiles of the return given (s, a)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, n_quantiles: int = 0):
        super().__init__()
        self.n_quantiles = int(n_quantiles)
        self.net = _mlp(obs_dim + act_dim, max(1, self.n_quantiles), hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([obs, act], -1))
        return out if self.n_quantiles > 0 else out.squeeze(-1)


class SACActor(nn.Module):
    """Tanh-squashed diagonal Gaussian policy; the mean action tanh(mu(s)) is used for evaluation."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, act_low: np.ndarray, act_high: np.ndarray):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.register_buffer("scale", torch.as_tensor((act_high - act_low) / 2.0, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor((act_high + act_low) / 2.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        log_std = torch.tanh(self.log_std(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return self.mean(h), log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized action and its log-probability under the squashed distribution."""
        mean, log_std = self(obs)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        y = torch.tanh(u)
        log_prob = torch.distributions.Normal(mean, std).log_prob(u) - torch.log(self.scale * (1.0 - y.pow(2)) + 1e-6)
        return y * self.scale + self.bias, log_prob.sum(-1)

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self(obs)[0]) * self.scale + self.bias


class SACAgent(nn.Module):
    """Actor, twin reward critics, cost critic, their target networks and the entropy temperature.

    The attribute names are the keys of the saved state dict (``policy.pt``) and must not change.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, act_low: np.ndarray, act_high: np.ndarray,
                 n_quantiles: int = 0):
        super().__init__()
        self.actor = SACActor(obs_dim, act_dim, hidden, act_low, act_high)
        self.q1, self.q2 = (QNetwork(obs_dim, act_dim, hidden, n_quantiles) for _ in range(2))
        self.qc = QNetwork(obs_dim, act_dim, hidden)
        self.q1_target, self.q2_target = (QNetwork(obs_dim, act_dim, hidden, n_quantiles) for _ in range(2))
        self.qc_target = QNetwork(obs_dim, act_dim, hidden)
        for net, target in self.target_pairs():
            target.load_state_dict(net.state_dict())
            target.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.zeros(1))

    def target_pairs(self) -> list[tuple[nn.Module, nn.Module]]:
        return [(self.q1, self.q1_target), (self.q2, self.q2_target), (self.qc, self.qc_target)]

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean (or sampled) action, with the signature of ``agent.Agent.act`` used by ``evaluate.rollout``."""
        if deterministic:
            a = self.actor.mean_action(obs)
            return a, torch.zeros(a.shape[:-1]), torch.zeros(a.shape[:-1])
        a, log_prob = self.actor.sample(obs)
        return a, log_prob, -log_prob


class IdentityNormalizer:
    """Stands in for ``agent.RunningMeanStd`` in ``evaluate.rollout``: SAC uses the observations as they are."""

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, np.float32)


class ReturnStatistics:
    """Running standard deviation of the discounted return of the training reward (cf. ``train.ReturnScaler``).

    A discounted return G <- gamma * G + r is tracked for every environment (reset at the end of an episode); the
    reward is divided by the standard deviation of all values of G seen so far when a minibatch is sampled.
    """

    def __init__(self, num_envs: int, gamma: float, enabled: bool = True):
        self.returns = np.zeros(num_envs)
        self.rms = RunningMeanStd(())
        self.gamma = gamma
        self.enabled = enabled

    def update(self, reward: np.ndarray, done: np.ndarray) -> None:
        self.returns = self.gamma * self.returns + reward
        self.rms.update(self.returns)
        self.returns[done] = 0.0

    def scale(self, reward: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return reward
        return (reward / float(np.sqrt(self.rms.var + 1e-8))).clamp(-10.0, 10.0)


class ReplayBuffer:
    """Uniform replay of (s, a, r~, c, terminated, s') with the unscaled training reward r~ and the cost c."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, seed: int):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.reward = np.zeros(capacity, np.float32)
        self.cost = np.zeros(capacity, np.float32)
        self.terminated = np.zeros(capacity, np.float32)
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, act, reward, cost, terminated, next_obs) -> None:
        for i in range(len(obs)):
            self.obs[self.pos], self.act[self.pos], self.next_obs[self.pos] = obs[i], act[i], next_obs[i]
            self.reward[self.pos], self.cost[self.pos], self.terminated[self.pos] = reward[i], cost[i], terminated[i]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = self.rng.integers(0, self.size, batch_size)
        return {k: torch.as_tensor(getattr(self, k)[idx]) for k in ("obs", "act", "reward", "cost", "terminated", "next_obs")}


def _next_observations(next_obs: np.ndarray, infos: dict[str, Any], done: np.ndarray) -> np.ndarray:
    """s_{t+1} of every transition: the final observation of an episode that just ended, not the reset observation."""
    real = np.array(next_obs, np.float32, copy=True)
    for i in np.nonzero(done)[0]:
        real[i] = infos["final_observation"][i]
    return real


def _polyak(agent: SACAgent, tau: float, cost_critic: bool) -> None:
    with torch.no_grad():
        for net, target in agent.target_pairs()[:3 if cost_critic else 2]:
            t_params, params = list(target.parameters()), list(net.parameters())
            torch._foreach_mul_(t_params, 1.0 - tau)
            torch._foreach_add_(t_params, params, alpha=tau)


def sac_update(agent: SACAgent, opts: dict[str, torch.optim.Optimizer], batch: dict[str, torch.Tensor],
               returns: ReturnStatistics, lam: float, cfg: TrainConfig, target_entropy: float) -> dict[str, float]:
    """One gradient step of the critics, the actor and the temperature on a minibatch; returns the losses."""
    alpha = agent.log_alpha.exp().detach()
    reward = returns.scale(batch["reward"])
    not_done = 1.0 - batch["terminated"]
    distributional = cfg.n_quantiles > 0
    with torch.no_grad():
        next_act, next_log_prob = agent.actor.sample(batch["next_obs"])
        if distributional:      # quantiles of the target critic with the smaller mean (clipped double-Q for the means)
            z1, z2 = agent.q1_target(batch["next_obs"], next_act), agent.q2_target(batch["next_obs"], next_act)
            z_next = torch.where((z1.mean(-1) <= z2.mean(-1)).unsqueeze(-1), z1, z2)
            y = reward.unsqueeze(-1) + cfg.gamma * not_done.unsqueeze(-1) * (z_next - alpha * next_log_prob.unsqueeze(-1))
        else:
            q_next = torch.min(agent.q1_target(batch["next_obs"], next_act), agent.q2_target(batch["next_obs"], next_act))
            y = reward + cfg.gamma * not_done * (q_next - alpha * next_log_prob)
        if cfg.lagrangian:
            y_c = batch["cost"] + cfg.gamma * not_done * agent.qc_target(batch["next_obs"], next_act)
    if distributional:
        q_loss = quantile_huber_loss(agent.q1(batch["obs"], batch["act"]), y) \
            + quantile_huber_loss(agent.q2(batch["obs"], batch["act"]), y)
    else:
        q_loss = 0.5 * ((agent.q1(batch["obs"], batch["act"]) - y) ** 2).mean() \
            + 0.5 * ((agent.q2(batch["obs"], batch["act"]) - y) ** 2).mean()
    opts["q"].zero_grad()
    q_loss.backward()
    opts["q"].step()
    out = dict(q_loss=float(q_loss))
    if cfg.lagrangian:
        qc_loss = 0.5 * ((agent.qc(batch["obs"], batch["act"]) - y_c) ** 2).mean()
        opts["qc"].zero_grad()
        qc_loss.backward()
        opts["qc"].step()
        out["qc_loss"] = float(qc_loss)

    act, log_prob = agent.actor.sample(batch["obs"])
    q1, q2 = agent.q1(batch["obs"], act), agent.q2(batch["obs"], act)
    q = torch.min(q1.mean(-1), q2.mean(-1)) if distributional else torch.min(q1, q2)
    actor_loss = alpha * log_prob - q
    if cfg.lagrangian:
        actor_loss = (actor_loss + lam * agent.qc(batch["obs"], act)) / (1.0 + lam)
    actor_loss = actor_loss.mean()
    opts["actor"].zero_grad()
    actor_loss.backward()
    opts["actor"].step()

    alpha_loss = -(agent.log_alpha * (log_prob.detach() + target_entropy)).mean()
    opts["alpha"].zero_grad()
    alpha_loss.backward()
    opts["alpha"].step()
    _polyak(agent, cfg.sac_tau, cfg.lagrangian)
    out.update(actor_loss=float(actor_loss), alpha=float(alpha))
    return out


def _save_run(out_dir: str, agent: SACAgent, cfg: TrainConfig, obs_dim: int, act_dim: int,
              history: list[dict[str, float]], episodes: list[dict[str, float]]) -> None:
    """Writes config.json, training.csv, episodes.csv (empty if no episode was completed) and, last, policy.pt.

    ``run_all.py`` takes an existing policy.pt for a finished run, so an interrupted save leaves a directory
    without policy.pt, which is trained again.
    """
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=1)
    _write_csv(os.path.join(out_dir, "training.csv"), history)
    _write_csv(os.path.join(out_dir, "episodes.csv"), episodes)
    torch.save(dict(agent=agent.state_dict(), cfg=cfg.to_dict(), obs_dim=obs_dim, act_dim=act_dim),
               os.path.join(out_dir, "policy.pt"))


def train_sac(cfg: TrainConfig, out_dir: str, log_every: int = 10, verbose: bool = True) -> list[dict[str, float]]:
    """Trains one policy with SAC(-Lagrangian) and saves it to ``out_dir``; returns the training log."""
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(1)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)     # network initialization and action sampling

    envs = make_vector_env(cfg, nu=cfg.train_noise, nu_max=cfg.train_noise_max)
    n = cfg.num_envs
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    low, high = envs.single_action_space.low, envs.single_action_space.high

    agent = SACAgent(obs_dim, act_dim, cfg.sac_hidden, low, high, cfg.n_quantiles)
    lr = cfg.sac_learning_rate
    opts = dict(q=torch.optim.Adam(list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=lr),
                qc=torch.optim.Adam(agent.qc.parameters(), lr=lr),
                actor=torch.optim.Adam(agent.actor.parameters(), lr=lr),
                alpha=torch.optim.Adam([agent.log_alpha], lr=lr))
    target_entropy = -float(act_dim)
    lagrange_multiplier = torch.zeros(1, requires_grad=True)   # lambda >= 0, kept by projection
    multiplier_optimizer = torch.optim.Adam([lagrange_multiplier], lr=cfg.lagrangian_lr)

    buffer = ReplayBuffer(min(cfg.sac_buffer_size, cfg.total_steps), obs_dim, act_dim, seed=cfg.seed)
    returns = ReturnStatistics(n, cfg.gamma, enabled=cfg.reward_normalization)
    action_rng = np.random.default_rng(cfg.seed + 777)
    n_recent = EPISODES_PER_ENV * n

    obs, _ = envs.reset(seed=[cfg.seed * 1000 + i for i in range(n)])
    global_step, next_multiplier_step = 0, cfg.sac_lambda_interval
    episodic_cost = 0.0
    losses: dict[str, float] = dict(alpha=1.0)
    history: list[dict[str, float]] = []
    episodes: list[dict[str, float]] = []
    recent: list[dict[str, float]] = []
    batches = 0
    t0 = time.time()

    while global_step < cfg.total_steps:
        if global_step < cfg.sac_learning_starts:
            actions = action_rng.uniform(low, high, size=(n, act_dim)).astype(np.float32)
        else:
            actions = agent.act(torch.as_tensor(obs, dtype=torch.float32))[0].numpy()
        next_obs, reward, terminated, truncated, infos = envs.step(actions)
        cost = _info_at_step(infos, "cost", n)
        dissimilarity = _info_at_step(infos, "dissimilarity", n)
        shaped = reward - cfg.beta * dissimilarity                   # r_t + r_tD - beta * D_t
        done = np.logical_or(terminated, truncated)
        returns.update(shaped, done)
        buffer.add(obs, actions, shaped, cost, terminated.astype(np.float32), _next_observations(next_obs, infos, done))
        obs = next_obs
        global_step += n

        finished = _completed_episodes(infos, n)
        episodes.extend(dict(step=global_step, ret=s["ret"], task_ret=s["task_ret"], cost=s["cost"], dis=s["dis"],
                             dis_median=s["dis_median"], within=s["within"], pitch=s["pitch"], flipped=s["flipped"])
                        for s in finished)
        recent = (recent + finished)[-n_recent:]
        if recent:
            episodic_cost = float(np.mean([s["cost"] for s in recent]))
        # As in train.train, the multiplier takes an Adam step at every interval, also before the first episode has
        # completed (J_c_hat = 0): lambda then stays at 0 through the projection while the moments of Adam fill with c_bar.
        while global_step >= next_multiplier_step:
            if cfg.lagrangian:
                _update_lagrange_multiplier(lagrange_multiplier, multiplier_optimizer, episodic_cost, cfg.cost_limit)
            next_multiplier_step += cfg.sac_lambda_interval
        lam = float(lagrange_multiplier.item()) if cfg.lagrangian else 0.0

        if global_step >= cfg.sac_learning_starts:
            for _ in range(n):
                losses = sac_update(agent, opts, buffer.sample(cfg.sac_batch_size), returns, lam, cfg, target_entropy)

        if finished:
            batches += 1
            row = dict(
                step=global_step,
                ep_return=float(np.mean([s["ret"] for s in recent])),
                ep_task_return=float(np.mean([s["task_ret"] for s in recent])),
                ep_cost=float(np.mean([s["cost"] for s in recent])),
                ep_dissimilarity=float(np.mean([s["dis"] for s in recent])),
                ep_dissimilarity_median=float(np.mean([s["dis_median"] for s in recent])),
                ep_within=float(np.mean([s["within"] for s in recent])),
                ep_pitch=float(np.mean([s["pitch"] for s in recent])),
                ep_flipped=float(np.mean([s["flipped"] for s in recent])),
                ep_length=float(np.mean([s["length"] for s in recent])),
                lagrange_multiplier=lam,
                seconds=time.time() - t0,
                alpha=float(losses["alpha"]),
            )
            history.append(row)
            if verbose and batches % log_every == 0:
                print(f"[{cfg.env}/{cfg.method}/sac/s{cfg.seed}] step {global_step:>8d}  task return {row['ep_task_return']:8.1f}  "
                      f"train return {row['ep_return']:8.1f}  cost {row['ep_cost']:6.1f}  D {row['ep_dissimilarity']:.3f}  "
                      f"flipped {row['ep_flipped']:.2f}  lambda {lam:.3f}  alpha {row['alpha']:.3f}  "
                      f"{global_step / row['seconds']:.0f} steps/s", flush=True)

    envs.close()
    _save_run(out_dir, agent, cfg, obs_dim, act_dim, history, episodes)
    return history
