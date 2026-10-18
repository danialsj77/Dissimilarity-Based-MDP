"""PPO-Lagrangian training loop shared by every method (Section 4.5, Appendix E, and Appendix F).

Every method maximizes the discounted return of the reward

    r~_t = r_t + r_tD - beta * D(rho_bar_t || rho_star)

with PPO and generalized advantage estimation, where r_t = v_t - (kappa * c_t)^2 - 0.1 * ||a_t||^2 (``envs.CostPenalty``)
is the benchmark reward for kappa = 0, the value of every reported method (kappa = 2 only in the design pilot), and
r_tD = +d if D <= D_thr and 0 otherwise is the threshold reward (``envs.ThresholdReward``). PPO, PPO-Lag, Dist-PPO, Dist-PPO-Lag, Implicit-PPO and
Implicit-PPO-Lag have d = 0 and beta = 0, and DMDP and DMDP-Lag have d = 10 and beta = BETA (``config.METHODS``). Methods with ``lagrangian=True`` also enforce the
episodic cost constraint J_c <= c_bar through a Lagrange multiplier lambda >= 0: before each policy
update, lambda takes one Adam step on -lambda * (J_c_hat - c_bar) and is projected onto lambda >= 0,
where J_c_hat is the mean cost of the most recently completed training episodes. The policy is
updated with the combined advantage (A^r_t - lambda * A^c_t) / (1 + lambda), where A^r is standardized and A^c is
centered over the rollout, so that the weight of the cost does not depend on the scale of the reward
(``combine_advantages``); the combined advantage is standardized again in every minibatch.

Distributional PPO and Implicit PPO (``n_quantiles > 0``) have a distributional reward critic: it outputs n_quantiles quantiles of the return,
and the value used by GAE is their mean, so the advantage is formed as in PPO. The quantiles are learned with the
quantile-regression loss (``agent.quantile_huber_loss``) toward per-quantile lambda-returns (``compute_quantile_returns``),
whose mean over the quantiles is the lambda-return target of the PPO critic. The training reward r~_t is
divided by the running standard deviation of its discounted return (``ReturnScaler``), so that the critic sees
targets of unit scale for every method.

Every episode ends at the time limit H by truncation, not by termination. The values of the final
observation are therefore bootstrapped at the time limit (``_bootstrap_truncated``), so that GAE
does not treat the time limit as a terminal state. The gradient norm is clipped separately for the
actor, the reward critic and the cost critic.

Each run writes four files to its output directory:
    policy.pt     network weights, input normalization, configuration and dimensions
    training.csv  one row per update, from the first update in which an episode was completed:
                  step, mean episodic return of r_t + r_tD and of r_t, cost, mean and median dissimilarity, fraction of
                  steps within D_thr, mean absolute pitch, fraction of steps with a flipped torso (|pitch| > pi/2) and
                  length of the last 4 * num_envs completed episodes, Lagrange multiplier and wall-clock seconds
    episodes.csv  one row per completed training episode: step (at the end of the update in which the
                  episode was completed), episodic return of r_t + r_tD and of r_t, cost, mean and median dissimilarity,
                  fraction within D_thr, mean absolute pitch and fraction of steps with a flipped torso
    config.json   the TrainConfig of the run
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .agent import Agent, RunningMeanStd, quantile_huber_loss
from .config import TrainConfig
from .envs import make_vector_env

EPISODES_PER_ENV = 4   # J_c_hat and the logged statistics use the last 4 * num_envs completed episodes


def _info_at_step(infos: dict[str, Any], key: str, num_envs: int) -> np.ndarray:
    """Per-environment value of ``info[key]`` for the step just taken.

    When an episode ends, the vector environment resets it automatically and stores the info of
    the final step in ``infos["final_info"]``; that value is used for the finished environments.
    """
    out = np.zeros(num_envs, np.float64)
    top = infos.get(key)
    top_mask = infos.get("_" + key, np.zeros(num_envs, bool))
    final = infos.get("final_info")
    final_mask = infos.get("_final_info", np.zeros(num_envs, bool))
    for i in range(num_envs):
        if final_mask[i] and final is not None and final[i] is not None and key in final[i]:
            out[i] = final[i][key]
        elif top is not None and top_mask[i]:
            out[i] = top[i]
    return out


def _completed_episodes(infos: dict[str, Any], num_envs: int) -> list[dict[str, float]]:
    """Episode statistics (see ``envs.EpisodeStats``) of the episodes that ended at this step."""
    final = infos.get("final_info")
    final_mask = infos.get("_final_info", np.zeros(num_envs, bool))
    stats = []
    for i in range(num_envs):
        if final_mask[i] and final is not None and final[i] is not None and "episode_stats" in final[i]:
            stats.append(final[i]["episode_stats"])
    return stats


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor,
                last_done: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation over a rollout of shape (num_steps, num_envs).

    ``dones[t]`` flags that the observation at step t starts a new episode. Returns the
    advantages and the value targets (advantages + values).
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal = 1.0 - last_done
            next_values = last_value
        else:
            nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    return adv, adv + values


def compute_quantile_returns(rewards: torch.Tensor, offsets: torch.Tensor, quantiles: torch.Tensor, dones: torch.Tensor,
                             last_quantiles: torch.Tensor, last_done: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """Per-quantile lambda-returns: the targets of a distributional critic over a rollout of shape (num_steps, num_envs).

    ``quantiles[t]`` (num_steps, num_envs, n) are the quantiles of the critic at the observation of step t, and
    ``offsets[t]`` adds, to the reward of an episode truncated at step t, gamma * (theta_j(s_final) - V(s_final)): the
    reward already contains gamma * V(s_final) (``_bootstrap_truncated``), and the offsets replace it quantile by quantile.
    The target of quantile j is the lambda-return with the j-th quantiles in place of the values,

        y_{t,j} = r_t + offsets_{t,j} + gamma * (1 - done_{t+1}) * ((1 - lam) * theta_j(s_{t+1}) + lam * y_{t+1,j}),

    with y_{T,j} = theta_j(s_T) after the last step. The mean over j is the lambda-return of ``compute_gae`` with the
    values V = mean_j theta_j, so the mean of the learned quantiles has the target of the PPO critic.
    """
    T = rewards.shape[0]
    targets = torch.zeros_like(quantiles)
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal = 1.0 - last_done
            next_quantiles = next_targets = last_quantiles
        else:
            nonterminal = 1.0 - dones[t + 1]
            next_quantiles, next_targets = quantiles[t + 1], targets[t + 1]
        targets[t] = (rewards[t].unsqueeze(-1) + offsets[t]
                      + gamma * nonterminal.unsqueeze(-1) * ((1.0 - lam) * next_quantiles + lam * next_targets))
    return targets


class ReturnScaler:
    """Divides the training reward by the running standard deviation of its discounted return.

    The rewards of the methods differ in scale by an order of magnitude (the threshold reward is +10, and the
    dissimilarity is about 7.1 at rest and about 10-17 with a flipped torso), and an unnormalized critic does not fit
    targets of that scale early in training. As in the ``NormalizeReward`` wrapper of Gymnasium, a discounted return
    G <- gamma * G + r is tracked for every environment (reset at the end of an episode), and the reward is divided by
    the standard deviation of all values of G seen so far; the scaled reward is clipped to [-10, 10]. Reported returns
    are never scaled.
    """

    def __init__(self, num_envs: int, gamma: float, enabled: bool = True):
        self.returns = np.zeros(num_envs)
        self.rms = RunningMeanStd(())
        self.gamma = gamma
        self.enabled = enabled

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return reward
        self.returns = self.gamma * self.returns + reward
        self.rms.update(self.returns)
        self.returns[done] = 0.0
        return np.clip(reward / np.sqrt(self.rms.var + 1e-8), -10.0, 10.0)


def combine_advantages(adv_r: torch.Tensor, adv_c: torch.Tensor, lam: float) -> torch.Tensor:
    """PPO-Lagrangian advantage (A_r - lam * A_c) / (1 + lam), with A_r standardized and A_c centered over the rollout.

    Standardizing the reward advantage makes the weight of the cost independent of the scale of the reward of the
    method. The factor 1 / (1 + lam) only keeps the combined advantage at a fixed scale: ``_ppo_update`` standardizes
    it again in every minibatch, so the factor cancels (up to the 1e-8 added to the standard deviation) and the policy
    step depends on lam only through the weight lam of the cost relative to the reward advantage.
    """
    a_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
    a_c = adv_c - adv_c.mean()
    return (a_r - lam * a_c) / (1.0 + lam)


def _bootstrap_truncated(rewards: torch.Tensor, costs: torch.Tensor, agent: Agent, rms: RunningMeanStd,
                         infos: dict[str, Any], terminated: np.ndarray, truncated: np.ndarray, gamma: float,
                         offsets: torch.Tensor | None = None) -> None:
    """Adds gamma * V(s_final) to the reward and gamma * V_c(s_final) to the cost of truncated episodes.

    ``rewards`` and ``costs`` are the rows of the rollout buffer for the current step (modified in
    place). The vector environment resets finished episodes automatically and stores their final
    observation in ``infos["final_observation"]``. The episode boundary itself stays in ``dones``,
    so the GAE trace is still cut there; only the missing bootstrap value is added. For a distributional
    critic, ``offsets`` (the row of the buffer, num_envs x n) receives gamma * (theta_j(s_final) - V(s_final)) for the
    per-quantile targets (``compute_quantile_returns``).
    """
    idx = np.nonzero(np.logical_and(truncated, np.logical_not(terminated)))[0]
    if idx.size == 0:
        return
    final = torch.as_tensor(rms.normalize(np.stack([infos["final_observation"][i] for i in idx])))
    idx_t = torch.as_tensor(idx)
    with torch.no_grad():
        rewards[idx_t] += gamma * agent.value(final)
        costs[idx_t] += gamma * agent.cost_value(final)
        if offsets is not None:
            q = agent.quantiles(final)
            offsets[idx_t] = gamma * (q - q.mean(-1, keepdim=True))


@dataclass
class RolloutBuffer:
    """Transitions of one rollout, each tensor of shape (num_steps, num_envs, ...)."""

    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor        # shaped rewards r~_t
    costs: torch.Tensor          # safety costs c_t
    dones: torch.Tensor
    values: torch.Tensor         # reward critic V_r
    cost_values: torch.Tensor    # cost critic V_c
    quantiles: torch.Tensor | None = None          # distributional critic: quantiles of the return (..., n_quantiles)
    quantile_offsets: torch.Tensor | None = None   # distributional critic: per-quantile bootstrap at truncation

    @classmethod
    def zeros(cls, num_steps: int, num_envs: int, obs_dim: int, act_dim: int, n_quantiles: int = 0) -> RolloutBuffer:
        shape = (num_steps, num_envs)
        dist = dict(quantiles=torch.zeros(shape + (n_quantiles,)), quantile_offsets=torch.zeros(shape + (n_quantiles,))) \
            if n_quantiles > 0 else {}
        return cls(obs=torch.zeros(shape + (obs_dim,)), actions=torch.zeros(shape + (act_dim,)),
                   log_probs=torch.zeros(shape), rewards=torch.zeros(shape), costs=torch.zeros(shape),
                   dones=torch.zeros(shape), values=torch.zeros(shape), cost_values=torch.zeros(shape), **dist)


def _collect_rollout(envs, agent: Agent, rms: RunningMeanStd, cfg: TrainConfig, buf: RolloutBuffer,
                     next_obs: torch.Tensor, next_done: torch.Tensor, finished: list[dict[str, float]],
                     scaler: ReturnScaler | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs ``cfg.num_steps`` steps in every environment and stores them in ``buf``.

    Statistics of completed episodes are appended to ``finished``. Returns the normalized
    observation and done flags after the last step.
    """
    n = cfg.num_envs
    act_low, act_high = envs.single_action_space.low, envs.single_action_space.high
    for step in range(cfg.num_steps):
        buf.obs[step] = next_obs
        buf.dones[step] = next_done
        with torch.no_grad():
            action, log_prob, _ = agent.act(next_obs)
            buf.values[step] = agent.value(next_obs)
            buf.cost_values[step] = agent.cost_value(next_obs)
            if buf.quantiles is not None:
                buf.quantiles[step] = agent.quantiles(next_obs)
        buf.actions[step] = action
        buf.log_probs[step] = log_prob
        raw_obs, reward, terminated, truncated, infos = envs.step(np.clip(action.numpy(), act_low, act_high))
        cost = _info_at_step(infos, "cost", n)
        dissimilarity = _info_at_step(infos, "dissimilarity", n)
        shaped = reward - cfg.beta * dissimilarity                   # r_t + r_tD - beta * D_t
        if scaler is not None:
            shaped = scaler(shaped, np.logical_or(terminated, truncated))
        buf.rewards[step] = torch.as_tensor(shaped, dtype=torch.float32)
        buf.costs[step] = torch.as_tensor(cost, dtype=torch.float32)
        if buf.quantile_offsets is not None:
            buf.quantile_offsets[step] = 0.0
        _bootstrap_truncated(buf.rewards[step], buf.costs[step], agent, rms, infos, terminated, truncated, cfg.gamma,
                             None if buf.quantile_offsets is None else buf.quantile_offsets[step])
        rms.update(raw_obs)
        next_obs = torch.as_tensor(rms.normalize(raw_obs))
        next_done = torch.as_tensor(np.logical_or(terminated, truncated), dtype=torch.float32)
        finished.extend(_completed_episodes(infos, n))
    return next_obs, next_done


def _update_lagrange_multiplier(lagrange_multiplier: torch.Tensor, optimizer: torch.optim.Optimizer,
                                episodic_cost: float, cost_limit: float) -> None:
    """One Adam step on -lambda * (J_c_hat - c_bar), followed by the projection onto lambda >= 0."""
    optimizer.zero_grad()
    (-lagrange_multiplier * (episodic_cost - cost_limit)).sum().backward()
    optimizer.step()
    with torch.no_grad():
        lagrange_multiplier.clamp_(min=0.0)


def _ppo_update(agent: Agent, optimizer: torch.optim.Optimizer, buf: RolloutBuffer, advantages: torch.Tensor,
                reward_returns: torch.Tensor, cost_returns: torch.Tensor, cfg: TrainConfig) -> None:
    """Clipped-surrogate PPO epochs on the combined advantage, with reward and cost value losses.

    The actor, the reward critic and the cost critic share no parameters; the gradient norm of each
    network is clipped separately, so that the actor step does not depend on the value losses.
    Minibatches are drawn with the global NumPy random generator (seeded in ``train``). For a distributional
    critic, ``reward_returns`` holds the per-quantile targets (num_steps, num_envs, n_quantiles), and the reward value
    loss is the quantile-regression loss.
    """
    networks = [list(agent.actor_mean.parameters()) + [agent.actor_logstd],
                list(agent.critic.parameters()), list(agent.cost_critic.parameters())]
    obs_dim, act_dim = buf.obs.shape[-1], buf.actions.shape[-1]
    b_obs = buf.obs.reshape(-1, obs_dim)
    b_act = buf.actions.reshape(-1, act_dim)
    b_logp = buf.log_probs.reshape(-1)
    b_adv = advantages.reshape(-1)
    b_ret_r = reward_returns.reshape(-1, agent.n_quantiles) if agent.n_quantiles > 0 else reward_returns.reshape(-1)
    b_ret_c = cost_returns.reshape(-1)

    batch = cfg.num_envs * cfg.num_steps
    minibatch = batch // cfg.num_minibatches
    idx = np.arange(batch)
    for _ in range(cfg.update_epochs):
        np.random.shuffle(idx)
        for start in range(0, batch, minibatch):
            mb = idx[start:start + minibatch]
            d = agent.dist(b_obs[mb])
            new_logp = d.log_prob(b_act[mb]).sum(-1)
            entropy = d.entropy().sum(-1)
            ratio = (new_logp - b_logp[mb]).exp()
            mb_adv = b_adv[mb]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
            if agent.n_quantiles > 0:
                v_loss = quantile_huber_loss(agent.quantiles(b_obs[mb]), b_ret_r[mb])
            else:
                v_loss = 0.5 * ((agent.value(b_obs[mb]) - b_ret_r[mb]) ** 2).mean()
            c_loss = 0.5 * ((agent.cost_value(b_obs[mb]) - b_ret_c[mb]) ** 2).mean()
            loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * (v_loss + c_loss)
            optimizer.zero_grad()
            loss.backward()
            for params in networks:
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()


def _write_csv(path: str, rows: list[dict[str, float]]) -> None:
    """Writes ``rows`` with a header row; the file is empty if there are no rows."""
    with open(path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _save_run(out_dir: str, agent: Agent, rms: RunningMeanStd, cfg: TrainConfig, obs_dim: int, act_dim: int,
              history: list[dict[str, float]], episodes: list[dict[str, float]]) -> None:
    """Writes config.json, training.csv, episodes.csv (empty if no episode was completed) and, last, policy.pt.

    ``run_all.py`` takes an existing policy.pt for a finished run, so an interrupted save leaves a directory
    without policy.pt, which is trained again.
    """
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=1)
    _write_csv(os.path.join(out_dir, "training.csv"), history)
    _write_csv(os.path.join(out_dir, "episodes.csv"), episodes)
    torch.save(dict(agent=agent.state_dict(), rms=rms.state_dict(), cfg=cfg.to_dict(),
                    obs_dim=obs_dim, act_dim=act_dim), os.path.join(out_dir, "policy.pt"))


def train(cfg: TrainConfig, out_dir: str, log_every: int = 10, verbose: bool = True) -> list[dict[str, float]]:
    """Trains one policy with PPO(-Lagrangian) and saves it to ``out_dir``; returns the training log."""
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(1)
    np.random.seed(cfg.seed)        # minibatch shuffling
    torch.manual_seed(cfg.seed)     # network initialization and action sampling

    envs = make_vector_env(cfg, nu=cfg.train_noise, nu_max=cfg.train_noise_max)
    n, T = cfg.num_envs, cfg.num_steps
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    agent = Agent(obs_dim, act_dim, cfg.hidden, cfg.init_log_std, cfg.n_quantiles)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    rms = RunningMeanStd(obs_dim)
    lagrange_multiplier = torch.zeros(1, requires_grad=True)   # lambda >= 0, kept by projection
    multiplier_optimizer = torch.optim.Adam([lagrange_multiplier], lr=cfg.lagrangian_lr)

    num_updates = cfg.total_steps // (n * T)
    buf = RolloutBuffer.zeros(T, n, obs_dim, act_dim, cfg.n_quantiles)
    n_recent = EPISODES_PER_ENV * n
    scaler = ReturnScaler(n, cfg.gamma, enabled=cfg.reward_normalization)

    raw_obs, _ = envs.reset(seed=[cfg.seed * 1000 + i for i in range(n)])
    rms.update(raw_obs)
    next_obs = torch.as_tensor(rms.normalize(raw_obs))
    next_done = torch.zeros(n)
    global_step = 0
    episodic_cost = 0.0             # J_c_hat, kept from the previous update if no episode was completed
    history: list[dict[str, float]] = []
    episodes: list[dict[str, float]] = []      # every completed training episode (episodes.csv)
    recent: list[dict[str, float]] = []
    t0 = time.time()

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate
        n_before = len(recent)
        next_obs, next_done = _collect_rollout(envs, agent, rms, cfg, buf, next_obs, next_done, recent, scaler)
        global_step += n * T
        episodes.extend(dict(step=global_step, ret=s["ret"], task_ret=s["task_ret"], cost=s["cost"], dis=s["dis"],
                             dis_median=s["dis_median"], within=s["within"], pitch=s["pitch"], flipped=s["flipped"])
                        for s in recent[n_before:])
        recent = recent[-n_recent:]

        if recent:
            episodic_cost = float(np.mean([s["cost"] for s in recent]))
        # The multiplier takes an Adam step at every update. Before the first episode has completed, J_c_hat = 0, so
        # the gradient c_bar > 0 pushes lambda below 0 and lambda stays at 0 through the projection, while the moments
        # of Adam fill with c_bar; they decay once J_c_hat exceeds c_bar and then delay the rise of lambda.
        if cfg.lagrangian:
            _update_lagrange_multiplier(lagrange_multiplier, multiplier_optimizer, episodic_cost, cfg.cost_limit)
        lam = float(lagrange_multiplier.item()) if cfg.lagrangian else 0.0

        with torch.no_grad():
            adv_r, ret_r = compute_gae(buf.rewards, buf.values, buf.dones, agent.value(next_obs), next_done,
                                       cfg.gamma, cfg.gae_lambda)
            adv_c, ret_c = compute_gae(buf.costs, buf.cost_values, buf.dones, agent.cost_value(next_obs), next_done,
                                       cfg.gamma, cfg.gae_lambda)
            if cfg.n_quantiles > 0:     # distributional critic: per-quantile targets; the advantage uses the mean value
                ret_r = compute_quantile_returns(buf.rewards, buf.quantile_offsets, buf.quantiles, buf.dones,
                                                 agent.quantiles(next_obs), next_done, cfg.gamma, cfg.gae_lambda)
        advantages = combine_advantages(adv_r, adv_c, lam)
        _ppo_update(agent, optimizer, buf, advantages, ret_r, ret_c, cfg)

        if recent:
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
            )
            history.append(row)
            if verbose and (update % log_every == 0 or update == num_updates):
                print(f"[{cfg.env}/{cfg.method}/s{cfg.seed}] step {global_step:>8d}  task return {row['ep_task_return']:8.1f}  "
                      f"train return {row['ep_return']:8.1f}  "
                      f"cost {row['ep_cost']:6.1f}  D {row['ep_dissimilarity']:.3f}  flipped {row['ep_flipped']:.2f}  "
                      f"lambda {lam:.3f}  "
                      f"{global_step / row['seconds']:.0f} steps/s", flush=True)

    envs.close()
    _save_run(out_dir, agent, rms, cfg, obs_dim, act_dim, history, episodes)
    return history
