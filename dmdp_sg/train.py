"""PPO-Lagrangian training loop shared by every method (Section 6.3 and Appendix D).

Every method maximizes the discounted return of the shaped reward

    r~_t = r_t - beta * D(rho_{omega_bar_t} || rho_star)          (beta = 0 for the baselines)

with PPO and generalized advantage estimation. Methods with ``lagrangian=True`` also enforce the
episodic cost constraint J_c <= c_bar through a Lagrange multiplier lambda >= 0: before each policy
update, lambda takes one Adam step on -lambda * (J_c_hat - c_bar) and is projected onto lambda >= 0,
where J_c_hat is the mean cost of the most recently completed training episodes. The policy is
updated with the combined advantage (A^r_t - lambda * A^c_t) / (1 + lambda).

Each run writes three files to its output directory:
    policy.pt     network weights, input normalization, configuration and dimensions
    training.csv  one row per update, from the first update in which an episode was completed:
                  step, mean episodic return, cost, dissimilarity and length of the last
                  4 * num_envs completed episodes, Lagrange multiplier and wall-clock seconds
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

from .agent import Agent, RunningMeanStd
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

    @classmethod
    def zeros(cls, num_steps: int, num_envs: int, obs_dim: int, act_dim: int) -> RolloutBuffer:
        shape = (num_steps, num_envs)
        return cls(obs=torch.zeros(shape + (obs_dim,)), actions=torch.zeros(shape + (act_dim,)),
                   log_probs=torch.zeros(shape), rewards=torch.zeros(shape), costs=torch.zeros(shape),
                   dones=torch.zeros(shape), values=torch.zeros(shape), cost_values=torch.zeros(shape))


def _collect_rollout(envs, agent: Agent, rms: RunningMeanStd, cfg: TrainConfig, buf: RolloutBuffer,
                     next_obs: torch.Tensor, next_done: torch.Tensor, finished: list[dict[str, float]]
                     ) -> tuple[torch.Tensor, torch.Tensor]:
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
        buf.actions[step] = action
        buf.log_probs[step] = log_prob
        raw_obs, reward, terminated, truncated, infos = envs.step(np.clip(action.numpy(), act_low, act_high))
        cost = _info_at_step(infos, "cost", n)
        dissimilarity = _info_at_step(infos, "dissimilarity", n)
        buf.rewards[step] = torch.as_tensor(reward - cfg.beta * dissimilarity, dtype=torch.float32)
        buf.costs[step] = torch.as_tensor(cost, dtype=torch.float32)
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

    Minibatches are drawn with the global NumPy random generator (seeded in ``train``).
    """
    obs_dim, act_dim = buf.obs.shape[-1], buf.actions.shape[-1]
    b_obs = buf.obs.reshape(-1, obs_dim)
    b_act = buf.actions.reshape(-1, act_dim)
    b_logp = buf.log_probs.reshape(-1)
    b_adv = advantages.reshape(-1)
    b_ret_r = reward_returns.reshape(-1)
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
            v_loss = 0.5 * ((agent.value(b_obs[mb]) - b_ret_r[mb]) ** 2).mean()
            c_loss = 0.5 * ((agent.cost_value(b_obs[mb]) - b_ret_c[mb]) ** 2).mean()
            loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * (v_loss + c_loss)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()


def _save_run(out_dir: str, agent: Agent, rms: RunningMeanStd, cfg: TrainConfig, obs_dim: int, act_dim: int,
              history: list[dict[str, float]]) -> None:
    """Writes policy.pt, training.csv (empty if no episode was completed) and config.json."""
    torch.save(dict(agent=agent.state_dict(), rms=rms.state_dict(), cfg=cfg.to_dict(),
                    obs_dim=obs_dim, act_dim=act_dim), os.path.join(out_dir, "policy.pt"))
    with open(os.path.join(out_dir, "training.csv"), "w", newline="") as f:
        if history:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=1)


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

    agent = Agent(obs_dim, act_dim, cfg.hidden)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    rms = RunningMeanStd(obs_dim)
    lagrange_multiplier = torch.zeros(1, requires_grad=True)   # lambda >= 0, kept by projection
    multiplier_optimizer = torch.optim.Adam([lagrange_multiplier], lr=cfg.lagrangian_lr)

    num_updates = cfg.total_steps // (n * T)
    buf = RolloutBuffer.zeros(T, n, obs_dim, act_dim)
    n_recent = EPISODES_PER_ENV * n

    raw_obs, _ = envs.reset(seed=[cfg.seed * 1000 + i for i in range(n)])
    rms.update(raw_obs)
    next_obs = torch.as_tensor(rms.normalize(raw_obs))
    next_done = torch.zeros(n)
    global_step = 0
    episodic_cost = 0.0             # J_c_hat, kept from the previous update if no episode was completed
    history: list[dict[str, float]] = []
    recent: list[dict[str, float]] = []
    t0 = time.time()

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate
        next_obs, next_done = _collect_rollout(envs, agent, rms, cfg, buf, next_obs, next_done, recent)
        global_step += n * T
        recent = recent[-n_recent:]

        if recent:
            episodic_cost = float(np.mean([s["cost"] for s in recent]))
        if cfg.lagrangian:
            _update_lagrange_multiplier(lagrange_multiplier, multiplier_optimizer, episodic_cost, cfg.cost_limit)
        lam = float(lagrange_multiplier.item()) if cfg.lagrangian else 0.0

        with torch.no_grad():
            adv_r, ret_r = compute_gae(buf.rewards, buf.values, buf.dones, agent.value(next_obs), next_done,
                                       cfg.gamma, cfg.gae_lambda)
            adv_c, ret_c = compute_gae(buf.costs, buf.cost_values, buf.dones, agent.cost_value(next_obs), next_done,
                                       cfg.gamma, cfg.gae_lambda)
        advantages = (adv_r - lam * adv_c) / (1.0 + lam)
        _ppo_update(agent, optimizer, buf, advantages, ret_r, ret_c, cfg)

        if recent:
            row = dict(
                step=global_step,
                ep_return=float(np.mean([s["ret"] for s in recent])),
                ep_cost=float(np.mean([s["cost"] for s in recent])),
                ep_dissimilarity=float(np.mean([s["dis"] for s in recent])),
                ep_length=float(np.mean([s["length"] for s in recent])),
                lagrange_multiplier=lam,
                seconds=time.time() - t0,
            )
            history.append(row)
            if verbose and (update % log_every == 0 or update == num_updates):
                print(f"[{cfg.env}/{cfg.method}/s{cfg.seed}] step {global_step:>8d}  return {row['ep_return']:8.1f}  "
                      f"cost {row['ep_cost']:6.1f}  D {row['ep_dissimilarity']:.3f}  lambda {lam:.3f}  "
                      f"{global_step / row['seconds']:.0f} steps/s", flush=True)

    envs.close()
    _save_run(out_dir, agent, rms, cfg, obs_dim, act_dim, history)
    return history
