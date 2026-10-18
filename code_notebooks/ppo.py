"""The notebook DMDP_PPO.ipynb: the complete code of the PPO learner."""

from __future__ import annotations

from . import examples, shared
from .builder import Notebook

FILE = "DMDP_PPO.ipynb"


def title(nb: Notebook) -> None:
    nb.md(r"""
# DMDP with PPO: the code

This notebook shows the complete code of the PPO learner of the experiments in
""" + shared.PAPER + r"""
(Section 4.5, Appendix E, and Appendix F of the paper): the target, the configuration, the environment wrappers, the networks, the PPO-Lagrangian training loop, the evaluation, and the metrics.
`DMDP_SAC.ipynb` shows the SAC learner in the same way.

**How the notebook is made.** `build_code_notebooks.py` copies the code cells without change from the package `dmdp_sg/`, one section per module, so the notebook shows the code that produced the stored results; `tests/test_code_notebooks.py` checks this.
The notebook does not import `dmdp_sg`: its cells define everything in one namespace, in the order of the dependencies between the modules.
Two things differ from the files of the package: the imports between the modules are commented out, because their names are defined in the cells above, and `load_policy` shows the branch for the PPO learner only.

**Methods trained with PPO.** Every method maximizes the return of its training reward with the same learner, and the methods differ only in the policy input, the training reward, the reward critic, and the cost constraint.

| Method | Policy input | Training reward | Reward critic | Cost constraint |
|---|---|---|---|---|
| PPO | $\mathbf{o}_t$ | $r^{\mathsf{b}}_t$ | expected return | no |
| PPO-Lag | $\mathbf{o}_t$ | $r^{\mathsf{b}}_t-\lambda c_t$ | expected return | yes |
| Distributional PPO (`Dist-PPO`) | $\mathbf{o}_t$ | $r^{\mathsf{b}}_t$ | 32 quantiles of the return | no |
| Distributional PPO-Lag (`Dist-PPO-Lag`) | $\mathbf{o}_t$ | $r^{\mathsf{b}}_t-\lambda c_t$ | 32 quantiles of the return | yes |
| Implicit DMDP (PPO) (`Implicit-PPO`) | $\hat\omega_t$ and $\mathbf{o}_t$ | $r^{\mathsf{b}}_t$ | 32 quantiles of the return | no |
| Implicit DMDP-Lag (PPO) (`Implicit-PPO-Lag`) | $\hat\omega_t$ and $\mathbf{o}_t$ | $r^{\mathsf{b}}_t-\lambda c_t$ | 32 quantiles of the return | yes |
| DMDP (PPO) (`DMDP`) | $\hat\omega_t$ and $\mathbf{o}_t$ | $r^{\mathsf{b}}_t+r_{t\mathbb{D}}-\beta\,\mathbb{D}_t$ | expected return | no |
| DMDP-Lag (PPO) (`DMDP-Lag`) | $\hat\omega_t$ and $\mathbf{o}_t$ | $r^{\mathsf{b}}_t+r_{t\mathbb{D}}-\beta\,\mathbb{D}_t-\lambda c_t$ | expected return | yes |

Here $r^{\mathsf{b}}_t=v_t-0.1\|\mathbf{a}_t\|^2$ is the benchmark reward, $c_t=\mathbf{1}[v_t>v_{\mathsf{lim}}]$ the safety cost, $\hat\omega_t$ the window statistics of the noisy readings, $\mathbb{D}_t$ the window dissimilarity to the target, and $r_{t\mathbb{D}}$ the threshold reward.
The docstring of `config.py` lists the notation.

**Sections.** 1 Setup · 2 Target distributions · 3 Configuration · 4 Environment wrappers · 5 Policy and critics · 6 PPO-Lagrangian training · 7 Evaluation rollouts · 8 Metrics · 9 Analysis · 10 Examples and checks · 11 Stored results.

**Running it.** Run the cells from top to bottom in the environment of `environment.yml`; the whole notebook takes less than a minute.
Sections 1 to 9 define the code, Section 10 calls it with small settings, and Section 11 reads the stored runs in `results/main`.
The full runs of the paper are launched with `run_all.py` (see the README).
""")


# ---------------------------------------------------------------------------------------------------------------------
# 5 Policy and critics
# ---------------------------------------------------------------------------------------------------------------------
def agent(nb: Notebook) -> None:
    nb.md(r"""
## 5 Policy and critics (`agent.py`)

`agent.py` defines the networks of the PPO-Lagrangian learner: the actor, the reward critic, and the cost critic.
They are three separate multilayer perceptrons with two hidden tanh layers of width 64.
The actor outputs the mean of a diagonal Gaussian with a learned, state-independent log standard deviation, which starts at $-1$ (a standard deviation of $0.37$), and the evaluation uses the mean action.
The inputs are normalized with running estimates of their mean and variance.
With `n_quantiles` $>0$ (Distributional and Implicit PPO), the reward critic outputs the quantiles of the return, and its value is their mean.
""")
    nb.docstring("agent")
    nb.imports("agent")
    nb.md(r"""
`RunningMeanStd` keeps the running mean and variance of the policy input; `normalize` standardizes an input and clips it to $[-10,10]$.
""")
    nb.take("agent", "RunningMeanStd")
    nb.md(r"""
`_layer` and `_mlp` build the networks with orthogonal initialization: the gain is $\sqrt2$ in the hidden layers and a parameter in the output layer ($0.01$ for the actor, so that the first actions are small, and $1$ for the critics).
""")
    nb.take("agent", "_layer", "_mlp", merge=True)
    nb.md(r"""
**Distributional critic.** The quantile levels are the midpoints $\tau_i=(2i-1)/(2n)$ of $n$ equal bins.
`quantile_huber_loss` is the quantile-regression loss of Dabney et al. (2018) with the Huber threshold $\kappa=1$: each output $\theta_i$ learns the $\tau_i$-quantile of the target distribution.
""")
    nb.take("agent", "quantile_levels", "quantile_huber_loss", merge=True)
    nb.md(r"""
**`Agent`** has the networks `actor_mean`, `critic` (the reward critic, with `n_quantiles` outputs for a distributional critic), and `cost_critic`, and the parameter `actor_logstd`.
`act` returns the action, its log-probability, and the entropy; `value` returns the reward value $V_r$, which is the mean of the quantiles for a distributional critic; `cost_value` returns the cost value $V_c$.
The attribute names are the keys of the stored state dict.
""")
    nb.take("agent", "Agent")
    nb.finish("agent")


# ---------------------------------------------------------------------------------------------------------------------
# 6 PPO-Lagrangian training
# ---------------------------------------------------------------------------------------------------------------------
def train(nb: Notebook) -> None:
    nb.md(r"""
## 6 PPO-Lagrangian training (`train.py`)

`train.py` is the training loop that every PPO method shares.
Every method maximizes the discounted return of the training reward $\tilde r_t=r^{\mathsf{b}}_t+r_{t\mathbb{D}}-\beta\,\mathbb{D}_t$, with $r_{t\mathbb{D}}=0$ and $\beta=0$ for the baselines.
The -Lag methods also enforce the constraint $J_c\leq\bar c$ on the episodic cost with a Lagrange multiplier $\lambda\geq0$.
One update has five steps:

1. `_collect_rollout` runs the policy for `num_steps` steps in each of the `num_envs` environments and stores the observations, actions, scaled training rewards, costs, and critic values.
2. The multiplier takes one Adam step on $-\lambda(\hat J_c-\bar c)$ and is projected onto $\lambda\geq0$ (`_update_lagrange_multiplier`); $\hat J_c$ is the mean cost of the last $4\times$`num_envs` completed episodes.
3. `compute_gae` computes the advantages and the value targets of the reward and of the cost by generalized advantage estimation; a distributional critic gets per-quantile targets from `compute_quantile_returns`.
4. `combine_advantages` forms the advantage $(\hat A^r_t-\lambda\hat A^c_t)/(1+\lambda)$.
5. `_ppo_update` runs the clipped-surrogate epochs over minibatches.

Every episode ends at the time limit $H$ by truncation and not by termination, so the value of the final observation is bootstrapped (`_bootstrap_truncated`).
""")
    nb.docstring("train")
    nb.imports("train")
    nb.md(r"""
**Bookkeeping.** $\hat J_c$ and the logged statistics use the last $4\times$`num_envs` completed episodes (`EPISODES_PER_ENV`).
When an episode ends, the vector environment resets it and stores the info of its last step in `final_info`; `_info_at_step` reads a value of the current step from there for the finished environments, and `_completed_episodes` collects the statistics of the finished episodes (`EpisodeStats`).
""")
    nb.take("train", "EPISODES_PER_ENV", "_info_at_step", "_completed_episodes", merge=True)
    nb.md(r"""
**Advantages and targets.** `compute_gae` is generalized advantage estimation over a rollout of shape (steps, environments); `dones[t]` flags that the observation at step $t$ starts a new episode.
For a distributional critic, `compute_quantile_returns` computes the $\lambda$-return of every quantile, and the mean over the quantiles equals the target of `compute_gae` for the mean value.
""")
    nb.take("train", "compute_gae", "compute_quantile_returns", merge=True)
    nb.md(r"""
**Reward scale.** `ReturnScaler` divides the training reward by the running standard deviation of its discounted return $G\leftarrow\gamma G+r$ and clips it to $[-10,10]$, so that the critics see targets of unit scale for every method; the threshold reward is $+10$ and the dissimilarity is about $7$ at rest.
The reported returns are never scaled.
""")
    nb.take("train", "ReturnScaler")
    nb.md(r"""
**Combined advantage.** The reward advantage is standardized and the cost advantage is centered over the rollout, so the weight of the cost does not depend on the scale of the reward.
The factor $1/(1+\lambda)$ keeps the scale fixed (`_ppo_update` standardizes the result again in every minibatch).
""")
    nb.take("train", "combine_advantages")
    nb.md(r"""
**Truncation.** `_bootstrap_truncated` adds $\gamma V_r(s_{\mathrm{final}})$ to the reward and $\gamma V_c(s_{\mathrm{final}})$ to the cost of an episode that ends at the time limit; the episode boundary stays in `dones`, so the GAE trace is still cut there.
For a distributional critic, it also stores the offsets that replace the bootstrap quantile by quantile.
""")
    nb.take("train", "_bootstrap_truncated")
    nb.md(r"""
**Rollout.** `RolloutBuffer` holds the transitions of one rollout, and `_collect_rollout` fills it.
The reward that the environment returns contains $r^{\mathsf{b}}_t+r_{t\mathbb{D}}$; `_collect_rollout` subtracts $\beta\,\mathbb{D}_t$, scales the result, and adds the bootstrap value at a truncation.
""")
    nb.take("train", "RolloutBuffer", "_collect_rollout")
    nb.md(r"""
**Multiplier and policy update.** `_update_lagrange_multiplier` is one projected Adam step on the multiplier.
`_ppo_update` runs `update_epochs` passes over the rollout in `num_minibatches` minibatches.
The actor minimizes the clipped surrogate on the standardized combined advantage, the reward critic the squared error (the quantile Huber loss for a distributional critic), and the cost critic the squared error.
The gradient norm is clipped separately for the actor, the reward critic, and the cost critic.
""")
    nb.take("train", "_update_lagrange_multiplier", "_ppo_update")
    nb.md(r"""
**Saving.** `_write_csv` writes a table with a header row.
`_save_run` writes `config.json`, `training.csv`, `episodes.csv`, and, last, `policy.pt`, so that an interrupted save leaves a run without `policy.pt`, which `run_all.py` trains again.
""")
    nb.take("train", "_write_csv", "_save_run")
    nb.md(r"""
**`train`** runs the loop.
It seeds NumPy and PyTorch with `cfg.seed`, builds the vector environment, the agent, one Adam optimizer for the three networks with a learning rate that decreases linearly to zero, and the optimizer of the multiplier.
For each of the `total_steps // (num_envs * num_steps)` updates, it performs the five steps above.
From the first completed episode on, it logs the mean statistics of the most recent episodes after every update.
The multiplier takes an Adam step at every update, also before the first episode has been completed; then $\hat J_c=0$, and the projection keeps $\lambda$ at $0$.
""")
    nb.take("train", "train")
    nb.finish("train")


# ---------------------------------------------------------------------------------------------------------------------
# examples for the PPO learner
# ---------------------------------------------------------------------------------------------------------------------
def agent_check(nb: Notebook) -> None:
    nb.md(r"""
**Networks.** The cell builds the agent of a distributional method with the 21 inputs of `hybrid` and the 6 actions, evaluates it on a batch of 5 inputs, and counts the parameters.
""")
    nb.code(r"""
agent = Agent(obs_dim=21, act_dim=6, hidden=64, init_log_std=INIT_LOG_STD, n_quantiles=N_QUANTILES)
batch = torch.randn(5, 21)
action, log_prob, entropy = agent.act(batch)
shapes = dict(action=action, log_prob=log_prob, entropy=entropy, value=agent.value(batch), quantiles=agent.quantiles(batch),
              cost_value=agent.cost_value(batch))
print({name: tuple(tensor.shape) for name, tensor in shapes.items()})
print("parameters:", sum(p.numel() for p in agent.parameters()), "   initial standard deviation of the actions:", float(agent.actor_logstd.exp()[0, 0]))
""")


def build() -> Notebook:
    nb = Notebook()
    title(nb)
    shared.setup(nb)
    shared.targets(nb)
    shared.config(nb)
    shared.envs(nb)
    agent(nb)
    train(nb)
    shared.evaluate(nb, "ppo")
    shared.metrics(nb)
    shared.analysis(nb)
    examples.start(nb, "ppo")
    examples.targets_check(nb, "ppo")
    examples.env_check(nb, "ppo")
    agent_check(nb)
    examples.quantile_check(nb, "ppo")
    examples.multiplier_check(nb, "ppo")
    examples.tiny_run(nb, "ppo")
    examples.stored(nb, "ppo")
    return nb
