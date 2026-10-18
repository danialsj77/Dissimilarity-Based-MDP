"""The notebook DMDP_SAC.ipynb: the complete code of the SAC learner."""

from __future__ import annotations

from . import examples, shared
from .builder import Notebook, used_names

FILE = "DMDP_SAC.ipynb"

# the helpers of the PPO modules that sac.py imports, in the order of their modules (checked against sac.py)
PPO_HELPERS = {
    "agent": ["RunningMeanStd", "quantile_levels", "quantile_huber_loss"],
    "train": ["EPISODES_PER_ENV", "_info_at_step", "_completed_episodes", "_update_lagrange_multiplier", "_write_csv"],
}


def title(nb: Notebook) -> None:
    nb.md(r"""
# DMDP with SAC: the code

This notebook shows the complete code of the SAC learner of the experiments in
""" + shared.PAPER + r"""
(Section 5.1 and Appendix E of the paper): the target, the configuration, the environment wrappers, the soft actor-critic learner and its Lagrangian variant, the evaluation, and the metrics.
`DMDP_PPO.ipynb` shows the PPO learner in the same way.

**How the notebook is made.** `build_code_notebooks.py` copies the code cells without change from the package `dmdp_sg/`, one section per module, so the notebook shows the code that produced the stored results; `tests/test_code_notebooks.py` checks this.
The notebook does not import `dmdp_sg`: its cells define everything in one namespace, in the order of the dependencies between the modules.
Two things differ from the files of the package: the imports between the modules are commented out, because their names are defined in the cells above, and `load_policy` shows the branch for the SAC learner only.
SAC takes a few helpers from the modules of the PPO learner; Section 5 shows them.

**Methods trained with SAC.** Every method maximizes the return of its training reward with the same learner, and the methods differ only in the policy input, the training reward, the reward critics, and the cost constraint.

| Method | Policy input | Training reward | Reward critics | Cost constraint |
|---|---|---|---|---|
| SAC | $\mathbf{o}_t$ | $r^{\mathsf{b}}_t$ | expected return | no |
| SAC-Lag | $\mathbf{o}_t$ | $r^{\mathsf{b}}_t-\lambda c_t$ | expected return | yes |
| Distributional SAC (`Dist-SAC`) | $\mathbf{o}_t$ | $r^{\mathsf{b}}_t$ | 32 quantiles of the return | no |
| Distributional SAC-Lag (`Dist-SAC-Lag`) | $\mathbf{o}_t$ | $r^{\mathsf{b}}_t-\lambda c_t$ | 32 quantiles of the return | yes |
| Implicit DMDP (SAC) (`Implicit-SAC`) | $\hat\omega_t$ and $\mathbf{o}_t$ | $r^{\mathsf{b}}_t$ | 32 quantiles of the return | no |
| Implicit DMDP-Lag (SAC) (`Implicit-SAC-Lag`) | $\hat\omega_t$ and $\mathbf{o}_t$ | $r^{\mathsf{b}}_t-\lambda c_t$ | 32 quantiles of the return | yes |
| DMDP (SAC) (`DMDP`) | $\hat\omega_t$ and $\mathbf{o}_t$ | $r^{\mathsf{b}}_t+r_{t\mathbb{D}}-\beta\,\mathbb{D}_t$ | expected return | no |
| DMDP-Lag (SAC) (`DMDP-Lag`) | $\hat\omega_t$ and $\mathbf{o}_t$ | $r^{\mathsf{b}}_t+r_{t\mathbb{D}}-\beta\,\mathbb{D}_t-\lambda c_t$ | expected return | yes |

Here $r^{\mathsf{b}}_t=v_t-0.1\|\mathbf{a}_t\|^2$ is the benchmark reward, $c_t=\mathbf{1}[v_t>v_{\mathsf{lim}}]$ the safety cost, $\hat\omega_t$ the window statistics of the noisy readings, $\mathbb{D}_t$ the window dissimilarity to the target, and $r_{t\mathbb{D}}$ the threshold reward.
The docstring of `config.py` lists the notation.

**Sections.** 1 Setup · 2 Target distributions · 3 Configuration · 4 Environment wrappers · 5 Helpers of the PPO learner · 6 SAC learner · 7 Evaluation rollouts · 8 Metrics · 9 Analysis · 10 Examples and checks · 11 Stored results.

**Running it.** Run the cells from top to bottom in the environment of `environment.yml`; the whole notebook takes less than a minute.
Sections 1 to 9 define the code, Section 10 calls it with small settings, and Section 11 reads the stored runs in `results/sac`.
The full runs of the paper are launched with `run_all.py` (see the README).
""")


# ---------------------------------------------------------------------------------------------------------------------
# 5 Helpers of the PPO learner
# ---------------------------------------------------------------------------------------------------------------------
def helpers(nb: Notebook) -> None:
    sac = nb.module("sac")
    imported = sac.relative_imports()
    for name, wanted in PPO_HELPERS.items():
        mod = nb.module(name)
        found = [mod.items[i].names[0] for i in mod.closure(imported[name])]
        if found != wanted:
            raise ValueError(f"sac.py imports {imported[name]} from {name}.py, which needs {found}, not {wanted}")
    nb.md(r"""
## 5 Helpers of the PPO learner (`agent.py`, `train.py`)

`sac.py` reuses a few functions of the PPO learner: from `agent.py`, the running mean and variance estimator and the quantile-regression loss of the distributional critics, and from `train.py`, the bookkeeping of the training loop and the update of the Lagrange multiplier.
The cells show only these functions; `DMDP_PPO.ipynb` shows `agent.py` and `train.py` in full.
""")
    agent, train = nb.module("agent"), nb.module("train")
    nb.imports("agent", used_names([agent.items[i].node for i in agent.closure(imported["agent"])]))
    nb.md(r"""
**`agent.py`.** `RunningMeanStd` keeps a running mean and variance; the SAC learner uses it for the scale of the reward (`ReturnStatistics`).
The quantile levels are the midpoints $\tau_i=(2i-1)/(2n)$ of $n$ equal bins, and `quantile_huber_loss` is the quantile-regression loss of Dabney et al. (2018) with the Huber threshold $\kappa=1$: each output $\theta_i$ of a distributional critic learns the $\tau_i$-quantile of the target distribution.
""")
    nb.take("agent", "RunningMeanStd")
    nb.take("agent", "quantile_levels", "quantile_huber_loss", merge=True)
    nb.imports("train", used_names([train.items[i].node for i in train.closure(imported["train"])]))
    nb.md(r"""
**`train.py`.** The logged statistics and the estimate $\hat J_c$ of the episodic cost use the last $4\times$`num_envs` completed episodes (`EPISODES_PER_ENV`).
When an episode ends, the vector environment resets it and stores the info of its last step in `final_info`; `_info_at_step` reads a value of the current step from there for the finished environments, and `_completed_episodes` collects the statistics of the finished episodes (`EpisodeStats`).
`_update_lagrange_multiplier` is one Adam step on $-\lambda(\hat J_c-\bar c)$ followed by the projection onto $\lambda\geq0$, and `_write_csv` writes a table with a header row.
""")
    nb.take("train", "EPISODES_PER_ENV", "_info_at_step", "_completed_episodes", merge=True)
    nb.take("train", "_update_lagrange_multiplier")
    nb.take("train", "_write_csv")


# ---------------------------------------------------------------------------------------------------------------------
# 6 SAC learner
# ---------------------------------------------------------------------------------------------------------------------
def learner(nb: Notebook) -> None:
    nb.md(r"""
## 6 SAC learner (`sac.py`)

`sac.py` is the second learner of the study.
Every method maximizes the same training reward as with PPO, $\tilde r_t=r^{\mathsf{b}}_t+r_{t\mathbb{D}}-\beta\,\mathbb{D}_t$, and only the learner differs.
The learner is soft actor-critic (Haarnoja et al., 2018) with twin reward critics $Q_1$ and $Q_2$ and target networks updated by Polyak averaging, a tanh-squashed Gaussian policy, a replay buffer, and an entropy temperature $\alpha$ that is tuned to the target entropy $-\dim\mathbf{a}=-6$.
The -Lag methods also learn a cost critic $Q_c$ and enforce $J_c\leq\bar c$ with a multiplier $\lambda\geq0$ as in SAC-Lagrangian (Ha et al., 2020): the actor minimizes $\big(\alpha\log\pi(a|s)-\min_i Q_i(s,a)+\lambda Q_c(s,a)\big)/(1+\lambda)$.
The multiplier is updated every `sac_lambda_interval` environment steps with the rule of the PPO learner.
Distributional SAC and Implicit DMDP (SAC) give $Q_1$ and $Q_2$ 32 quantile outputs each, learned with the quantile Huber loss.

One iteration of the loop steps the `num_envs` environments once, stores the transitions in the replay buffer, and then takes `num_envs` gradient steps, one per environment step.
The first `sac_learning_starts` steps use uniformly random actions and no gradient steps.
""")
    nb.docstring("sac")
    nb.imports("sac")
    nb.md(r"""
**Networks.** `_mlp` builds a network with two hidden ReLU layers of width `sac_hidden` $=256$.
`QNetwork` is $Q(s,a)$, or the quantiles of the return given $(s,a)$ for a distributional critic.
`SACActor` outputs the mean and the log standard deviation of a Gaussian (the log standard deviation is limited to $[-5,2]$); `sample` draws the reparameterized action $\tanh(u)$, scaled to the action bounds, with its log-probability corrected for the squashing, and `mean_action` is the deterministic action of the evaluation.
""")
    nb.take("sac", "LOG_STD_MIN", "_mlp", "QNetwork", "SACActor", merge=True)
    nb.md(r"""
**`SACAgent`** holds the actor, the twin reward critics `q1` and `q2`, the cost critic `qc`, their target networks, and $\log\alpha$.
`act` has the signature of `Agent.act` of the PPO learner, so `evaluate.rollout` runs both learners.
`IdentityNormalizer` stands in for `RunningMeanStd` in the evaluation: SAC uses the observations as they are.
""")
    nb.take("sac", "SACAgent", "IdentityNormalizer", merge=True)
    nb.md(r"""
**Reward scale.** `ReturnStatistics` tracks the running standard deviation of the discounted return of the training reward, as `ReturnScaler` does for PPO.
The replay buffer stores the unscaled reward, and the division happens when a minibatch is sampled, so all stored transitions are scaled alike; the scaled reward is clipped to $[-10,10]$.
""")
    nb.take("sac", "ReturnStatistics")
    nb.md(r"""
**Replay buffer.** `ReplayBuffer` stores $(s,a,\tilde r,c,\text{terminated},s')$ and samples uniformly.
`_next_observations` returns the final observation of an episode that has just ended, and not the reset observation, as $s'$: episodes end at the time limit by truncation, so the critic targets bootstrap from the final observation.
""")
    nb.take("sac", "ReplayBuffer", "_next_observations", merge=True)
    nb.md(r"""
**Gradient step.** `sac_update` takes one gradient step on a minibatch in four parts: the critics, the cost critic (-Lag methods), the actor, and the temperature.
The target of the critics is $\tilde r+\gamma(1-\text{terminated})\big(\min_i Q_i^{\text{target}}(s',a')-\alpha\log\pi(a'|s')\big)$ with $a'\sim\pi(\cdot|s')$.
For a distributional critic, the target samples are the quantiles of the target critic with the smaller mean, and the actor uses the means of the quantiles.
`_polyak` then moves the target networks toward the critics with the coefficient `sac_tau`.
""")
    nb.take("sac", "_polyak", "sac_update", merge=True)
    nb.md(r"""
**Saving.** `_save_run` writes `config.json`, `training.csv`, `episodes.csv`, and, last, `policy.pt`.
It differs from `_save_run` of the PPO learner in what `policy.pt` holds: SAC has no observation normalizer.
""")
    nb.take("sac", "_save_run")
    nb.md(r"""
**`train_sac`** runs the loop.
It builds the vector environment, the agent, one Adam optimizer each for the reward critics, the cost critic, the actor, and the temperature, the replay buffer, and the optimizer of the multiplier.
Each iteration takes uniformly random actions during the first `sac_learning_starts` steps and the actions of the policy afterwards, stores the transitions, and updates the multiplier when `sac_lambda_interval` steps have passed since its last update (also before the first episode has been completed; then $\hat J_c=0$ and the projection keeps $\lambda$ at $0$).
After the learning starts, it takes `num_envs` gradient steps per iteration with `sac_update`.
Each time episodes have been completed, it logs the mean statistics of the most recent episodes and the temperature $\alpha$.
""")
    nb.take("sac", "train_sac")
    nb.finish("sac")


# ---------------------------------------------------------------------------------------------------------------------
# examples for the SAC learner
# ---------------------------------------------------------------------------------------------------------------------
def agent_check(nb: Notebook) -> None:
    nb.md(r"""
**Networks.** The cell builds the agent of a distributional method with the 21 inputs of `hybrid` and the 6 actions, evaluates it on a batch of 5 inputs, and counts the parameters.
The sampled actions lie within the action bounds $[-1,1]$.
""")
    nb.code(r"""
bound = np.ones(6, np.float32)
agent = SACAgent(obs_dim=21, act_dim=6, hidden=256, act_low=-bound, act_high=bound, n_quantiles=N_QUANTILES)
batch = torch.randn(5, 21)
action, log_prob, _ = agent.act(batch)
print("action", tuple(action.shape), " log_prob", tuple(log_prob.shape), " q1", tuple(agent.q1(batch, action).shape),
      " qc", tuple(agent.qc(batch, action).shape))
print("parameters:", sum(p.numel() for p in agent.parameters()), "   largest |action|:", round(float(action.abs().max()), 3))
""")


def build() -> Notebook:
    nb = Notebook()
    title(nb)
    shared.setup(nb)
    shared.targets(nb)
    shared.config(nb)
    shared.envs(nb)
    helpers(nb)
    learner(nb)
    shared.evaluate(nb, "sac")
    shared.metrics(nb)
    shared.analysis(nb)
    examples.start(nb, "sac")
    examples.targets_check(nb, "sac")
    examples.env_check(nb, "sac")
    agent_check(nb)
    examples.quantile_check(nb, "sac")
    examples.multiplier_check(nb, "sac")
    examples.tiny_run(nb, "sac")
    examples.stored(nb, "sac")
    return nb
