"""Sections that the notebooks of the PPO and the SAC learner share: setup, targets, configuration, environment wrappers,
evaluation rollouts, metrics, analysis, and the examples that call them."""

from __future__ import annotations

import ast

from .builder import Notebook

PAPER = "*Dissimilarity-Based Markov Decision Processes for Stabilizing State Distributions*"

# the methods of each learner in the order of the tables of the paper: each method followed by its -Lag variant
METHOD_ORDER = {
    "ppo": ["PPO", "PPO-Lag", "Dist-PPO", "Dist-PPO-Lag", "Implicit-PPO", "Implicit-PPO-Lag", "DMDP", "DMDP-Lag"],
    "sac": ["SAC", "SAC-Lag", "Dist-SAC", "Dist-SAC-Lag", "Implicit-SAC", "Implicit-SAC-Lag", "DMDP", "DMDP-Lag"],
}


# ---------------------------------------------------------------------------------------------------------------------
# 1 Setup
# ---------------------------------------------------------------------------------------------------------------------
def setup(nb: Notebook) -> None:
    nb.md(r"""
## 1 Setup

The first cell postpones the evaluation of annotations, as every module of the package does, and registers the Safety-Gymnasium environments.
Gymnasium-Robotics prints a warning about its Adroit tasks while it is imported, and the cell hides it.
The sections below start with the imports of their module.
""")
    nb.code(r"""
from __future__ import annotations

import contextlib
import io

with contextlib.redirect_stderr(io.StringIO()):     # hides the warning about the Adroit tasks of Gymnasium-Robotics
    import safety_gymnasium  # noqa: F401  (registers the Safety-Gymnasium environments)
""", package=True)


# ---------------------------------------------------------------------------------------------------------------------
# 2 Target distributions
# ---------------------------------------------------------------------------------------------------------------------
def targets(nb: Notebook) -> None:
    nb.md(r"""
## 2 Target distributions (`targets.py`)

A DMDP compares the distribution of the state with a target distribution $\rho_\star$.
In one dimension, the squared 2-Wasserstein distance of two distributions is
$W_2^2(\rho,\tilde\rho)=\int_0^1\big(F^{-1}(u)-\tilde F^{-1}(u)\big)^2\,\mathrm{d}u$ with their quantile functions $F^{-1}$ and $\tilde F^{-1}$.
`targets.py` computes it exactly for the empirical distribution of a window of $n$ samples and a target: the $i$-th smallest sample is matched with the $i$-th of $n$ equal slices of the target.
The sum of this distance for the velocity and for the pitch is the window dissimilarity $\mathbb{D}_t$ of the training reward (Appendix E).
""")
    nb.docstring("targets")
    nb.imports("targets")
    nb.md(r"""
A target is a member of a location-scale family with a given mean and standard deviation.
`standard_quantile` is the quantile function of the standardized member, either Gaussian or uniform; the targets of the paper are uniform.
""")
    nb.take("targets", "TARGET_FAMILIES", "_SQRT3", "standard_quantile", merge=True)
    nb.md(r"""
`target_mean` gives the mean of a target that earlier designs derived from a cost limit; the paper does not use it.
`config.py` wraps this function under the same name.
""")
    nb.take("targets", "target_mean")
    nb.md(r"""
`Target` holds a target distribution and what the code needs from it: the quantile function, its integral (`integrated_quantile`), the weights $m_i=\int_{(i-1)/n}^{i/n}F^{-1}(u)\,\mathrm{d}u$ of `w2_weights`, the density, the tail probability $\mathbb{P}[v>v_{\mathsf{lim}}]$, and sampling.
""")
    nb.take("targets", "Target")
    nb.md(r"""
**Window dissimilarity.** For the sorted samples $x_{(1)}\leq\dots\leq x_{(n)}$, the squared distance to the target is
$W_2^2=\frac1n\sum_i x_{(i)}^2-2\sum_i x_{(i)}m_i+\mu_\star^2+\varsigma_\star^2$.
`empirical_w2_squared` evaluates it for one window, and `empirical_w2_squared_batch` for many windows at once.
Unlike the closed form for two Gaussians, it also compares the shape of the window with the target.
""")
    nb.take("targets", "empirical_w2_squared", "empirical_w2_squared_batch", merge=True)
    nb.md(r"""
A window of finitely many samples never equals $\rho_\star$, so even a window drawn from $\rho_\star$ has a positive dissimilarity.
The two functions below estimate the quantiles of this dissimilarity and the probability that it is at most a threshold (deterministic and cached).
They characterize the threshold $\mathbb{D}_{\mathsf{thr}}$ of the threshold reward and are not used in training.
""")
    nb.take("targets", "_window_quantile", "window_dissimilarity_quantile", "_window_within", "window_within_probability", merge=True)
    nb.finish("targets")


# ---------------------------------------------------------------------------------------------------------------------
# 3 Configuration
# ---------------------------------------------------------------------------------------------------------------------
def config(nb: Notebook) -> None:
    nb.md(r"""
## 3 Configuration (`config.py`)

`config.py` fixes the study: the task, the target, the evaluation protocol, the 14 methods, and the hyperparameters of both learners.
`TrainConfig` collects every setting of one run, and `make_config` builds it for a method.
The docstring below lists the notation of the paper.
""")
    nb.docstring("config")
    nb.imports("config")
    nb.md(r"""
**Constants.** `V_LIMIT` is the velocity limit $v_{\mathsf{lim}}$ of the cost $c_t=\mathbf{1}[v_t>v_{\mathsf{lim}}]$, `COST_LIMIT` is the cost limit $\bar c$ of the -Lag methods, `D_THRESHOLD` and `D_BONUS` are $\mathbb{D}_{\mathsf{thr}}$ and $d$ of the threshold reward, `TARGET_BAND` and `PITCH_HALF_WIDTH` define the target, and `NOISE_OFFSET` and `NOISE_SD` define the bimodal sensor noise (Tables Task and Hyperparameters in Appendix E).
`SAC_TOTAL_STEPS` and `SAC_NUM_ENVS` are the step budget and the number of environments of the SAC learner.
`COST_PENALTY` and `BENCHMARK_COST_LIMIT` serve the design pilot and earlier designs only: every reported method has `cost_penalty` $=0$.
""")
    nb.take("config", "EPISODE_LENGTH", "V_LIMIT", "BENCHMARK_COST_LIMIT", "COST_LIMIT", "COST_PENALTY", "D_THRESHOLD", "TARGET_BAND",
            "PITCH_HALF_WIDTH", "INIT_LOG_STD", "D_BONUS", "LEARNERS", "SAC_TOTAL_STEPS", "SAC_NUM_ENVS", "NOISE_OFFSET", "NOISE_SD",
            "N_QUANTILES", merge=True)
    nb.md(r"""
**Velocity target.** `band_target` gives the mean and standard deviation of the velocity target $U[0.70\,v_{\mathsf{lim}},0.95\,v_{\mathsf{lim}}]$: a fast distribution with a margin of 5% below the velocity limit, so the cost under $\rho_\star$ is zero.
`target_mean` and `uniform_target` give the targets of earlier designs, which were derived from a cost limit; the paper does not use them.
""")
    nb.take("config", "target_mean", "uniform_target", "band_target", merge=True)
    nb.md(r"""
**Task and protocol.** `ENVIRONMENTS` describes the HalfCheetah task: the Safety-Gymnasium id, the positions of the forward velocity and of the pitch in the observation vector, the velocity limit, the moments of the velocity target, the dissimilarity (`empirical_w2`, the exact distance of `targets.py`), and the step budget of PPO.
The sensor noise $\eta$ is drawn from $U[0,0.3]$ for every training episode (`TRAIN_NOISE_MAX`), and the evaluation uses the levels `EVAL_NOISES` with `EVAL_EPISODES` episodes each.
""")
    nb.take("config", "_UNIFORM_MU", "ENVIRONMENTS", "SEEDS", "TRAIN_NOISE", "TRAIN_NOISE_MAX", "EVAL_NOISES", "EVAL_EPISODES", merge=True)
    nb.md(r"""
**Methods.** `METHODS` lists the 14 methods by the settings in which they differ: the policy input (`state`: $\mathbf{o}_t$; `hybrid`: $(\hat\omega_t,\mathbf{o}_t)$), the cost constraint (`lagrangian`), the threshold reward (`d_bonus`) and the weight $\beta$ of the dissimilarity, and the number of quantiles of the reward critic (`n_quantiles`).
The comments in the cell define each method.
`METHOD_LABELS` and `method_label` give the names of the paper; DMDP and DMDP-Lag carry the learner in parentheses.
""")
    nb.take("config", "BETA", "METHODS", "METHOD_LABELS", "method_label", merge=True)
    nb.md(r"""
**`TrainConfig`** holds every setting of one run.
The field names are stored in `config.json` and in `policy.pt`, so they stay fixed.
The fields with the prefix `sac_` belong to the SAC learner.
`config_from_dict` reads a stored configuration, also from runs of earlier designs, and `make_config` builds the configuration of a method from `ENVIRONMENTS` and `METHODS`; keyword arguments override any field.
""")
    nb.take("config", "_HALF_CHEETAH", "TrainConfig", "config_from_dict", "make_config", merge=True)
    nb.finish("config")


# ---------------------------------------------------------------------------------------------------------------------
# 4 Environment wrappers
# ---------------------------------------------------------------------------------------------------------------------
def envs(nb: Notebook) -> None:
    nb.md(r"""
## 4 Environment wrappers (`envs.py`)

`envs.py` turns the benchmark environment into the practical DMDP interface of Section 4.5 with a stack of Gymnasium wrappers:

```text
HalfCheetah (Safety-Gymnasium) → CostPenalty → SensorNoise → DistributionEstimator → ThresholdReward → EpisodeStats
```

| Wrapper | What it does |
|---|---|
| `CostPenalty` | keeps the benchmark reward $r^{\mathsf{b}}_t=v_t-0.1\|\mathbf{a}_t\|^2$ for every reported method (the cost penalty has the weight $\kappa=0$) |
| `SensorNoise` | returns the noisy reading $\mathbf{o}_t=\mathbf{s}_t+\eta\,\boldsymbol{\xi}_t$ |
| `DistributionEstimator` | returns the policy input $(\hat\omega_t,\mathbf{o}_t)$ from a window of readings, and the window dissimilarity $\mathbb{D}_t$ from a window of true values |
| `ThresholdReward` | adds $r_{t\mathbb{D}}=d$ if $\mathbb{D}_t\leq\mathbb{D}_{\mathsf{thr}}$ |
| `EpisodeStats` | collects the return, the cost, and the statistics of the dissimilarity of every finished episode |
""")
    nb.docstring("envs")
    nb.imports("envs")
    nb.md(r"""
**Options.** The policy input is either `state` ($\mathbf{o}_t$) or `hybrid` ($(\hat\omega_t,\mathbf{o}_t)$).
The dissimilarity is either the exact empirical distance of `targets.py` (`empirical_w2`, used by every reported method) or the closed form for two Gaussians evaluated at the window mean and standard deviation (`gaussian_w2`, with `gaussian_w2_squared`).
""")
    nb.take("envs", "POLICY_INPUTS", "DISSIMILARITIES", "gaussian_w2_squared", merge=True)
    nb.md(r"""
**`CostPenalty`** subtracts a cost penalty from the reward and keeps the benchmark reward in `info["benchmark_reward"]`.
It is the identity on the reward for $\kappa=0$.
""")
    nb.take("envs", "CostPenalty")
    nb.md(r"""
**`SensorNoise`** adds the noise $\eta\,\boldsymbol{\xi}_t$ to every channel of the observation.
The entries of $\boldsymbol{\xi}_t$ are drawn independently from the bimodal mixture $\tfrac12\mathcal{N}(-0.8,0.6^2)+\tfrac12\mathcal{N}(0.8,0.6^2)$, which has mean $0$ and variance $1$, so $\eta$ is the standard deviation of the noise.
In training, $\eta$ is drawn from $U[0,0.3]$ at every reset (`nu_max`); in the evaluation, it is fixed.
""")
    nb.take("envs", "SensorNoise")
    nb.md(r"""
**`DistributionEstimator`** is the core of the interface.
It keeps four sliding windows of $n_{\mathsf{w}}=10$ steps: the noisy readings of the velocity and of the pitch, which the policy sees, and the true velocities and pitch angles, which only the training dissimilarity uses.
With the `hybrid` input, the policy receives the window means and standard deviations $\hat\omega_t$ of the two readings, followed by the 17 entries of $\mathbf{o}_t$ (21 inputs).
`_dissimilarity` computes $\mathbb{D}_t$, the sum of the exact squared 2-Wasserstein distances of the true velocity window and of the true pitch window to their targets.
At the start of an episode, the windows of readings are filled with the first reading, and the windows of true values contain only the steps taken so far.
The wrapper reports $\mathbb{D}_t$ and its two terms, $\mathbb{D}_{\mathsf{thr}}$, and the pitch angle in `info`.
""")
    nb.take("envs", "wrap_angle", "DistributionEstimator")
    nb.md(r"""
**`ThresholdReward`** adds the threshold reward $r_{t\mathbb{D}}=d$ if $\mathbb{D}_t\leq\mathbb{D}_{\mathsf{thr}}$ and $0$ otherwise (with $d=0$ for the baselines), and keeps it in `info["r_tD"]`.
**`EpisodeStats`** accumulates the return with and without $r_{t\mathbb{D}}$, the cost, and the dissimilarity over an episode, and reports them when the episode ends.
""")
    nb.take("envs", "ThresholdReward", "EpisodeStats")
    nb.md(r"""
**Constructors.** `make_env` stacks the wrappers for one environment, and `make_vector_env` creates the `num_envs` environments of a training run with the seeds $1000\cdot\text{seed}+i$.
`env_settings` lists the settings of a `TrainConfig` that `make_env` takes.
""")
    nb.take("envs", "env_settings", "make_env", "make_vector_env", merge=True)
    nb.finish("envs")


# ---------------------------------------------------------------------------------------------------------------------
# 7 Evaluation rollouts
# ---------------------------------------------------------------------------------------------------------------------
def _load_policy_cell(nb: Notebook, learner: str) -> str:
    """Source of ``evaluate.load_policy`` for one learner: the branch of the other learner is left out."""
    mod = nb.module("evaluate")
    item = next(it for it in mod.items if it.names == ("load_policy",))
    fn = item.node
    iff = next(n for n in fn.body if isinstance(n, ast.If))
    head = mod.lines[item.start - 1:iff.lineno - 1]                            # def line up to the statement before the if
    if learner == "ppo":
        body = mod.lines[iff.end_lineno:fn.end_lineno]                        # the statements after the if
        note = "# The package function also loads the checkpoints of the SAC learner (the branch of `if cfg.learner == \"sac\"`);\n" \
               "# DMDP_SAC.ipynb shows that branch."
    else:
        first = iff.body[1].lineno                                             # after the lazy import of the SAC classes
        body = [line[4:] if line.startswith("    ") else line for line in mod.lines[first - 1:iff.end_lineno]]
        head = [head[0].replace("tuple[Agent, RunningMeanStd, TrainConfig]", "tuple[SACAgent, IdentityNormalizer, TrainConfig]")] + head[1:]
        note = "# The package function also loads the checkpoints of the PPO learner (the statements after the `if`);\n" \
               "# DMDP_PPO.ipynb shows that branch."
    return note + "\n" + "\n".join(head + body)


def evaluate(nb: Notebook, learner: str) -> None:
    nb.md(r"""
## 7 Evaluation rollouts (`evaluate.py`)

`evaluate.py` runs a trained policy with deterministic actions, the mean action, for `EVAL_EPISODES` episodes at every noise level.
At every step, it records the true velocity, the reward (the training reward without $-\beta\,\mathbb{D}_t$), the cost, the window dissimilarity, the threshold reward, and the pitch of the torso.
Entries after the end of an episode are NaN.
Episode $j$ of every run starts with the seed `EVAL_SEED` $+\,7919\,j$, so all methods are evaluated with the same initial states and the same noise sequences.
The arrays are stored in `evaluation.npz`.
""")
    nb.docstring("evaluate")
    nb.imports("evaluate")
    nb.take("evaluate", "EVAL_SEED", "RECORDED", merge=True)
    other = "SAC" if learner == "ppo" else "PPO"
    nb.md(rf"""
`load_policy` reads a `policy.pt` and returns the agent, the normalizer of the observations, and the configuration of the run.
The package function reads the checkpoints of both learners; this notebook shows the branch of the {learner.upper()} learner, and `DMDP_{other}.ipynb` shows the other one.
""")
    nb.code(_load_policy_cell(nb, learner), package=True)
    nb.skip("evaluate", "load_policy")
    nb.md(r"""
`rollout` steps one environment per episode and evaluates the policy for all of them in one batch, and `evaluate_policy` repeats it for every noise level and stores the arrays.
Noise levels that are already stored are not evaluated again.
""")
    nb.take("evaluate", "rollout", "evaluate_policy")
    nb.finish("evaluate")


# ---------------------------------------------------------------------------------------------------------------------
# 8 Metrics
# ---------------------------------------------------------------------------------------------------------------------
def metrics(nb: Notebook) -> None:
    nb.md(r"""
## 8 Metrics (`metrics.py`)

`metrics.py` measures distributional stability from the evaluation episodes (Section 5).
$\rho_k$ is the distribution of the velocity at step $k$ across the episodes, and $\hat\rho$ is the stable distribution, pooled over the steps $k\geq k_0=500$.
The stability distance $\mathbb{D}(\rho_k\|\hat\rho)$ is the 1-Wasserstein distance, which is exact in one dimension.
To keep it from being zero by construction, $\hat\rho$ and $\rho_k$ come from disjoint halves of the episodes (cross-fitting), and the two assignments are averaged.
""")
    nb.docstring("metrics")
    nb.imports("metrics")
    nb.md(r"""
**1-Wasserstein distance.** `Reference` prepares a fixed reference sample, and `wasserstein_1` computes the exact distance of a sample to it in $O(n\log m)$.
""")
    nb.take("metrics", "Reference", "wasserstein_1", merge=True)
    nb.md(r"""
**Stability curve.** `split_halves` splits the episodes, `estimate_rho_hat` pools the late steps, and `stability_curve` returns $\mathbb{D}(\rho_k\|\hat\rho)$ at every step or for blocks of steps.
`_cross_fitted_blocks` does the cross-fitting.
""")
    nb.take("metrics", "split_halves", "estimate_rho_hat", "stability_curve", "_cross_fitted_blocks", merge=True)
    nb.md(r"""
**Stationary reference value.** Because $\rho_k$ and $\hat\rho$ are estimated from finitely many samples, $\mathbb{D}(\rho_k\|\hat\rho)$ stays positive even for a stationary closed loop.
`sampling_floor` computes this value from surrogates in which every episode is shifted circularly in time by an independent random amount (Appendix F).
""")
    nb.take("metrics", "sampling_floor")
    nb.md(r"""
**Target distance and reliability.** `distance_to_target` is $\mathbb{D}(\hat\rho\|\rho^{v}_\star)$, with $\rho_\star$ represented by its midpoint quantiles.
`exceedance_probability` is $\mathbb{P}[R\geq R_{\mathsf{Thr}}]$, the fraction of episodic returns that reach a threshold.
""")
    nb.take("metrics", "distance_to_target", "exceedance_probability", merge=True)
    nb.finish("metrics")


# ---------------------------------------------------------------------------------------------------------------------
# 9 Analysis
# ---------------------------------------------------------------------------------------------------------------------
def analysis(nb: Notebook) -> None:
    nb.md(r"""
## 9 Analysis (`analysis.py`)

`analysis.py` turns the stored files of a run (`config.json`, `training.csv`, `episodes.csv`, `policy.pt`, `evaluation.npz`) into the quantities of the tables of the paper.
`summarize_run` returns one row per noise level: the returns, the cost, the stability distances, the target distance, the fraction of steps within $\mathbb{D}_{\mathsf{thr}}$, and the posture of the torso.
`summarize` collects the runs of several methods and seeds, and `aggregate` gives the mean and the standard error over the seeds.
""")
    nb.docstring("analysis")
    nb.imports("analysis")
    nb.md(r"""
**Stored runs.** A run is stored in `<root>/<env>/<method>/seed<s>/`; the functions below read its files.
""")
    nb.take("analysis", "Evaluation", "run_dir", "load_evaluation", "load_config", "_read_csv", "load_training", "load_episodes", merge=True)
    nb.md(r"""
**Returns.** Every return is the sum over the $H=1000$ steps of an episode, and none of them contains $-\beta\,\mathbb{D}_t$ or $-\lambda c_t$.
`episode_returns` is $R$, the return of the reward on which the method is trained: $r^{\mathsf{b}}_t$ for the baselines and $r^{\mathsf{b}}_t+r_{t\mathbb{D}}$ for the DMDP methods; the training curves and the reliability of the return use it.
`benchmark_returns` is the return of $r^{\mathsf{b}}_t=v_t-0.1\|\mathbf{a}_t\|^2$ for every method.
""")
    nb.take("analysis", "penalty_per_cost", "episode_returns", "benchmark_returns", merge=True)
    nb.md(r"""
**Threshold reward from the record.** `dissimilarity_threshold` gives the threshold $\mathbb{D}_{\mathsf{thr}}$ of a run, and `threshold_reward` recomputes $r_{t\mathbb{D}}$ from the recorded dissimilarity for any method.
Runs without the threshold reward use the current threshold, so the fraction of steps within it, and the evaluation reward below, are measured alike for every method.
""")
    nb.take("analysis", "dissimilarity_threshold", "threshold_reward", merge=True)
    nb.md(r"""
**Evaluation reward.** The evaluation tables report the return of the evaluation reward, which is the same for every method:

$$
r^{\mathsf{eva}}_t=\begin{cases} r^{\mathsf{b}}_t+r_{t\mathbb{D}}, & v_t\leq v_{\mathsf{lim}},\\ -r_{\mathsf{pen}}, & v_t>v_{\mathsf{lim}},\end{cases}
\qquad r_{\mathsf{pen}}=v_{\mathsf{lim}}.
$$

The threshold reward $r_{t\mathbb{D}}$ is recomputed from the recorded window dissimilarity, also for the baselines, which are trained without it.
The evaluation reward credits only safe steps, with their benchmark reward and with $r_{t\mathbb{D}}$ when the window is close to the target, whereas $r^{\mathsf{b}}_t$ grows with the velocity and thus rewards violating the limit.
`evaluation_returns` is its return.
""")
    nb.take("analysis", "evaluation_penalty", "evaluation_rewards", "evaluation_returns", merge=True)
    nb.md(r"""
**Summary of a run.** `summarize_run` computes, for every noise level, the three returns above, the cost, $\mathbb{D}(\rho_k\|\hat\rho)$ averaged over $k<200$ (`D_early`) and $k\geq500$ (`D_late`), the target distance $\mathbb{D}(\hat\rho\|\rho^{v}_\star)$ (`D_target`), the shift of $\hat\rho$ caused by noise (`D_shift`), the fraction of steps with $\mathbb{D}_t\leq\mathbb{D}_{\mathsf{thr}}$ (`within`), and the pitch statistics (`pitch`, `flipped`).
A torso is flipped (upside down) at a step with $|\theta_t|>\pi/2$.
""")
    nb.take("analysis", "_finite_fraction", "_pitch_summary", "summarize_run", "summarize", "aggregate", merge=True)
    nb.finish("analysis")
