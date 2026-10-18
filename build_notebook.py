"""Builds DMDP_SafetyGymnasium.ipynb, the notebook that reproduces Section 5 and Appendix F of the paper.

Usage:
    python build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace DMDP_SafetyGymnasium.ipynb

The notebook reads the stored results in results/ and writes every figure (PDF and PNG) and LaTeX
table to figures/ (or to the directory given by the environment variable DMDP_FIG_DIR).
"""

from __future__ import annotations

import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "DMDP_SafetyGymnasium.ipynb")

CELLS: list = []


def md(source: str) -> None:
    CELLS.append(nbf.v4.new_markdown_cell(source.strip("\n"), id=f"cell-{len(CELLS):03d}"))     # fixed ids: the build is reproducible


def code(source: str) -> None:
    CELLS.append(nbf.v4.new_code_cell(source.strip("\n"), id=f"cell-{len(CELLS):03d}"))


md(r"""
# DMDP on Safety-Gymnasium

This notebook reproduces the figures and tables of Section 5 (Empirical Validation) and Appendix F (Additional Results) of
*Dissimilarity-Based Markov Decision Processes for Stabilizing State Distributions* from the stored results in `results/`.
The implementation that is evaluated is described in Section 4.5 (Practical Implementation) and Appendix E (Implementation Details).
The theory of Sections 2–4 is not implemented: no Lyapunov certificate of Theorems 2–5 is computed.

**What is evaluated.** The experiments measure empirically whether the distribution $\rho_k$ of the closed loop settles at a stable distribution
and how close that distribution comes to the target $\rho_\star$:

* **Distributional stability** — the dissimilarity $\mathbb{D}(\rho_k\|\hat{\rho})$ between the distribution $\rho_k$ of the forward velocity at step $k$
  (across evaluation episodes) and a stable distribution $\hat{\rho}$ estimated from the steps $k\geq k_0$ of *other* evaluation episodes.
  Because the training objective has several components (task reward, safety cost, dissimilarity), $\hat{\rho}$ need not equal $\rho_\star$;
  $\mathbb{D}(\hat{\rho}\|\rho^{v}_\star)$ is therefore reported separately.
* **Reward and safety** — the episodic return of the evaluation reward $r^{\mathsf{eva}}_t$, which is the same for every method (section 2), and the episodic cost. The evaluation reward equals $r^{\mathsf{b}}_t+r_{t\mathbb{D}}$ at a safe step ($v_t\leq v_{\mathsf{lim}}$), with the benchmark reward $r^{\mathsf{b}}_t=v_t-0.1\|\boldsymbol{\mathrm{a}}_t\|^2$ and the threshold reward $r_{t\mathbb{D}}$ computed from the window dissimilarity for every method, and $-v_{\mathsf{lim}}$ at an unsafe step ($v_t>v_{\mathsf{lim}}$).
* **Reliability of the return** — the probability $\mathbb{P}[R\geq x]$ that the return $R$ of the reward each method is trained on, without $-\beta\,\mathbb{D}_t$, reaches $x$ in an episode, and the 10–90% range of $R$ (Section 5.1 and Appendix F.2).
* **Sensor noise** — readings $\boldsymbol{\mathrm{o}}_t=\boldsymbol{\mathrm{s}}_t+\eta\boldsymbol{\xi}_t$, where the entries of $\boldsymbol{\xi}_t$ are independent draws from the bimodal Gaussian mixture $\tfrac12\mathcal{N}(-0.8,0.6^2)+\tfrac12\mathcal{N}(0.8,0.6^2)$ with mean $0$ and variance $1$ (Appendix E); robustness is an additional property, not the motivation.
* **Posture** — the benchmark does not penalize running with a flipped torso (upside down, $|\theta_t|>\pi/2$), so the fraction of flipped steps is reported for every method.

**Methods.** PPO is trained on the benchmark reward $r^{\mathsf{b}}_t=v_t-0.1\|\boldsymbol{\mathrm{a}}_t\|^2$ and PPO-Lag on $r^{\mathsf{b}}_t-\lambda c_t$ (likewise SAC and SAC-Lag). DMDP is trained on $r^{\mathsf{b}}_t+r_{t\mathbb{D}}-\beta\,\mathbb{D}_t$ and DMDP-Lag
on $r^{\mathsf{b}}_t+r_{t\mathbb{D}}-\beta\,\mathbb{D}_t-\lambda c_t$, where $\mathbb{D}_t$ is the window dissimilarity and $r_{t\mathbb{D}}$ the threshold reward (Section 4.5 and Appendix E; symbols in section 2); their policy is $\phi(\hat{\omega}_t,\boldsymbol{\mathrm{o}}_t)$.
Distributional PPO and Distributional SAC are the baselines from distributional reinforcement learning: PPO and SAC, trained like them on $r^{\mathsf{b}}_t$ without a cost constraint and with the policy $\pi(\boldsymbol{\mathrm{o}}_t)$,
whose reward critic (with SAC, both reward critics) learns `N_QUANTILES` quantiles of the distribution of the return by quantile regression; the policy uses the mean of these quantiles.
Implicit DMDP (PPO) and Implicit DMDP (SAC) are Distributional PPO and SAC with the policy input of the DMDP methods, $\phi(\hat{\omega}_t,\boldsymbol{\mathrm{o}}_t)$:
distribution feedback with the benchmark reward and the distributional critic, but without a target $\rho_\star$ (no $r_{t\mathbb{D}}$, no $\mathbb{D}_t$) and without a cost constraint.
Distributional PPO-Lag, Distributional SAC-Lag, Implicit DMDP-Lag (PPO), and Implicit DMDP-Lag (SAC) add the cost constraint of PPO-Lag and SAC-Lag to these four baselines (reward $r^{\mathsf{b}}_t-\lambda c_t$).
The cost limit of the constrained methods is $\bar c$ (`COST_LIMIT`).

**Outputs.** Figures (PDF and PNG) and LaTeX tables are written to `figures/`, or to the directory given by the environment variable `DMDP_FIG_DIR`.
Figure text is typeset with LaTeX (Latin Modern, as in the paper) when a `latex` executable is found, and with matplotlib's mathtext otherwise.

**Structure.** 1 Setup · 2 Notation and configuration · 3 Training (optional) · 4 Evaluation summary ·
5 Figures (5.4 training curves, 5.5 reliability of the return) · 6 Tables · 7 Sanity checks · 8 Second learner: SAC.
Sections 4–7 analyze the main study with the PPO learner (`results/main`); section 8 repeats the figures and tables for the eight methods trained with SAC (`results/sac`).
Run times are given in the README.
""")

md("## 1 Setup")
code(r"""
import os
import shutil
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.abspath(".")
sys.path.insert(0, HERE)

from dmdp_sg.analysis import (aggregate, benchmark_returns, episode_returns, evaluation_returns, load_config, load_episodes, load_evaluation,
                              load_training, summarize, threshold_reward)
from dmdp_sg.config import (BETA, COST_LIMIT, COST_PENALTY, D_BONUS, D_THRESHOLD, ENVIRONMENTS, EPISODE_LENGTH, EVAL_NOISES,
                             METHODS, PITCH_HALF_WIDTH, TARGET_BAND, TRAIN_NOISE_MAX, make_config, method_label)
from dmdp_sg.metrics import estimate_rho_hat, exceedance_probability, sampling_floor, stability_curve
from dmdp_sg.targets import Target
from dmdp_sg.train import EPISODES_PER_ENV

ENV = "HalfCheetah"
TAG = "main"                                   # results/<TAG>/<ENV>/<method>/seed<s>
ROOT = os.path.join(HERE, "results", TAG)
FIG_DIR = os.path.abspath(os.environ.get("DMDP_FIG_DIR", os.path.join(HERE, "figures")))
os.makedirs(FIG_DIR, exist_ok=True)

TASK = ["PPO", "PPO-Lag", "Dist-PPO", "Dist-PPO-Lag", "Implicit-PPO", "Implicit-PPO-Lag", "DMDP", "DMDP-Lag"]   # the eight methods (Appendix F), each followed by its -Lag variant
STABLE = TASK                                                         # stable distributions (Figure stable_distributions)
SEEDS = [0, 1, 2, 3, 4]
K0, BLOCK, K_EARLY = 500, 1, 200                                     # rho_hat from k >= K0; rho_k at every single step
R_THR = 5000.0                                                       # return threshold R_Thr (paper Section 5, notebook 5.5): average reward 5 per step
LATE_FRACTION = 0.1                                                  # end of training: the last 10% of the training steps (notebook 5.4, 5.5)
SAC_ROOT = os.path.join(HERE, "results", "sac")                      # second learner (paper Section 5.1, notebook section 8)
SAC_TASK = ["SAC", "SAC-Lag", "Dist-SAC", "Dist-SAC-Lag", "Implicit-SAC", "Implicit-SAC-Lag", "DMDP", "DMDP-Lag"]   # with SAC (results/sac/<ENV>/<method>/seed<s>), in the same order
SAC_STABLE = SAC_TASK
CURVE_NUS = [0.0, 0.1, 0.3, 0.5]                                     # noise levels of the stability curves (Figures stability_curves, sac_stability_curves)
PILOT_SEEDS = [10, 11, 12]                                             # seeds of the design pilot (not used in the reported study)
# design pilot (folder under results/pilot/, method, beta, cost limit)
PILOT_SETTINGS = [("beta0.5_c10", "DMDP-Lag", 0.5, 10.0), ("beta1_c10", "DMDP-Lag", 1.0, 10.0), ("beta2_c10", "DMDP-Lag", 2.0, 10.0),
                  ("beta1_c5", "DMDP-Lag", 1.0, 5.0), ("beta1_c2.5", "DMDP-Lag", 1.0, 2.5),
                  ("beta0.5_c5", "DMDP-Lag", 0.5, 5.0), ("beta0.5_c2.5", "DMDP-Lag", 0.5, 2.5)]
PILOT_BASELINES = ("baselines_c10", ["PPO", "PPO-Lag"])                # PPO and PPO-Lag (c_bar = 10, with the cost penalty of the pilot) on the pilot seeds: reference of the rule for c_bar
PILOT_BASELINES_BENCHMARK = "baselines_c10_benchmark_reward"            # the same runs with the benchmark reward of the reported baselines

# Figure text: LaTeX with Latin Modern (as in the paper) if a latex executable is available, mathtext otherwise.
TEX_BIN = "/Library/TeX/texbin"                                      # MacTeX
if os.path.isdir(TEX_BIN) and TEX_BIN not in os.environ["PATH"].split(os.pathsep):
    os.environ["PATH"] = TEX_BIN + os.pathsep + os.environ["PATH"]
USE_TEX = shutil.which("latex") is not None
mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
if USE_TEX:
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
    })
else:
    mpl.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})
print("figure text   :", "LaTeX" if USE_TEX else "matplotlib mathtext (no latex executable found)")
print("output folder :", os.path.relpath(FIG_DIR, HERE))

# One hue per family of methods and a dashed line for the variants with the cost constraint (-Lag), so that the eight methods of
# a learner stay apart: the four hues pass the color-vision checks for every pair (lightness band, chroma, CVD and normal-vision
# separation), and the line style is a second cue besides the color.
FAMILY_COLORS = {"base": "#2a78d6", "Dist": "#1baf7a", "Implicit": "#4a3aa7", "DMDP": "#eb6834"}


def family(m):
    return "DMDP" if m.startswith("DMDP") else "Dist" if m.startswith("Dist-") else "Implicit" if m.startswith("Implicit-") else "base"


COLORS = {m: FAMILY_COLORS[family(m)] for m in METHODS}
LINESTYLE = {m: "--" if m.endswith("-Lag") else "-" for m in METHODS}
TEXTWIDTH = 6.5                                                        # inches, TMLR text width

# symbols used in the paper
S_D_RHOK = r"$\mathbb{D}(\rho_k\|\hat{\rho})$"
S_D_TARGET = r"$\mathbb{D}(\hat{\rho}\|\rho^{v}_\star)$"
S_RHOHAT = r"$\hat{\rho}$"
S_NU = r"noise level $\eta$"
S_K = r"step $k$"
S_V = r"forward velocity $v$"


def label_hline(ax, y, text):
    # writes the name of a horizontal reference line at y just above it, at the left end of the panel
    trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.02, y, text, transform=trans, ha="left", va="bottom", fontsize=8)


def label_vline(ax, x, text, top=True):
    # writes the name of a vertical reference line at x next to it, at the top (or bottom) of the panel
    ax.annotate(text, (x, 1.0 if top else 0.0), xycoords=("data", "axes fraction"), xytext=(2, -2 if top else 2),
                textcoords="offset points", ha="left", va="top" if top else "bottom", fontsize=8)


def save_figure(fig, name):
    # Saves the figure as <FIG_DIR>/<name>.pdf (for the paper) and <name>.png (200 dpi, for the README).
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=200)


def save_table(name, tex):
    with open(os.path.join(FIG_DIR, name), "w") as f:
        f.write(tex)
""")

md(r"""
## 2 Notation and configuration

| Paper | Code | Meaning |
|---|---|---|
| $\boldsymbol{\mathrm{s}}_t$, $\boldsymbol{\mathrm{o}}_t=\boldsymbol{\mathrm{s}}_t+\eta\boldsymbol{\xi}_t$ | `envs.SensorNoise` | observation vector and its noisy reading, noise level $\eta$; the entries of $\boldsymbol{\xi}_t$ are independent draws from $\tfrac12\mathcal{N}(-m,s^2)+\tfrac12\mathcal{N}(m,s^2)$, $m=0.8$ (`NOISE_OFFSET`), $s=0.6$ (`NOISE_SD`) (Appendix E) |
| $v_k$, $\theta_k$ | `info["x_velocity"]`, `info["pitch"]` | regulated features: forward velocity averaged over step $k$ (torso displacement divided by the step duration) and pitch angle of the torso (Appendix E) |
| $\rho_k$ | `velocity[:, k]` across episodes | distribution of $v_k$ across evaluation episodes at step $k$ |
| $\rho_\star$ | `targets.Target`, `mu_star`, `sd_star`, `config.band_target`, `pitch_mu_star`, `pitch_sd_star` | target: $U[0.70\,v_{\mathsf{lim}},0.95\,v_{\mathsf{lim}}]$ for the velocity (`TARGET_BAND`) and $U[-0.2,0.2]$ rad for the pitch (`PITCH_HALF_WIDTH`) (Appendix E) |
| $o^v_t$, $o^\theta_t$ | `obs[vel_index]`, `obs[pitch_index]` | noisy readings of the instantaneous forward velocity and of the pitch of the torso |
| $\hat{\omega}_t$ | window mean and std of the last $n_{\mathsf{w}}$ readings $o^v_t$ and $o^\theta_t$ | sliding-window estimate and policy input (Section 4.5) |
| $\bar{\rho}_t$ | empirical distributions of the true $v_{t-n_{\mathsf{w}}+1},\ldots,v_t$ and $\theta_{t-n_{\mathsf{w}}+1},\ldots,\theta_t$ | used only in the training dissimilarity (Appendix E) |
| $\phi(\hat{\omega}_t,\boldsymbol{\mathrm{o}}_t;\theta)$, $\pi(\boldsymbol{\mathrm{o}}_t)$ | `policy_input="hybrid"`, `policy_input="state"` | policy of the DMDP methods and of Implicit DMDP and Implicit DMDP-Lag (Section 4.5); policy of the other baselines |
| quantiles of the return | `n_quantiles` (`N_QUANTILES`), `agent.quantile_huber_loss`, `train.compute_quantile_returns` | distributional critic of the Distributional and Implicit DMDP methods: quantiles at the levels $(2i-1)/(2N)$, learned by quantile regression; their mean is the value |
| $\mathbb{D}_t=\mathbb{D}(\bar{\rho}_t\|\rho_\star)$ | `targets.empirical_w2_squared`, `info["dissimilarity"]`, `info["dissimilarity_velocity"]`, `info["dissimilarity_pitch"]` | training dissimilarity: sum of the exact squared 2-Wasserstein distances of the velocity window and of the pitch window to their targets (Appendix E) |
| $\mathbb{D}_{\mathsf{thr}}$ | `info["d_threshold"]`, `D_THRESHOLD`, `d_threshold` | threshold of the window dissimilarity, the same for every window length (Appendix E) |
| $r_{t\mathbb{D}}$, $d$ | `envs.ThresholdReward`, `info["r_tD"]`, `D_BONUS`, `d_bonus` | threshold reward of DMDP and DMDP-Lag: $+d$ if $\mathbb{D}_t\le\mathbb{D}_{\mathsf{thr}}$, $0$ otherwise (Appendix E) |
| $\beta$ | `beta`, `BETA` | weight of $\mathbb{D}_t$ in the rewards of DMDP and DMDP-Lag |
| $c_t=\mathbb{1}[v_t>v_{\mathsf{lim}}]$, $\bar c$ | `info["cost"]`, `COST_LIMIT` | safety cost and cost limit of the constrained methods (PPO-Lag, SAC-Lag, the distributional and implicit -Lag baselines, DMDP-Lag) |
| $r^{\mathsf{b}}_t=v_t-0.1\|\boldsymbol{\mathrm{a}}_t\|^2$ | `info["benchmark_reward"]`, `cost_penalty` $=0$ | benchmark reward, the training reward of PPO, SAC, Distributional PPO and SAC, and Implicit DMDP (and, with $-\lambda c_t$, of their -Lag variants), and the base of the rewards of the DMDP methods (Appendix E); only the design pilot (section 6.1) subtracted the cost penalty $(2c_t)^2$ (`envs.CostPenalty`, `COST_PENALTY`) |
| $r^{\mathsf{eva}}_t$, $r_{\mathsf{pen}}$ | `analysis.evaluation_rewards`, `analysis.evaluation_penalty`, `ret_eva` | evaluation reward, the same for every method: $r^{\mathsf{eva}}_t=r^{\mathsf{b}}_t+r_{t\mathbb{D}}$ if $v_t\leq v_{\mathsf{lim}}$ ($c_t=0$) and $r^{\mathsf{eva}}_t=-r_{\mathsf{pen}}$ with $r_{\mathsf{pen}}=v_{\mathsf{lim}}$ if $v_t>v_{\mathsf{lim}}$ ($c_t=1$), where $r_{t\mathbb{D}}$ is computed from the recorded window dissimilarity for every method (`analysis.threshold_reward`, $d=10$), also for the baselines, which are trained without it. It contains neither $-\beta\,\mathbb{D}_t$ nor $-\lambda c_t$. `ret_eva` is its episodic return, a column of the evaluation tables; `ret_benchmark` is the episodic return of $r^{\mathsf{b}}_t$ (`analysis.benchmark_returns`) |
| $R$, $R_{\mathsf{Thr}}$ | `episode_returns`, `ret` (`episodes.csv`), `ep_return` (`training.csv`), `R_THR` | episodic return of the reward each method is trained on, without $-\beta\,\mathbb{D}_t$: $\sum_t r^{\mathsf{b}}_t$ for the baselines, $\sum_t(r^{\mathsf{b}}_t+r_{t\mathbb{D}})$ for the DMDP methods; used by the training curves and the reliability analysis (section 5.5) with the return threshold |
| flipped | `flipped`, `ep_flipped` | fraction of steps with $|\theta_t|>\pi/2$ (the torso is upside down) |
| $\hat{\rho}$ | `metrics.estimate_rho_hat` | stable distribution from the steps $k\geq k_0$ of held-out episodes (Section 5) |
| $\mathbb{D}(\rho_k\|\hat{\rho})$ | `metrics.stability_curve` | evaluation dissimilarity, 1-Wasserstein (Section 5) |
| stationary reference value | `metrics.sampling_floor` | value of $\mathbb{D}(\rho_k\|\hat{\rho})$ for a stationary surrogate (Appendix F.2) |
""")
code(r"""
spec = ENVIRONMENTS[ENV]
H = EPISODE_LENGTH
target = Target(spec["target_family"], spec["mu_star"], spec["sd_star"])
print(f"environment          : {spec['env_id']}")
print(f"velocity threshold   : v_lim = {spec['v_limit']:.4f}")
cfg0 = make_config(ENV, "DMDP", 0)
pitch_target = Target("uniform", cfg0.pitch_mu_star, cfg0.pitch_sd_star)
print(f"cost limit           : c_bar = {COST_LIMIT:g} (every -Lag method) per episode of H = {H} steps")
print(f"velocity target      : {target.label()} = U[{TARGET_BAND[0]:g} v_lim, {TARGET_BAND[1]:g} v_lim];  mean {spec['mu_star']:.4f}, std {spec['sd_star']:.4f}; "
      f"expected cost under the target: {H * target.tail(spec['v_limit']):.1f}")
print(f"pitch target         : {pitch_target.label()} rad (PITCH_HALF_WIDTH = {PITCH_HALF_WIDTH:g})")
print(f"rewards              : every method is trained on the benchmark reward r^b_t = v_t - 0.1 ||a_t||^2 (cost_penalty = "
      f"{cfg0.cost_penalty:g}); the DMDP methods add r_tD - beta D; the reported return R is that of the reward each method is "
      f"trained on, without -beta D (design pilot only: cost penalty ({COST_PENALTY:g} c_t)^2)")
assert all(make_config(ENV, m, 0).cost_penalty == 0.0 for m in TASK + SAC_TASK)
D_THR = cfg0.d_threshold
N_W = cfg0.window
print(f"dissimilarity        : D_t = W_2^2(velocity window, velocity target) + W_2^2(pitch window, pitch target), window of {N_W} steps")
print(f"threshold D_thr      : {D_THR:g} for every window length (D_THRESHOLD = {D_THRESHOLD:g}); a window that is constant at the target means has "
      f"D = {spec['sd_star'] ** 2 + cfg0.pitch_sd_star ** 2:.3f}")
print(f"threshold reward     : r_tD = +{cfg0.d_bonus:g} if D <= D_thr, else 0 (DMDP and DMDP-Lag; D_BONUS = {D_BONUS:g})")
print(f"dissimilarity weight : beta = {BETA:g} (the same methods; chosen in the design pilot, Section 6.1 of this notebook)")
print(f"learner              : reward normalization = {cfg0.reward_normalization}; advantage (A_r - lambda A_c) / (1 + lambda) with A_r standardized")
print(f"initial action std   : exp({cfg0.init_log_std:g}) = {np.exp(cfg0.init_log_std):.2f} for every method (the PPO default is 1)")
print("\ntraining configuration (shared by all methods):")
method_specific = ("env", "method", "seed", "policy_input", "lagrangian", "beta", "d_bonus", "cost_penalty", "learner", "n_quantiles",
                   "train_noise")
print({k: v for k, v in cfg0.to_dict().items() if k not in method_specific})
pd.DataFrame(METHODS).T.loc[TASK]
""")

md(r"""
## 3 Training (optional)

Set `RUN_TRAINING = True` to train all runs of the main study from scratch; the command for every results folder is given in the README.
Each run trains one policy and evaluates it at every noise level in `EVAL_NOISES`. Finished runs are skipped, and only the
noise levels missing from `evaluation.npz` are evaluated (delete `evaluation.npz` of a run to re-evaluate its saved policy without retraining).
""")
code(r"""
RUN_TRAINING = False
if RUN_TRAINING:
    subprocess.run([sys.executable, "run_all.py", "--envs", ENV, "--methods", *TASK, "--seeds", *map(str, SEEDS),
                    "--tag", TAG, "--workers", "18"], check=True)
missing = [(m, s) for m in TASK for s in SEEDS if load_evaluation(ROOT, ENV, m, s) is None]
print("missing runs:", missing if missing else "none")
""")

md(r"""
## 4 Evaluation summary

For each policy and noise level, $\rho_k$ is estimated at every step $k$ from one half of the evaluation episodes
and $\hat{\rho}$ from the steps $k\geq 500$ of the other half; the two assignments are averaged (cross-fitting, Appendix F.2).
The distances $\mathbb{D}(\rho_k\|\hat{\rho})$, $\mathbb{D}(\hat{\rho}\|\rho^{v}_\star)$, and $\mathbb{D}(\hat{\rho}_\eta\|\hat{\rho}_0)$ are 1-Wasserstein distances.

* `D_late` is the mean of $\mathbb{D}(\rho_k\|\hat{\rho})$ over $k\geq 500$ (residual non-stationarity) and `D_early` the mean over $k<200$ (transient).
* `D_target` $=\mathbb{D}(\hat{\rho}\|\rho^{v}_\star)$, `D_target_pitch` is the same distance for the pitch angle, and `D_shift` $=\mathbb{D}(\hat{\rho}_\eta\|\hat{\rho}_0)$ is the shift of the stable distribution caused by sensor noise.
* `within` $=\mathbb{P}[\mathbb{D}(\bar{\rho}_t\|\rho_\star)\le\mathbb{D}_{\mathsf{thr}}]$ is the fraction of the evaluation steps $k\geq500$ whose window dissimilarity (Appendix E) is at most the threshold of the threshold reward.
* `flipped` is the fraction of the steps $k\geq500$ with $|\theta_k|>\pi/2$.

The data of a study (per-seed table `df`, stability `curves`, `mean` and `sem` over the seeds, and from section 5.5 on `returns` and `rel`) are held by a `Study` object:
`main` for the main study here, `sac` for the SAC study in section 8. The functions below take the study as their first argument.
""")
code(r"""
class Study:
    # evaluation summary of the runs in one results folder, trained with one learner ("ppo" or "sac")
    def __init__(self, root, methods, learner):
        self.root, self.methods, self.learner = root, methods, learner
        self.label = {m: method_label(m, learner) for m in methods}          # names in the paper, e.g. "DMDP (PPO)"
        self.df, self.curves = summarize(root, ENV, methods, SEEDS, k0=K0, block=BLOCK, k_early=K_EARLY)
        self.mean, self.sem = aggregate(self.df)


def mean_se(study, cols, digits=3):
    # "mean ± standard error over the seeds" of the columns cols, for every method and noise level
    return study.mean[cols].round(digits).astype(str) + " ± " + study.sem[cols].round(digits).astype(str)


main = Study(ROOT, TASK, "ppo")
cols = ["ret_eva", "ret_benchmark", "ret", "cost", "D_late", "D_early", "D_target", "within", "flipped", "D_target_pitch", "D_shift", "mu_hat", "sd_hat"]
mean_se(main, cols).loc[(TASK, [0.0, 0.2]), :]
""")
md(r"""
**Returns.** The evaluation tables report `ret_eva`, the episodic return of the evaluation reward $r^{\mathsf{eva}}_t$.
It is computed from the recorded rewards and window dissimilarities, the same way for every method: the evaluation reward is $r^{\mathsf{b}}_t+r_{t\mathbb{D}}$ at a safe step ($c_t=0$), with the benchmark reward $r^{\mathsf{b}}_t=v_t-0.1\|\boldsymbol{\mathrm{a}}_t\|^2$ and the threshold reward $r_{t\mathbb{D}}$, and $-r_{\mathsf{pen}}=-v_{\mathsf{lim}}$ at an unsafe step ($c_t=1$).
`ret_benchmark` is the episodic return of $r^{\mathsf{b}}_t$.
`ret` is the return $R$ of the reward each method is trained on, without the dissimilarity term $-\beta\,\mathbb{D}_t$,
i.e. of $r^{\mathsf{b}}_t$ for the six baselines of each learner and of $r^{\mathsf{b}}_t+r_{t\mathbb{D}}$ for the two DMDP methods; it is used by the training curves and the reliability analysis.
The decomposition below gives, for every method, the evaluation return (`eva`), the benchmark return (`benchmark`), the episodic threshold reward computed from the recorded window dissimilarity (`rtD`, between $0$ and $dH$;
the baselines are not trained on it), and `ret` (mean ± standard error over the seeds).
""")
code(r"""
rows = []
for m in TASK:
    for s in SEEDS:
        ev, cfg_run = load_evaluation(ROOT, ENV, m, s), load_config(ROOT, ENV, m, s)
        for nu in (0.0, 0.2):
            B = np.nansum(threshold_reward(ev[nu], cfg_run), axis=1)
            rows.append(dict(method=m, seed=s, nu=nu, eva=evaluation_returns(ev, nu, cfg_run).mean(), benchmark=benchmark_returns(ev, nu, cfg_run).mean(),
                             rtD=B.mean(), ret=episode_returns(ev, nu).mean()))
decomposition = pd.DataFrame(rows).groupby(["method", "nu"], sort=False)
(decomposition.mean().drop(columns="seed").round(1).astype(str) + " ± " + decomposition.sem().drop(columns="seed").round(1).astype(str))
""")

md("## 5 Figures")
md(r"""
### 5.1 Convergence of $\rho_k$ to the stable distribution $\hat{\rho}$ (Appendix F.2)

Mean over seeds (line) and standard error (band) of $\mathbb{D}(\rho_k\|\hat{\rho})$ against the step $k$ for $\eta\in\{0,0.1,0.3,0.5\}$ (`CURVE_NUS`).
$\mathbb{D}(\rho_k\|\hat{\rho})$ is computed at every single step; for display only, each seed's curve is averaged over 10 consecutive steps.
Averaging the dissimilarities (rather than pooling the samples of 10 steps into one distribution) keeps an oscillation of $\rho_k$ visible as a higher level.
The lower limit of the logarithmic axis is set from the data.
""")
code(r"""
DISPLAY_AVERAGE = 10                                                  # steps; display only, the tables use single steps


def curve_stats(study, method, nu):
    Ds = [study.curves[(method, s, nu)][1] for s in SEEDS if (method, s, nu) in study.curves]
    c = study.curves[(method, SEEDS[0], nu)][0]
    Ds = pd.DataFrame(np.array(Ds).T).rolling(DISPLAY_AVERAGE, center=True, min_periods=1).mean().values.T
    return c, np.nanmean(Ds, 0), np.nanstd(Ds, 0, ddof=1) / np.sqrt(len(Ds))


def log_axis_floor(x):
    # the largest of 1, 2, 5 times a power of ten that does not exceed x
    e = 10.0 ** np.floor(np.log10(x))
    return max(m * e for m in (1, 2, 5) if m * e <= x)


def legend_columns(methods):
    # one row of legend entries for up to five methods, otherwise rows of four
    return len(methods) if len(methods) <= 5 else 4


def row_major(handles, labels, ncol):
    # matplotlib fills a legend column by column; this order makes its rows read in the order of the methods
    nrows = -(-len(handles) // ncol)
    order = [r * ncol + c for c in range(ncol) for r in range(nrows) if r * ncol + c < len(handles)]
    return [handles[i] for i in order], [labels[i] for i in order]


def plot_curves(study, nus, name):
    fig, axes = plt.subplots(1, len(nus), figsize=(TEXTWIDTH, 1.9), sharey=True)
    lowest = np.inf
    for ax, nu in zip(np.atleast_1d(axes), nus):
        for m in study.methods:
            c, mu, se = curve_stats(study, m, nu)
            lowest = min(lowest, np.nanmin(np.where(mu - se > 0, mu - se, mu)))     # also the lower edge of the band
            ax.plot(c, mu, color=COLORS[m], ls=LINESTYLE[m], lw=1.2, label=study.label[m])
            ax.fill_between(c, mu - se, mu + se, color=COLORS[m], alpha=0.2, lw=0)
        ax.set_xlabel(S_K)
        ax.set_title(rf"$\eta={nu:g}$", fontsize=9)
        ax.set_yscale("log")
        ax.axvline(K0, color="k", lw=0.5, ls=":")
        label_vline(ax, K0, r"$k_0$")
    np.atleast_1d(axes)[0].set_ylim(bottom=log_axis_floor(0.9 * lowest))     # below the lowest curve and band
    np.atleast_1d(axes)[0].set_ylabel(S_D_RHOK)
    ncol = legend_columns(study.methods)
    h, l = row_major(*np.atleast_1d(axes)[0].get_legend_handles_labels(), ncol)
    fig.legend(h, l, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol)
    fig.tight_layout(w_pad=0.8)
    save_figure(fig, name)
    return fig


plot_curves(main, CURVE_NUS, "stability_curves");


def plateau_step(study, method, nu, tol=0.1):
    # first step at which the displayed curve is within 10% of its mean for k >= K0
    c, mu, _ = curve_stats(study, method, nu)
    return int(c[np.argmax(mu <= (1 + tol) * np.nanmean(mu[c >= K0]))])


pd.DataFrame({nu: {main.label[m]: plateau_step(main, m, nu) for m in TASK} for nu in CURVE_NUS})
""")

md(r"""
### 5.2 Stable distributions $\hat{\rho}$ and the target $\rho_\star$ (Appendix F.2)

Histogram of $v_k$ for $k\geq 500$, pooled over all evaluation episodes and seeds ($\eta=0$), with the velocity target (the same for all methods) and the threshold $v_{\mathsf{lim}}$,
for all eight methods (`STABLE`; with SAC, `SAC_STABLE`), four per row.
""")
code(r"""
def hist_bins(v, lo=-1.5, hi=6.0, width=0.125):
    # bins of fixed width that cover every sample, so that no sample is dropped from the density
    lo = min(lo, np.floor(np.min(v) / width) * width)
    hi = max(hi, np.ceil(np.max(v) / width) * width)
    return np.arange(lo, hi + width / 2, width)


def pooled_velocities(study, method, nu):
    # v_k for k >= K0 of all evaluation episodes and seeds
    out = []
    for s in SEEDS:
        ev = load_evaluation(study.root, ENV, method, s)
        if ev is not None:
            out.append(estimate_rho_hat(ev[nu]["velocity"], K0))
    return np.concatenate(out)


def plot_stable_distributions(study, methods, name, nu=0.0, ncols=4):
    nrows = -(-len(methods) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(TEXTWIDTH, 1.75 * nrows), sharey=True, sharex=True, squeeze=False)
    for ax in axes.flat[len(methods):]:
        ax.set_visible(False)
    axes = axes.flat
    pooled = {m: pooled_velocities(study, m, nu) for m in methods}
    lo = min(-1.5, np.floor(2 * min(v.min() for v in pooled.values())) / 2)  # show the whole distribution, also of failed runs
    grid = np.linspace(lo, 6, 400)
    for i, (ax, m) in enumerate(zip(axes, methods)):
        v = pooled[m]
        ax.hist(v, bins=hist_bins(v), density=True, color=COLORS[m], alpha=0.55, lw=0, label=S_RHOHAT)
        cfg = load_config(study.root, ENV, m, SEEDS[0])
        ax.plot(grid, Target(cfg["target_family"], cfg["mu_star"], cfg["sd_star"]).pdf(grid), "k--", lw=1, label=r"$\rho^{v}_\star$")
        ax.axvline(spec["v_limit"], color="k", lw=0.8, ls=":")
        label_vline(ax, spec["v_limit"], r"$v_{\mathsf{lim}}$")
        ax.set_title(study.label[m], fontsize=8.5)
        if i >= len(methods) - ncols:                                      # x labels on the bottom row only
            ax.set_xlabel(S_V)
        ax.set_xlim(lo, 6)
    for i in range(0, len(methods), ncols):
        axes[i].set_ylabel("density")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=0.6, h_pad=0.4)
    save_figure(fig, name)
    return fig


plot_stable_distributions(main, STABLE, "stable_distributions");
""")

md(r"""
### 5.3 Cost axis

The effect of the sensor noise is reported by the shift of the stable distribution (section 6, `table_shift.tex`) and by the noise sweep in section 7.
The helper below draws the cost axis of the training curves: linear up to 10 and logarithmic above, with the cost limit $\bar c$ as a dashed line named on the axis.
""")
code(r"""
def symlog_cost_axis(ax, limit):
    # cost axis: linear up to 10, logarithmic above, with plain tick labels and room for costs of 100 and more;
    # the cost limit is the dashed line, named by its tick label
    ax.axhline(limit, color="k", lw=0.6, ls="--")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(0, max(100, ax.get_ylim()[1]))
    ticks = sorted({0.0, 10.0, 100.0, limit})
    ax.set_yticks(ticks)
    ax.set_yticklabels([rf"$\bar{{c}}={t:g}$" if t == limit else f"{t:g}" for t in ticks])

""")

md(r"""
### 5.4 Training curves (Appendix F.2)

Episodic return, episodic cost, and window dissimilarity during training. Each logged value is the mean over the most recently completed training episodes;
the curves are smoothed further over ten logged values. The return is that of the reward each method is trained on, without $-\beta\,\mathbb{D}_t$ (`ep_return`): the benchmark reward for the baselines and $r^{\mathsf{b}}_t+r_{t\mathbb{D}}$ for the DMDP methods.
The dotted line in the return panel marks the return threshold $R_{\mathsf{Thr}}$ (section 5.5), and the dashed line in the cost panel the cost limit $\bar c$.
The third panel shows the window dissimilarity $\mathbb{D}(\bar\rho_t\|\rho_\star)$ of the training episodes (logarithmic axis) with the threshold $\mathbb{D}_{\mathsf{thr}}$ (dashed).
Each episode is summarized by the median of $\mathbb{D}_t$ over its steps, which lies below $\mathbb{D}_{\mathsf{thr}}$ if more than half of its steps receive $r_{t\mathbb{D}}=+d$.
At each logged step, the line is the median and the band the interquartile range of these episode medians over the most recently completed training episodes (`EPISODES_PER_ENV` per environment,
the episodes of a logged value) of all seeds pooled; both are smoothed by a running median over ten logged values. Unlike the other panels, the band shows the variation between episodes, not a standard error.
The line therefore lies below $\mathbb{D}_{\mathsf{thr}}$ when most of the recent episodes receive $r_{t\mathbb{D}}$ on most of their steps.
`training.csv` stores the *mean* over the recent episodes of a seed (`ep_dissimilarity_median`). The episode medians are heavy-tailed: a minority of the episodes, mostly with a tilted or flipped torso,
have medians far above $\mathbb{D}_{\mathsf{thr}}$ (up to about $16$). They raise this mean, and a mean over the seeds, far above the typical episode, so that it lies above $\mathbb{D}_{\mathsf{thr}}$
even when most episodes receive $r_{t\mathbb{D}}$ on most of their steps; it is therefore not plotted (the cell checks that the episodes of `episodes.csv` reproduce it).
The table after the figure gives, at the end of training (last 10% of the steps), the fraction of steps with $\mathbb{D}_t\le\mathbb{D}_{\mathsf{thr}}$
(the steps that receive $r_{t\mathbb{D}}$ in the methods trained on it) and the fraction of episodes whose median lies below $\mathbb{D}_{\mathsf{thr}}$ (mean ± standard error over the seeds).
""")
code(r"""
def recent_episode_medians(root, method, seed, training):
    # per-episode median of D of the most recently completed training episodes (the episodes of a logged value) at every
    # logged step of training.csv: shape (logged steps, EPISODES_PER_ENV * num_envs), NaN where fewer episodes were completed
    ep = load_episodes(root, ENV, method, seed)
    n_recent = EPISODES_PER_ENV * load_config(root, ENV, method, seed)["num_envs"]
    steps = ep["step"].values
    assert np.all(np.diff(steps) >= 0), "episodes.csv must be in the order of completion"
    count = np.searchsorted(steps, training["step"].values, side="right")      # episodes completed up to each logged step
    idx = count[:, None] - n_recent + np.arange(n_recent)[None, :]
    values = np.where(idx >= 0, ep["dis_median"].values[np.clip(idx, 0, None)], np.nan)
    assert count.min() >= 1 and np.allclose(np.nanmean(values, axis=1), training["ep_dissimilarity_median"].values), \
        f"the episodes of {method}/seed{seed} do not reproduce the logged mean"
    return values


def median_episode_curve(study, method, smooth=10):
    # quartiles (25%, 50%, 75%) of the per-episode median of D over the recent training episodes of all seeds pooled, at every
    # logged step, smoothed by a running median over `smooth` logged values
    blocks, steps = [], []
    for s in SEEDS:
        tr = load_training(study.root, ENV, method, s)
        if tr is not None:
            blocks.append(recent_episode_medians(study.root, method, s, tr))
            steps.append(tr["step"].values)
    n = min(len(b) for b in blocks)
    assert all(np.array_equal(st[:n], steps[0][:n]) for st in steps), f"the seeds of {method} are logged at different steps"
    q = np.nanquantile(np.concatenate([b[:n] for b in blocks], axis=1), [0.25, 0.5, 0.75], axis=1)
    return steps[0][:n], pd.DataFrame(q.T).rolling(smooth, min_periods=1).median().values.T


def plot_training(study, name):
    keys = [("ep_return", "episodic return"), ("ep_cost", "episodic cost")]
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(TEXTWIDTH, 1.9))
    for m in study.methods:
        runs = [load_training(study.root, ENV, m, s) for s in SEEDS]       # ep_return: return of the training reward without -beta D
        runs = [r for r in runs if r is not None]
        n = min(len(r) for r in runs)
        x = runs[0]["step"].values[:n] / 1e6
        for ax, (key, lab) in zip(axes, keys):
            Y = np.array([r[key].values[:n] for r in runs])
            Y = pd.DataFrame(Y.T).rolling(10, min_periods=1).mean().values.T
            mu, se = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y))
            ax.plot(x, mu, color=COLORS[m], ls=LINESTYLE[m], lw=1.1, label=study.label[m])
            ax.fill_between(x, mu - se, mu + se, color=COLORS[m], alpha=0.2, lw=0)
            ax.set_title(lab, fontsize=8.5)
        steps, (q25, q50, q75) = median_episode_curve(study, m)          # median and interquartile range over episodes
        axes[2].plot(steps / 1e6, q50, color=COLORS[m], ls=LINESTYLE[m], lw=1.1)
        axes[2].fill_between(steps / 1e6, q25, q75, color=COLORS[m], alpha=0.1, lw=0)
    axes[2].set_title(r"$\mathbb{D}(\bar{\rho}_t\|\rho_\star)$ (median episode)", fontsize=8.5)
    for ax in axes:
        ax.set_xlabel(r"steps ($\times 10^6$)")
    axes[0].axhline(R_THR, color="k", lw=0.6, ls=":")
    label_hline(axes[0], R_THR, r"$R_{\mathsf{Thr}}$")
    axes[2].axhline(D_THR, color="k", lw=0.8, ls="--")
    axes[2].set_yscale("log")
    axes[2].set_ylim(bottom=min(0.1, axes[2].get_ylim()[0]))                # so that the tick 10^-1 is labelled
    label_hline(axes[2], D_THR, r"$\mathbb{D}_{\mathsf{thr}}$")
    symlog_cost_axis(axes[1], COST_LIMIT)
    ncol = legend_columns(study.methods)
    h, l = row_major(*axes[0].get_legend_handles_labels(), ncol)
    fig.tight_layout(w_pad=0.6)
    fig.legend(h, l, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol)
    save_figure(fig, name)
    return fig


plot_training(main, "training_curves");

def end_of_training_threshold(study, late_fraction=LATE_FRACTION):
    # end of training (last 10% of the steps): fraction of the steps with D <= D_thr (the steps that receive r_tD in the methods
    # trained on it) and fraction of the episodes whose median of D lies below D_thr; mean +- standard error over the seeds
    rows = []
    for m in study.methods:
        for s in SEEDS:
            ep = load_episodes(study.root, ENV, m, s)
            late = ep[ep.step > (1 - late_fraction) * load_config(study.root, ENV, m, s)["total_steps"]]
            rows.append(dict(method=study.label[m], seed=s, steps_within=late.within.mean(), episodes_median_below=(late.dis_median <= D_THR).mean()))
    g = pd.DataFrame(rows).groupby("method", sort=False)[["steps_within", "episodes_median_below"]]
    return g.mean().round(2).astype(str) + " ± " + g.sem().round(2).astype(str)


end_of_training_threshold(main)
""")

md(r"""
### 5.5 Reliability of the return (Section 5.1 and Appendix F.2)

The probability $\mathbb{P}[R\geq x]$ that the episodic return $R$ (the return of the reward each method is trained on, without $-\beta\,\mathbb{D}_t$) reaches a value $x$ is estimated as the fraction of episodes whose return is at least $x$,
for two sets of episodes:

* **end of training**: the training episodes completed during the last 10% of the training steps, with stochastic actions and a noise level $\eta\sim\mathcal{U}[0,0.3]$ drawn for each episode (`episodes.csv`, column `ret`);
* **evaluation**: the evaluation episodes of each seed, with deterministic actions, at $\eta=0$ and $\eta=0.2$ (`evaluation.npz`).

The probability is computed for each seed and averaged over the seeds (mean ± standard error; `P`); `P_pooled` is the fraction over all episodes pooled over the seeds.
As a function of $x$, $\mathbb{P}[R\geq x]$ is the complementary distribution function of the return: it falls gradually over a wide range of $x$ if the return varies widely between episodes or seeds,
and steeply if the return is reliable. The range over which it falls from $0.9$ to $0.1$ is the 10–90% range of $R$ (`range_10_90`).
The tables use $x=R_{\mathsf{Thr}}$ (`R_THR`), an average reward of 5 per step; the grid below shows how the comparison depends on this threshold.
""")
code(r"""
REL_NUS = [0.0, 0.2]


def late_training_returns(method, root):
    # {seed: returns of the training episodes completed during the last LATE_FRACTION of the training steps}
    out = {}
    for s in SEEDS:
        ep = load_episodes(root, ENV, method, s)
        if ep is None or ep.empty:
            continue
        total = load_config(root, ENV, method, s)["total_steps"]
        late = ep[ep.step > (1 - LATE_FRACTION) * total]
        out[s] = late["ret"].values                                      # return of the training reward without -beta D
    return out


def evaluation_returns(method, root, nu):
    # {seed: returns of the evaluation episodes at the noise level nu}
    out = {}
    for s in SEEDS:
        ev = load_evaluation(root, ENV, method, s)
        if ev is not None:
            out[s] = episode_returns(ev, nu)
    return out


RETURN_SETS = {"end of training": late_training_returns}
RETURN_SETS.update({f"evaluation, nu={nu:g}": (lambda m, root, nu=nu: evaluation_returns(m, root, nu)) for nu in REL_NUS})


def reliability(per_seed, thr=R_THR):
    p = np.array([exceedance_probability(r, thr)[0] for r in per_seed.values()])
    pooled = np.concatenate(list(per_seed.values()))
    q10, q50, q90 = np.percentile(pooled, [10, 50, 90])
    means = np.array([np.mean(r) for r in per_seed.values()])
    return dict(P=p.mean(), P_se=p.std(ddof=1) / np.sqrt(len(p)), P_pooled=exceedance_probability(pooled, thr)[0],
                mean=pooled.mean(), q10=q10, median=q50, q90=q90, range_10_90=q90 - q10, sd_seed_means=means.std(ddof=1),
                episodes=pooled.size, P_per_seed=np.round(p, 2).tolist())


def reliability_frames(study):
    # returns of every set of episodes and their reliability statistics for the runs of the study
    rets = {(name, m): fn(m, study.root) for name, fn in RETURN_SETS.items() for m in study.methods}
    frame = pd.DataFrame([dict(set=name, method=m, **reliability(rets[(name, m)])) for name in RETURN_SETS for m in study.methods])
    return rets, frame.set_index(["set", "method"])


main.returns, main.rel = reliability_frames(main)
print(f"R_Thr = {R_THR:g}")
main.rel.round(3)
""")
code(r"""
# P[R >= x] on a grid of thresholds (pooled over the seeds), to check whether a comparison depends on the threshold
def threshold_grid(study, grid=np.arange(1000, 12001, 1000)):
    return pd.concat({set_name: pd.DataFrame({study.label[m]: exceedance_probability(np.concatenate(list(study.returns[(set_name, m)].values())), grid)
                                              for m in study.methods}, index=grid).T for set_name in RETURN_SETS}, axis=1).round(2)


threshold_grid(main)
""")

md(r"""
## 6 Tables

Mean ± standard error over the seeds. The methods are listed in the order of `TASK` (each method followed by its -Lag variant), and bold marks
the best value among all methods. The LaTeX tables are written to the output folder (`figures/` by default) and are included in the paper with `\input`:
`table_main.tex`, `table_reliability.tex`, and `table_shift.tex` (Appendix F.2); `table_floor.tex` (both learners, written in section 8; Appendix F.2); `table_pilot.tex` (design pilot, not in the paper).
""")
code(r"""
# Bold marks the best mean in a column (for each noise level) among all methods; the methods are listed in the order of study.methods.
def stack(*lines, align="c", pos="b"):
    # a header or label cell with several lines
    return rf"\begin{{tabular}}[{pos}]{{@{{}}{align}@{{}}}}" + r" \\ ".join(lines) + r"\end{tabular}"


def label_rows(label):
    # a long method name is split over the two rows (nu = 0 and nu = 0.2) that the method occupies in a table
    for long, first in (("Distributional PPO", "Distributional"), ("Distributional SAC", "Distributional"),
                        ("Distributional PPO-Lag", "Distributional"), ("Distributional SAC-Lag", "Distributional"),
                        ("PPO-Lag with the pitch term of $\mathbb{D}$", "PPO-Lag with the")):
        if label == long:
            return [first, label[len(first) + 1:]]
    if label.startswith("Implicit DMDP"):                                 # e.g. "Implicit DMDP-Lag" / "(PPO)"
        return label.rsplit(" ", 1)
    return [label, ""]


H_RET_EVA = stack("return", r"$r^{\mathsf{eva}}_t$ $\uparrow$")
H_DK_EARLY = stack(r"$\mathbb{D}(\rho_k\|\hat{\rho})$", r"$k<200$ $\downarrow$")
H_DK_LATE = stack(r"$\mathbb{D}(\rho_k\|\hat{\rho})$", r"$k\geq500$ $\downarrow$")
H_TARGET = stack(r"$\mathbb{D}(\hat{\rho}\|\rho^{v}_\star)$", r"$\downarrow$")
H_WITHIN = stack(r"$\mathbb{P}[\mathbb{D}(\bar{\rho}_t\|\rho_\star)$", r"$\leq\mathbb{D}_{\mathsf{thr}}]$ $\uparrow$")


def fmt_pm(m, s, d=2, bold=False):
    body = f"{m:.{d}f}\\pm{s:.{d}f}"
    return f"$\\bm{{{body}}}$" if bold else f"${body}$"


def best_keys(values, digits, higher):
    # keys whose value, rounded as displayed, is the best (ties are all marked); none if there are no candidates
    shown = {k: round(float(v), digits) for k, v in values.items()}
    if not shown:
        return set()
    best = max(shown.values()) if higher else min(shown.values())
    return {k for k, v in shown.items() if v == best}


def best_overall(methods, value, digits, higher):
    # methods whose value is the best among all methods (see best_keys)
    return best_keys({m: value(m) for m in methods}, digits, higher)


# column, decimals, direction (True: higher is better, False: lower is better, None: not bolded)
MAIN_COLUMNS = [("ret_eva", 0, True), ("cost", 1, False), ("D_early", 2, False), ("D_late", 3, False), ("D_target", 2, False),
                ("within", 2, True)]   # the flipped fraction is not shown here (posture: Section 5.1 and Appendix F)


def latex_table(study, nus, name):
    methods = study.methods
    best = {(nu, col): best_overall(methods, lambda m: study.mean.loc[(m, nu), col], d, hi)
            for nu in nus for col, d, hi in MAIN_COLUMNS if hi is not None}
    rows = []
    for m in methods:
        for nu in nus:
            mu, se = study.mean.loc[(m, nu)], study.sem.loc[(m, nu)]
            cells = [fmt_pm(mu[col], se[col], d, bold=hi is not None and m in best[(nu, col)]) for col, d, hi in MAIN_COLUMNS]
            rows.append(" & ".join([label_rows(study.label[m])[nus.index(nu)] if nus.index(nu) < 2 else "", f"${nu:g}$"] + cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    head = (r"\begin{tabular}{llcccccc}" "\n" r"\toprule" "\n"
            r"Method & $\eta$ & " + H_RET_EVA + r" & cost $\downarrow$ & " + " & ".join([H_DK_EARLY, H_DK_LATE, H_TARGET, H_WITHIN])
            + r" \\" "\n" r"\midrule")
    tex = head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    save_table(name, tex)
    return tex


print(latex_table(main, [0.0, 0.2], "table_main.tex"))

# shift of the stable distribution caused by sensor noise, D(rho_hat(nu) || rho_hat(0))
SHIFT_NUS = [nu for nu in EVAL_NOISES if nu > 0.0]                   # every noise level nu > 0


def shift_table(study, name):
    methods = study.methods
    best_shift = {nu: best_overall(methods, lambda m: study.mean.loc[(m, nu), "D_shift"], 2, False) for nu in SHIFT_NUS}
    lines = [" & ".join([study.label[m]] + [fmt_pm(study.mean.loc[(m, nu), "D_shift"], study.sem.loc[(m, nu), "D_shift"], bold=m in best_shift[nu])
                                        for nu in SHIFT_NUS]) + r" \\" for m in methods]
    n = len(SHIFT_NUS)
    tex = (r"\begin{tabular}{l" + "c" * n + "}" "\n" r"\toprule" "\n"
           rf" & \multicolumn{{{n}}}{{c}}{{$\mathbb{{D}}(\hat{{\rho}}_\eta\|\hat{{\rho}}_0)$ $\downarrow$}} \\" "\n" rf"\cmidrule(lr){{2-{n + 1}}}" "\n"
           r"Method & " + " & ".join(rf"$\eta={nu:g}$" for nu in SHIFT_NUS) + r" \\" "\n" r"\midrule" "\n"
           + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    save_table(name, tex)
    return tex


print(shift_table(main, "table_shift.tex"))

# reliability of the return: P[R >= R_Thr] (mean +- standard error over seeds) and the 10%-90% range of R, in the three
# sets of episodes, in one full-width table with one-line headers
REL_SETS = [("end of training", "end of training"), ("evaluation, nu=0", r"evaluation, $\eta=0$"),
            ("evaluation, nu=0.2", r"evaluation, $\eta=0.2$")]


def reliability_table(study, name):
    methods = study.methods
    cols = [(key, stat) for key, _ in REL_SETS for stat in ("P", "range_10_90")]
    best_rel = {c: best_overall(methods, lambda m: study.rel.loc[(c[0], m), c[1]], 2 if c[1] == "P" else 0, c[1] == "P") for c in cols}

    def rel_cell(m, c):
        row = study.rel.loc[(c[0], m)]
        if c[1] == "P":
            return fmt_pm(row.P, row.P_se, 2, bold=m in best_rel[c])
        return f"$\\bm{{{row.range_10_90:.0f}}}$" if m in best_rel[c] else f"${row.range_10_90:.0f}$"

    lines = [" & ".join([study.label[m]] + [rel_cell(m, c) for c in cols]) + r" \\" for m in methods]
    head = " & ".join([""] + [rf"\multicolumn{{2}}{{c}}{{{title}}}" for _, title in REL_SETS]) + r" \\"
    sub = " & ".join(["Method"] + [r"$\mathbb{P}[R\geq R_{\mathsf{Thr}}]$ $\uparrow$", r"10--90\% range $\downarrow$"] * len(REL_SETS)) + r" \\"
    tex = (r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccccc@{}}" "\n" r"\toprule" "\n" + head + "\n"
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n" + sub + "\n" r"\midrule" "\n"
           + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular*}" "\n")
    save_table(name, tex)
    return tex


print(reliability_table(main, "table_reliability.tex"))
""")

md(r"""
### 6.1 Design pilot on separate seeds (Appendix E)

The weight $\beta$ and the cost limit $\bar c$ were chosen in a pilot of DMDP-Lag on the seeds `PILOT_SEEDS`, which are not used in the reported study
(`results/pilot/<setting>/`), with PPO and PPO-Lag ($\bar c=10$) on the same seeds as the reference (`results/pilot/baselines_c10/`).
In this pilot, DMDP-Lag, PPO, and PPO-Lag were trained with the cost penalty $(2c_t)^2$ subtracted from the reward; the reported methods were later all
changed to the benchmark reward (the DMDP methods to $r^{\mathsf{b}}_t+r_{t\mathbb{D}}-\beta\,\mathbb{D}_t$), and $\beta=0.5$ and $\bar c=10$ were kept. With pilot runs of PPO and PPO-Lag on the benchmark reward (`results/pilot/baselines_c10_benchmark_reward/`), the rule would give $\bar c=5$ (last cell).
The pilot used the initial standard deviation $1$ of the actions (the PPO default) and the sensor noise of the main study.
The table reports the mean over the pilot seeds, without noise and with $\eta=0.2$. The rules, fixed before the pilot was run:

* $\beta$: among the settings with $\bar c=10$, the $\beta$ with the highest mean return without noise;
* $\bar c$: the largest of $\{10, 5, 2.5\}$ at this $\beta$ for which the mean cost of DMDP-Lag is below those of PPO and PPO-Lag, both without noise and at $\eta=0.2$.

The settings of the main study are `BETA` and `COST_LIMIT` in `dmdp_sg/config.py`; the last cell checks that they follow from the rules.
""")
code(r"""
rows = []
for tag, method, beta, limit in PILOT_SETTINGS:
    d, _ = summarize(os.path.join(HERE, "results", "pilot", tag), ENV, [method], PILOT_SEEDS, k0=K0, block=BLOCK, k_early=K_EARLY)
    assert len(d) == len(PILOT_SEEDS) * len(EVAL_NOISES), f"pilot runs missing for {tag}"
    cfgs = [load_config(os.path.join(HERE, "results", "pilot", tag), ENV, method, s) for s in PILOT_SEEDS]
    assert all(c["beta"] == beta and c["cost_limit"] == limit for c in cfgs), tag
    d["setting"] = tag
    rows.append(d)
pilot = pd.concat(rows).groupby(["setting", "nu"], sort=False)[["ret", "cost", "D_target", "within", "flipped", "D_late"]].mean()
display(pilot.loc[(slice(None), [0.0, 0.2]), :].round(3))

PILOT_COLUMNS = [("ret", 0, True), ("cost", 1, False), ("D_target", 2, False), ("within", 2, True), ("flipped", 2, False)]
lines = []
for tag, method, beta, limit in PILOT_SETTINGS:
    for nu in [0.0, 0.2]:
        row = pilot.loc[(tag, nu)]
        best = {col: (pilot.xs(nu, level="nu")[col].max() if hi else pilot.xs(nu, level="nu")[col].min()) for col, _, hi in PILOT_COLUMNS}
        cells = []
        for col, d, hi in PILOT_COLUMNS:
            v = f"{row[col]:.{d}f}"
            cells.append(f"$\\bm{{{v}}}$" if round(row[col], d) == round(best[col], d) else f"${v}$")
        lines.append(" & ".join(([f"${beta:g}$", f"${limit:g}$"] if nu == 0.0 else ["", ""]) + [f"${nu:g}$"] + cells) + r" \\")
tex = (r"\begin{tabular}{lllccccc}" "\n" r"\toprule" "\n"
       r"$\beta$ & $\bar{c}$ & $\eta$ & return $\uparrow$ & cost $\downarrow$ & " + H_TARGET + " & " + H_WITHIN + r" & flipped $\downarrow$ \\" "\n"
       r"\midrule" "\n" + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
save_table("table_pilot.tex", tex)
print(tex)
main_setting = [t for t in PILOT_SETTINGS if t[2] == BETA and t[3] == COST_LIMIT]
assert len(main_setting) == 1, "the setting of the main study (BETA, COST_LIMIT) must be one of the pilot settings"

# PPO and PPO-Lag on the pilot seeds and the two selection rules
d, _ = summarize(os.path.join(HERE, "results", "pilot", PILOT_BASELINES[0]), ENV, PILOT_BASELINES[1], PILOT_SEEDS, k0=K0, block=BLOCK, k_early=K_EARLY)
assert len(d) == len(PILOT_BASELINES[1]) * len(PILOT_SEEDS) * len(EVAL_NOISES), "pilot runs of PPO and PPO-Lag missing"
pilot_baselines = d.groupby(["method", "nu"], sort=False)[["ret", "cost", "D_target", "within", "flipped", "D_late"]].mean()
display(pilot_baselines.loc[(slice(None), [0.0, 0.2]), :].round(3))
beta_rule = max((t for t in PILOT_SETTINGS if t[3] == 10.0), key=lambda t: pilot.loc[(t[0], 0.0), "ret"])[2]
below = lambda t: all(pilot.loc[(t[0], nu), "cost"] < pilot_baselines.loc[(m, nu), "cost"] for m in PILOT_BASELINES[1] for nu in (0.0, 0.2))
limits = [t[3] for t in PILOT_SETTINGS if t[2] == beta_rule and below(t)]
print(f"rule for beta : {beta_rule:g};  rule for c_bar : {max(limits) if limits else None}")
assert (BETA, COST_LIMIT) == (beta_rule, max(limits)), "BETA and COST_LIMIT must follow the selection rules of the pilot"

# the rule for c_bar with the pilot runs of PPO and PPO-Lag on the benchmark reward (not used: c_bar = 10 was kept)
d, _ = summarize(os.path.join(HERE, "results", "pilot", PILOT_BASELINES_BENCHMARK), ENV, PILOT_BASELINES[1], PILOT_SEEDS, k0=K0, block=BLOCK, k_early=K_EARLY)
pilot_benchmark = d.groupby(["method", "nu"], sort=False)[["ret", "cost", "D_target", "within", "flipped", "D_late"]].mean()
display(pilot_benchmark.loc[(slice(None), [0.0, 0.2]), :].round(3))
below_b = lambda t: all(pilot.loc[(t[0], nu), "cost"] < pilot_benchmark.loc[(m, nu), "cost"] for m in PILOT_BASELINES[1] for nu in (0.0, 0.2))
limits_b = [t[3] for t in PILOT_SETTINGS if t[2] == beta_rule and below_b(t)]
print(f"rule for c_bar with the benchmark-reward baselines: {max(limits_b) if limits_b else None} (kept: {COST_LIMIT:g})")
""")

md(r"""
## 7 Sanity checks

**Stationary reference value of the evaluation metric (Appendix F.2).** Even a perfectly stationary process has $\mathbb{D}(\rho_k\|\hat{\rho})>0$, because $\rho_k$ and $\hat{\rho}$ are estimated from finitely many episodes.
`metrics.sampling_floor` estimates the value for a stationary process by applying the same estimator to a surrogate that is stationary by construction:
the segment $k\geq500$ of every episode is shifted circularly in time by an independent random amount, which keeps $\hat{\rho}$, the differences between episodes, and the correlation between consecutive steps.
The value is averaged over several surrogates.
The tables report the ratio of $\mathbb{D}(\rho_k\|\hat{\rho})$ for $k\geq500$ to this stationary reference value. A ratio close to one is consistent with a stationary $\rho_k$ up to sampling error;
a ratio clearly above one means that $\rho_k$ still changes, for example because the episodes are synchronized.
""")
code(r"""
FLOOR_NUS = list(EVAL_NOISES)                                          # every noise level


def floor_table(study):
    rows = []
    for m in study.methods:
        for nu in FLOOR_NUS:
            # one generator per (seed, nu): the surrogates do not depend on the order of the runs or on the other cells
            floors = [sampling_floor(load_evaluation(study.root, ENV, m, s)[nu]["velocity"], K0, BLOCK,
                                     np.random.default_rng([s, round(10 * nu)])) for s in SEEDS]
            rows.append(dict(method=m, nu=nu, floor=np.mean(floors), D_late=study.mean.loc[(m, nu), "D_late"]))
    out = pd.DataFrame(rows).set_index(["method", "nu"])
    out["ratio"] = out["D_late"] / out["floor"]
    return out


def floor_table_both(studies, name):
    # one table for both learners: a block of rows per learner, labelled by the learner
    n = len(FLOOR_NUS)
    lines = []
    for learner, study, out in studies:
        lines.append(rf"\multicolumn{{{n + 1}}}{{@{{}}l}}{{\emph{{with {learner}}}}} \\")
        lines += [" & ".join([study.label[m]] + [f"${out.loc[(m, nu), 'ratio']:.2f}$" for nu in FLOOR_NUS]) + r" \\" for m in study.methods]
        lines.append(r"\midrule")                                       # between the two learners
    tex = (rf"\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}l*{{{n}}}{{c}}@{{}}}}" "\n" r"\toprule" "\n"
           r"Method & " + " & ".join(rf"$\eta={nu:g}$" for nu in FLOOR_NUS) + r" \\" "\n" r"\midrule" "\n"
           + "\n".join(lines[:-1]) + "\n" r"\bottomrule" "\n" r"\end{tabular*}" "\n")
    save_table(name, tex)
    return tex


floor = floor_table(main)          # written together with the SAC study in section 8 (table_floor.tex)
floor.round(3)
""")
md(r"""
**Stationarity test of each run (Appendix F.2).** For each trained policy and noise level, $\mathbb{D}(\rho_k\|\hat{\rho})$ for $k\geq500$ is compared with its values on `NULL_DRAWS` independently drawn
circular-shift surrogates. The one-sided $p$-value is $(1+n_{\geq})/(1+N)$, where $N$ is the number of surrogates (`NULL_DRAWS`) and $n_{\geq}$ counts those whose value is at least that of the run.
The test is approximate, because circular shifting keeps the distribution of a stationary process at every single step but not its joint distribution at the wrap-around point.
""")
code(r"""
NULL_DRAWS = 60


def stationarity_tests(study):
    rows = []
    for m in study.methods:
        for nu in FLOOR_NUS:
            for s in SEEDS:
                v = load_evaluation(study.root, ENV, m, s)[nu]["velocity"]
                c, D, _ = stability_curve(v, k0=K0, block=BLOCK)
                late = np.nanmean(D[c >= K0])
                f = sampling_floor(v, K0, BLOCK, np.random.default_rng([s, round(10 * nu)]))
                rng = np.random.default_rng([99, s, round(10 * nu)])
                null = np.array([sampling_floor(v, K0, BLOCK, rng, draws=1) for _ in range(NULL_DRAWS)])
                rows.append(dict(method=m, seed=s, nu=nu, ratio=late / f, p_value=(1 + np.sum(null >= late)) / (1 + NULL_DRAWS)))
    out = pd.DataFrame(rows).set_index(["method", "seed", "nu"])
    rejected = out[out.p_value < 0.05]
    print(f"stationarity rejected at the 5% level in {len(rejected)} of {len(out)} runs "
          f"(about {len(out) * 3 / (NULL_DRAWS + 1):.1f} expected by chance); "
          f"{(rejected.ratio < 1.3).sum()} of them with a ratio below 1.3")
    return out


tests = stationarity_tests(main)
tests[tests.p_value < 0.05].sort_values("ratio", ascending=False).round(3)
""")
md(r"""
**Synchronized episodes (Section 5.1 and Appendix F.2).** If the episodes run with independent phases, the mean velocity across the evaluation episodes fluctuates over $k\geq500$ only by its sampling error.
`amplitude_ratio` is the standard deviation over $k$ of this mean divided by its standard error; a run is called synchronized if the ratio exceeds `SYNC_THRESHOLD`.
It is computed without noise and at $\eta=0.1$ (`SYNC_NUS`); `period` is that of the largest peak of the spectrum of the mean velocity without noise. All runs are listed, sorted by `amplitude_ratio` without noise;
`ratio` and `p_value` are those of the previous test. The same table for the SAC study is `sac_phase` in section 8.
""")
code(r"""
SYNC_THRESHOLD = 2.0
SYNC_NUS = (0.0, 0.1)


def synchronization(study, tests_of_study):
    rows = []
    for m in study.methods:
        for s in SEEDS:
            ev = load_evaluation(study.root, ENV, m, s)
            row = dict(method=m, seed=s)
            for nu in SYNC_NUS:
                v = ev[nu]["velocity"][:, K0:]
                mean_k = v.mean(0)                                        # mean velocity across episodes at each step
                standard_error = np.mean(v.std(0, ddof=1)) / np.sqrt(v.shape[0])
                row[f"amplitude_ratio_nu{nu:g}"] = mean_k.std() / standard_error
                if nu == 0.0:
                    spectrum = np.abs(np.fft.rfft(mean_k - mean_k.mean()))
                    freq = np.fft.rfftfreq(mean_k.size)
                    row["period"] = 1 / freq[1 + np.argmax(spectrum[1:])]
            rows.append(row)
    out = pd.DataFrame(rows).set_index(["method", "seed"])
    for nu in SYNC_NUS:
        out[f"synchronized_nu{nu:g}"] = out[f"amplitude_ratio_nu{nu:g}"] > SYNC_THRESHOLD
    out = out.join(tests_of_study.xs(0.0, level="nu")[["ratio", "p_value"]])
    for nu in SYNC_NUS:
        print(f"synchronized runs at nu = {nu:g} (amplitude_ratio > {SYNC_THRESHOLD:g}): {int(out[f'synchronized_nu{nu:g}'].sum())} of {len(out)}")
    print(out.groupby("method", sort=False)[[f"synchronized_nu{nu:g}" for nu in SYNC_NUS]].sum().loc[study.methods].to_string())
    return out


phase = synchronization(main, tests)
phase.round(2).sort_values("amplitude_ratio_nu0", ascending=False)
""")
md(r"""
**Per-seed details behind Appendix F.2.** Noise-free return, cost, distance to $\rho_\star$, and stable distribution per seed, its shift at $\eta=0.3$ and $\eta=0.5$, and the decomposition
of the variation of $v_k$ ($k\geq500$, $\eta=0$) into the average within-episode standard deviation and the standard deviation of the episode means.
""")
code(r"""
rows = []
for m in TASK:
    for s in SEEDS:
        v = load_evaluation(ROOT, ENV, m, s)[0.0]["velocity"][:, K0:]
        r0 = main.df[(main.df.method == m) & (main.df.seed == s) & (main.df.nu == 0.0)].iloc[0]
        r3 = main.df[(main.df.method == m) & (main.df.seed == s) & (main.df.nu == 0.3)].iloc[0]
        r5 = main.df[(main.df.method == m) & (main.df.seed == s) & (main.df.nu == 0.5)].iloc[0]
        rows.append(dict(method=m, seed=s, ret=r0.ret, cost=r0.cost, D_target=r0.D_target, mu_hat=r0.mu_hat, sd_hat=r0.sd_hat,
                         shift_nu03=r3.D_shift, shift_nu05=r5.D_shift,
                         within_sd=np.mean(np.std(v, axis=1)), between_sd=np.std(np.mean(v, axis=1))))
per_seed = pd.DataFrame(rows).set_index(["method", "seed"]).round(2)
per_seed.loc[TASK]
""")

md(r"""
**Noise sweep (Appendix F.2).** Mean ± standard error over the seeds of the evaluation return, the benchmark return, the return of the training reward, the cost, $\mathbb{D}(\rho_k\|\hat{\rho})$ for $k\geq500$, the distance to the target, and of the mean and standard deviation of $\hat{\rho}$ (`mu_hat`, `sd_hat`), for every method and noise level.
""")
code(r"""
mean_se(main, ["ret_eva", "ret_benchmark", "ret", "cost", "D_late", "D_target", "mu_hat", "sd_hat"]).loc[TASK]
""")

md(r"""
**Body orientation (Appendices E and F.2).** HalfCheetah can also run on its back, which the benchmark does not penalize. For every trained policy and noise level,
the fraction of the evaluation steps $k\geq500$ with a flipped torso ($|\theta_k|>\pi/2$), from the recorded pitch angle; a run counts as flipped at a noise level if this fraction exceeds $0.5$.
The second table gives the mean fraction per method and noise level, and the third the mean absolute pitch per seed without noise.
""")
code(r"""
orientation = main.df.pivot_table(index=["method", "seed"], columns="nu", values="flipped").loc[TASK]
print("flipped runs (fraction of flipped steps > 0.5), by noise level:")
print((orientation > 0.5).groupby("method", sort=False).sum().loc[TASK].to_string())
display(orientation.groupby("method", sort=False).mean().loc[TASK].round(3))
main.df[main.df.nu == 0.0].pivot(index="method", columns="seed", values="pitch").loc[TASK].round(2)
""")

md(r"""
**Episodes with a negative benchmark return (Appendix F.2).** A running gait has a positive return of the benchmark reward $v_t-0.1\|\boldsymbol{\mathrm{a}}_t\|^2$, so a negative benchmark return (`analysis.benchmark_returns`, without $r_{t\mathbb{D}}$) means that the robot made almost no
forward progress in the episode. The table counts, for every noise level, the evaluation episodes with a negative return, the end-of-training
episodes (section 5.5) with a negative return, and the noise-free evaluation episodes in which the robot comes to rest: the policy holds a fixed posture,
so that both the absolute mean and the standard deviation of $v_k$ over $k\geq500$ are below $0.05$.
""")
code(r"""
def at_rest(v, k0=K0, tol=0.05):
    late = v[:, k0:]
    return int(np.sum((np.abs(np.nanmean(late, 1)) < tol) & (np.nanstd(late, 1) < tol)))


rows = []
for m in TASK:
    for s in SEEDS:
        ev, ep, cfg_run = load_evaluation(ROOT, ENV, m, s), load_episodes(ROOT, ENV, m, s), load_config(ROOT, ENV, m, s)
        late_ep = ep.loc[ep.step > (1 - LATE_FRACTION) * cfg_run["total_steps"]]
        late = late_ep["task_ret"] + cfg_run.get("cost_penalty", 0.0) ** 2 * late_ep["cost"]     # benchmark return
        rows.append(dict(method=m, seed=s, **{f"neg_nu{nu:g}": int((benchmark_returns(ev, nu, cfg_run) < 0).sum()) for nu in EVAL_NOISES},
                         at_rest_nu0=at_rest(ev[0.0]["velocity"]), end_of_training_negative=int((late < 0).sum()),
                         end_of_training_episodes=int(late.size)))
negative = pd.DataFrame(rows).set_index(["method", "seed"])
totals = negative.groupby("method", sort=False).sum().loc[TASK]
totals["neg_nu<=0.3"] = totals[[f"neg_nu{nu:g}" for nu in EVAL_NOISES if nu <= TRAIN_NOISE_MAX]].sum(axis=1)
print(totals.to_string())
# largest mean velocity (over the whole episode) of an evaluation episode with a negative return, for every noise level and without noise
for m in TASK:
    vbar = {nu: [] for nu in EVAL_NOISES}
    for s in SEEDS:
        ev, cfg_run = load_evaluation(ROOT, ENV, m, s), load_config(ROOT, ENV, m, s)
        for nu in EVAL_NOISES:
            neg = benchmark_returns(ev, nu, cfg_run) < 0
            vbar[nu] += list(np.nanmean(ev[nu]["velocity"][neg], axis=1))
    every = [x for v in vbar.values() for x in v]
    if every:
        print(f"{main.label[m]:28s} largest mean velocity of a negative-return episode: {max(every):.2f} (any nu), "
              f"{max(vbar[0.0]) if vbar[0.0] else float('nan'):.2f} (nu = 0)")
negative
""")

md(r"""
**Episode survival.** HalfCheetah has no termination, so all evaluation episodes last $H$ steps; the check below confirms this (minimum fraction of episodes still running at the last step).
""")
code(r"""
main.df.groupby("method", sort=False)["survival"].min().loc[TASK]
""")


md(r"""
## 8 Second learner: SAC (paper Section 5.1 and Appendix F.1)

The eight methods are trained a second time with soft actor-critic (SAC, `dmdp_sg/sac.py`) in place of PPO: SAC, SAC-Lag, Distributional SAC, Distributional SAC-Lag, Implicit DMDP (SAC), and Implicit DMDP-Lag (SAC) take the places of PPO, PPO-Lag, Distributional PPO, Distributional PPO-Lag, Implicit DMDP (PPO), and Implicit DMDP-Lag (PPO)
and, like them, are trained on the benchmark reward, and DMDP (SAC) and DMDP-Lag (SAC) are trained on the same rewards as DMDP (PPO) and DMDP-Lag (PPO) in the main study. The environment, the target, the rewards, $\beta$, $\bar c$,
$\mathbb{D}_{\mathsf{thr}}$, $d$, the training noise, the seeds, and the evaluation are those of the main study; only the learner and the number of training steps
differ. The SAC hyperparameters are common default values and were not tuned (Appendix E). The figures and tables below are produced by the
same functions as in sections 4–7, applied to the study object `sac`, and are written with the prefix `sac_`. The command that trains the runs is given in the README.
""")
code(r"""
missing = [(m, s) for m in SAC_TASK for s in SEEDS if load_evaluation(SAC_ROOT, ENV, m, s) is None]
print("missing SAC runs:", missing if missing else "none")
sac = Study(SAC_ROOT, SAC_TASK, "sac")
sac.returns, sac.rel = reliability_frames(sac)
cfg_sac = load_config(SAC_ROOT, ENV, "DMDP-Lag", SEEDS[0])
print({k: v for k, v in cfg_sac.items()
       if k.startswith("sac_") or k in ("learner", "total_steps", "num_envs", "beta", "cost_limit", "d_bonus", "d_threshold", "gamma")})
mean_se(sac, cols).loc[(SAC_TASK, [0.0, 0.2]), :]
""")
md(r"""
### 8.1 Figures with SAC

The same figures as in section 5. Each logged training value is the mean over the most recently completed training episodes of a SAC run
(the dissimilarity panel is computed as in section 5.4), and the training curves are smoothed over 10 logged values.
""")
code(r"""
plot_training(sac, "sac_training_curves");
end_of_training_threshold(sac)
""")
code(r"""
plot_curves(sac, CURVE_NUS, "sac_stability_curves");
pd.DataFrame({nu: {sac.label[m]: plateau_step(sac, m, nu) for m in SAC_TASK} for nu in CURVE_NUS})
""")
code(r"""
plot_stable_distributions(sac, SAC_STABLE, "sac_stable_distributions");
""")
code(r"""
display(sac.rel.round(3))
threshold_grid(sac)
""")
md(r"""
### 8.2 Tables with SAC

The evaluation results, the reliability of the return, the shift of the stable distribution under noise, the ratio to the stationary reference value,
and the stationarity test of each run, for the eight methods trained with SAC.
""")
code(r"""
print(latex_table(sac, [0.0, 0.2], "table_sac_main.tex"))
print(reliability_table(sac, "table_sac_reliability.tex"))
print(shift_table(sac, "table_sac_shift.tex"))
sac_floor = floor_table(sac)
display(sac_floor.round(3))
print(floor_table_both([("PPO", main, floor), ("SAC", sac, sac_floor)], "table_floor.tex"))
sac_tests = stationarity_tests(sac)
display(sac_tests[sac_tests.p_value < 0.05].sort_values("ratio", ascending=False).round(3))
sac_phase = synchronization(sac, sac_tests)                            # synchronized runs with SAC (section 7)
sac_phase.round(2).sort_values("amplitude_ratio_nu0", ascending=False)
""")
md(r"""
### 8.3 Details with SAC

Noise sweep, posture, and per-seed results of the eight methods trained with SAC, and the activity of the Lagrange multipliers of the constrained methods
with both learners (fraction of the logged training values with $\lambda>0$, and the final and largest values).
""")
code(r"""
display(mean_se(sac, ["ret_eva", "ret_benchmark", "ret", "cost", "D_late", "D_target", "mu_hat", "sd_hat"]).loc[SAC_TASK])
orientation_sac = sac.df.pivot_table(index=["method", "seed"], columns="nu", values="flipped").loc[SAC_TASK]
print("flipped runs (fraction of flipped steps > 0.5), by noise level:")
print((orientation_sac > 0.5).groupby("method", sort=False).sum().loc[SAC_TASK].to_string())
display(orientation_sac.groupby("method", sort=False).mean().loc[SAC_TASK].round(3))
display(sac.df[sac.df.nu == 0.0].pivot(index="method", columns="seed", values="pitch").loc[SAC_TASK].round(2))
sac.df[sac.df.nu == 0.0].set_index(["method", "seed"])[["ret", "cost", "D_target", "within", "flipped", "mu_hat", "sd_hat"]].loc[SAC_TASK].round(2)
""")
code(r"""
rows = []
for root, methods, learner in ((ROOT, [m for m in TASK if m.endswith("-Lag")], "PPO"), (SAC_ROOT, [m for m in SAC_TASK if m.endswith("-Lag")], "SAC")):
    for m in methods:
        for s in SEEDS:
            tr = load_training(root, ENV, m, s)
            rows.append(dict(learner=learner, method=method_label(m, learner.lower()), seed=s, active=(tr.lagrange_multiplier > 0).mean(),
                             final=tr.lagrange_multiplier.iloc[-1], largest=tr.lagrange_multiplier.max()))
multipliers = pd.DataFrame(rows).set_index(["learner", "method", "seed"])
display(multipliers.groupby(["learner", "method"], sort=False).agg(["min", "max"]).round(2))
multipliers.round(2)
""")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "torch_safety", "display_name": "torch_safety", "language": "python"}
    nb.cells = CELLS
    nbf.write(nb, NOTEBOOK)
    print(f"wrote {os.path.basename(NOTEBOOK)} with {len(CELLS)} cells")


if __name__ == "__main__":
    main()
