"""Builds DMDP_SafetyGymnasium.ipynb, the notebook that reproduces Section 7 and Appendix D of the paper.

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
    CELLS.append(nbf.v4.new_markdown_cell(source.strip("\n")))


def code(source: str) -> None:
    CELLS.append(nbf.v4.new_code_cell(source.strip("\n")))


md(r"""
# DMDP on Safety-Gymnasium

This notebook reproduces every number, figure, and table of Section 7 (Empirical Validation) and Appendix D (Experimental Details)
of *Dissimilarity-Based Markov Decision Processes for Stabilizing State Distributions*.
The practical implementation that is evaluated is described in Section 6 (Practical Implementation).

**What is evaluated.** The theory of Sections 3–5 concerns the convergence of the state distribution $\rho_k$ to a stable distribution.
The experiments measure this empirically:

* **Distributional stability** — the dissimilarity $\mathbb{D}(\rho_k\|\hat{\rho})$ between the distribution $\rho_k$ of the forward velocity at step $k$
  (estimated across evaluation episodes) and the stable distribution $\hat{\rho}$ estimated from the late steps of *other* evaluation episodes.
  Because the training objective has several components (task reward, safety cost, dissimilarity), $\hat{\rho}$ need not equal the target $\rho_\star$;
  $\mathbb{D}(\hat{\rho}\|\rho_\star)$ is therefore reported separately.
* **Reward and safety** — episodic return and cost. The target $\rho_\star$ is derived from the task so that it corresponds to fast (high-reward) running
  whose expected cost equals the cost limit (Section 6.1). A deliberately slow target shows that stability alone does not imply reward.
* **Sensor noise** — readings $\boldsymbol{\mathrm{o}}_t=\boldsymbol{\mathrm{s}}_t+\nu\boldsymbol{\xi}_t$ (Section 6.2); robustness is an additional property, not the motivation.

No Lyapunov certificate of Theorems 2–5 is computed for the practical implementation (Section 6).

**Outputs.** Figures (PDF and PNG) and LaTeX tables are written to `figures/`, or to the directory given by the environment variable `DMDP_FIG_DIR`.
Figure text is typeset with LaTeX (Latin Modern, as in the paper) when a `latex` executable is found, and with matplotlib's mathtext otherwise.

**Structure.** 1 Setup · 2 Notation and configuration · 3 Training (optional, about 1 h with 14 parallel workers) · 4 Evaluation summary ·
5 Figures · 6 Tables · 7 Sanity checks.
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
from scipy.stats import norm

HERE = os.path.abspath(".")
sys.path.insert(0, HERE)

from dmdp_sg.analysis import aggregate, load_config, load_evaluation, load_training, summarize
from dmdp_sg.config import COST_LIMIT, ENVIRONMENTS, EPISODE_LENGTH, METHOD_LABELS, METHODS, make_config
from dmdp_sg.metrics import estimate_rho_hat, sampling_floor

ENV = "HalfCheetah"
TAG = "main"                                   # results/<TAG>/<ENV>/<method>/seed<s>
ROOT = os.path.join(HERE, "results", TAG)
FIG_DIR = os.path.abspath(os.environ.get("DMDP_FIG_DIR", os.path.join(HERE, "figures")))
os.makedirs(FIG_DIR, exist_ok=True)

MAIN = ["PPO", "PPO-Lag", "Window-Lag", "DMDP"]                      # main comparison (Section 7.2)
ABLATION = ["DMDP", "DMDP-noD", "State-D"]                           # omega_hat_t input, D term, or both (Section 7.4)
TARGETS = ["DMDP", "DMDP-slow"]                                      # stability does not imply reward (Section 7.2)
ALL = list(dict.fromkeys(MAIN + ABLATION + TARGETS))
SEEDS = [0, 1, 2, 3, 4]
K0, BLOCK, K_EARLY = 500, 10, 200                                    # rho_hat from k >= K0; blocks of 10 steps
BETA_TAGS = {0.25: "beta025", 4.0: "beta4"}                          # beta = 1 is the main run

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

COLORS = {"PPO": "#7f7f7f", "PPO-Lag": "#1f77b4", "Window-Lag": "#17becf", "DMDP": "#d62728",
          "DMDP-noD": "#ff7f0e", "State-D": "#9467bd", "DMDP-slow": "#2ca02c"}
LABEL = dict(METHOD_LABELS)
TEXTWIDTH = 6.5                                                        # inches, TMLR text width

# symbols used in the paper
S_D_RHOK = r"$\mathbb{D}(\rho_k\|\hat{\rho})$"
S_D_TARGET = r"$\mathbb{D}(\hat{\rho}\|\rho_\star)$"
S_RHOSTAR = r"$\rho_\star$"
S_RHOHAT = r"$\hat{\rho}$"
S_NU = r"noise level $\nu$"
S_K = r"step $k$"
S_V = r"forward velocity $v$"


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
| $\boldsymbol{\mathrm{s}}_t$, $\boldsymbol{\mathrm{o}}_t=\boldsymbol{\mathrm{s}}_t+\nu\boldsymbol{\xi}_t$ | `envs.SensorNoise` | observation vector and its noisy reading, noise level $\nu$ (Section 6.2) |
| $v_k$ | `info["x_velocity"]` | forward velocity (the regulated feature, Section 6.1) |
| $\rho_k$ | `velocity[:, k]` across episodes | distribution of $v_k$ across evaluation episodes |
| $\rho_\omega=\mathcal{N}(\mu,\varsigma^2)$, $\omega=(\mu,\varsigma)$ | `envs.DistributionEstimator` | Gaussian family (Section 6.1) |
| $\rho_\star=\mathcal{N}(\mu_\star,\varsigma_\star^2)$ | `mu_star`, `sd_star`, `config.target_mean` | target distribution, $\mu_\star=v_{\mathsf{lim}}-z_{1-\bar c/H}\,\varsigma_\star$ (Section 6.1) |
| $o^v_t$ | `obs[vel_index]` | noisy reading of the forward velocity |
| $\hat{\omega}_t=(\hat\mu_t,\hat\varsigma_t)$ | window mean and std of the last $n_{\mathsf{w}}$ readings $o^v_t$ | sliding-window estimate and policy input (Section 6.2) |
| $\bar{\omega}_t$ | window mean and std of the true velocity | used only in the training dissimilarity (Section 6.3) |
| $\phi(\hat{\omega}_t,\boldsymbol{\mathrm{o}}_t;\theta)$ | `policy_input="hybrid"` | DMDP policy (Section 6.2) |
| $\mathbb{D}(\rho_{\bar{\omega}_t}\|\rho_\star)=(\bar\mu_t-\mu_\star)^2+(\bar\varsigma_t-\varsigma_\star)^2$ | `info["dissimilarity"]` | training dissimilarity, $W_2^2$ between Gaussians (Section 6.1) |
| $\beta$ | `beta` | weight of the dissimilarity in the shaped reward $\tilde r_t=r_t-\beta\,\mathbb{D}$ (Section 6.3) |
| $c_t=\mathbb{1}[v_t>v_{\mathsf{lim}}]$, $\bar c$ | `info["cost"]`, `COST_LIMIT` | safety cost and episodic cost limit |
| $\hat{\rho}$ | `metrics.estimate_rho_hat` | stable distribution from late steps $k\ge k_0$ of held-out episodes (Section 7.1) |
| $\mathbb{D}(\rho_k\|\hat{\rho})$ | `metrics.stability_curve` | evaluation dissimilarity, 1-Wasserstein (Section 7.1) |
| sampling floor | `metrics.sampling_floor` | value of $\mathbb{D}(\rho_k\|\hat{\rho})$ for a stationary surrogate (Appendix D) |
""")
code(r"""
spec = ENVIRONMENTS[ENV]
H = EPISODE_LENGTH
z = norm.ppf(1 - COST_LIMIT / H)
print(f"environment          : {spec['env_id']}")
print(f"velocity threshold   : v_lim = {spec['v_limit']:.4f}")
print(f"cost limit           : c_bar = {COST_LIMIT:.0f} per episode of H = {H} steps")
print(f"target distribution  : rho_star = N({spec['mu_star']:.3f}, {spec['sd_star']:.2f}^2),  "
      f"mu_star = v_lim - z_{{1 - c_bar/H}} sd_star = {spec['v_limit']:.4f} - {z:.3f} x {spec['sd_star']:.2f}")
expected_cost = H * norm.sf(spec["v_limit"], spec["mu_star"], spec["sd_star"])
expected_cost_slow = H * norm.sf(spec["v_limit"], spec["mu_star_slow"], spec["sd_star"])
print(f"expected cost under rho_star: H * P(v > v_lim) = {expected_cost:.1f}")
print(f"slow target          : N({spec['mu_star_slow']:.2f}, {spec['sd_star']:.2f}^2), expected cost {expected_cost_slow:.2e}")
cfg = make_config(ENV, "DMDP", 0)
print("\ntraining configuration (shared by all methods):")
method_specific = ("env", "method", "seed", "policy_input", "lagrangian", "beta", "train_noise")
print({k: v for k, v in cfg.to_dict().items() if k not in method_specific})
pd.DataFrame(METHODS).T.loc[ALL]
""")

md(r"""
## 3 Training (optional)

Set `RUN_TRAINING = True` to reproduce all runs from scratch. Each run trains one policy for $3\times10^6$ environment steps and then
evaluates it on 64 episodes for every noise level $\nu\in\{0,0.1,0.2,0.3,0.4\}$. Finished runs are skipped
(delete `evaluation.npz` of a run to re-evaluate its saved policy without retraining).
The same commands can be run in a terminal:
`python run_all.py --envs HalfCheetah --tag main --workers 14`, and
`python run_all.py --methods DMDP --tag beta4 --beta 4` (likewise `--tag beta025 --beta 0.25`) for the sensitivity to $\beta$.
""")
code(r"""
RUN_TRAINING = False
if RUN_TRAINING:
    cmd = [sys.executable, "run_all.py", "--envs", ENV, "--methods", *ALL, "--seeds", *map(str, SEEDS),
           "--tag", TAG, "--workers", "14"]
    subprocess.run(cmd, check=True)
    for beta, tag in BETA_TAGS.items():                       # sensitivity to the weight beta (DMDP only)
        subprocess.run([sys.executable, "run_all.py", "--envs", ENV, "--methods", "DMDP", "--seeds", *map(str, SEEDS),
                        "--tag", tag, "--beta", str(beta), "--workers", "4"], check=True)
missing = [(m, s) for m in ALL for s in SEEDS if load_evaluation(ROOT, ENV, m, s) is None]
print("missing runs:", missing if missing else "none")
""")

md(r"""
## 4 Evaluation summary

For each policy and noise level, $\rho_k$ is estimated from blocks of 10 consecutive steps across one half of the 64 evaluation episodes,
and $\hat{\rho}$ from the steps $k\ge 500$ of the other half; the two assignments are averaged (cross-fitting, Section 7.1).
`D_late` is the mean of $\mathbb{D}(\rho_k\|\hat{\rho})$ over $k\ge 500$ (residual non-stationarity),
`D_early` the mean over $k<200$ (transient), `D_target` $=\mathbb{D}(\hat{\rho}\|\rho_\star)$,
and `D_shift` $=\mathbb{D}(\hat{\rho}_\nu\|\hat{\rho}_0)$, the shift of the stable distribution caused by sensor noise.
All dissimilarities in the evaluation are 1-Wasserstein distances.
""")
code(r"""
df, curves = summarize(ROOT, ENV, ALL, SEEDS, k0=K0, block=BLOCK, k_early=K_EARLY)
mean, sem = aggregate(df)
cols = ["ret", "cost", "D_late", "D_early", "D_target", "D_shift", "mu_hat", "sd_hat"]
fmt = mean[cols].round(3).astype(str) + " ± " + sem[cols].round(3).astype(str)
fmt.loc[(ALL, [0.0, 0.2]), :]
""")

md("## 5 Figures")
md(r"""
### 5.1 Convergence of $\rho_k$ to the stable distribution $\hat{\rho}$ (Section 7.2)

Mean over seeds (line) and standard error (band) of $\mathbb{D}(\rho_k\|\hat{\rho})$ against the step $k$, without noise and with $\nu=0.2$.
""")
code(r"""
def curve_stats(method, nu):
    Ds = [curves[(method, s, nu)][1] for s in SEEDS if (method, s, nu) in curves]
    c = curves[(method, SEEDS[0], nu)][0]
    Ds = np.array(Ds)
    return c, np.nanmean(Ds, 0), np.nanstd(Ds, 0, ddof=1) / np.sqrt(len(Ds))


def plot_curves(methods, nus, name, ylog=True):
    fig, axes = plt.subplots(1, len(nus), figsize=(TEXTWIDTH, 2.0), sharey=True)
    for ax, nu in zip(np.atleast_1d(axes), nus):
        for m in methods:
            c, mu, se = curve_stats(m, nu)
            ax.plot(c, mu, color=COLORS[m], lw=1.2, label=LABEL[m])
            ax.fill_between(c, mu - se, mu + se, color=COLORS[m], alpha=0.2, lw=0)
        ax.set_xlabel(S_K)
        ax.set_title(rf"$\nu={nu:g}$", fontsize=9)
        if ylog:
            ax.set_yscale("log")
        ax.axvline(K0, color="k", lw=0.5, ls=":")
    np.atleast_1d(axes)[0].set_ylabel(S_D_RHOK)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(methods))
    fig.tight_layout(w_pad=0.8)
    save_figure(fig, name)
    return fig


plot_curves(MAIN, [0.0, 0.2], "stability_curves");
""")

md(r"""
### 5.2 Stable distributions $\hat{\rho}$ and the target $\rho_\star$ (Section 7.2)

Histogram of $v_k$ for $k\ge 500$, pooled over all evaluation episodes and seeds ($\nu=0$), with the target $\rho_\star$ and the threshold $v_{\mathsf{lim}}$.
""")
code(r"""
def pooled_late(method, nu):
    out = []
    for s in SEEDS:
        ev = load_evaluation(ROOT, ENV, method, s)
        if ev is not None:
            out.append(estimate_rho_hat(ev[nu]["velocity"], K0))
    return np.concatenate(out)


def plot_stable_distributions(methods, name, nu=0.0):
    fig, axes = plt.subplots(1, len(methods), figsize=(TEXTWIDTH, 1.7), sharey=True, sharex=True)
    grid = np.linspace(-1.5, 6, 400)
    for ax, m in zip(axes, methods):
        v = pooled_late(m, nu)
        ax.hist(v, bins=np.linspace(-1.5, 6, 61), density=True, color=COLORS[m], alpha=0.55, lw=0, label=S_RHOHAT)
        cfg = load_config(ROOT, ENV, m, SEEDS[0])
        ax.plot(grid, norm.pdf(grid, cfg["mu_star"], cfg["sd_star"]), "k--", lw=1, label=S_RHOSTAR)
        ax.axvline(spec["v_limit"], color="k", lw=0.8, ls=":")
        ax.set_title(LABEL[m], fontsize=8.5)
        ax.set_xlabel(S_V)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, loc="upper left")
    save_figure(fig, name)
    return fig


plot_stable_distributions(MAIN, "stable_distributions");
""")

md(r"""
### 5.3 Sensor noise (Section 7.3)

Return, cost, residual non-stationarity $\mathbb{D}(\rho_k\|\hat{\rho})$ for $k\ge500$, and distance of the stable distribution to the target, against $\nu$.
Training used $\nu\sim\mathcal{U}[0,0.3]$, so $\nu=0.4$ lies outside the training range.
""")
code(r"""
def plot_noise(methods, name):
    panels = [("ret", "episodic return"), ("cost", "episodic cost"),
              ("D_late", S_D_RHOK + r", $k\geq 500$"), ("D_target", S_D_TARGET)]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, 1.75))
    for ax, (key, lab) in zip(axes, panels):
        for m in methods:
            mu, se = mean.loc[m][key], sem.loc[m][key]
            ax.errorbar(mu.index, mu.values, yerr=se.values, color=COLORS[m], lw=1.1, ms=3, marker="o",
                        capsize=1.5, label=LABEL[m])
        if key == "cost":
            ax.axhline(COST_LIMIT, color="k", lw=0.6, ls="--")
            ax.set_yscale("symlog", linthresh=10)
            ax.set_ylim(bottom=0)
        ax.set_xlabel(S_NU)
        ax.set_title(lab, fontsize=8.5)
        ax.axvspan(0.3, 0.4, color="0.9", lw=0, zorder=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.6)
    fig.legend(h, l, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(methods))
    save_figure(fig, name)
    return fig


plot_noise(MAIN, "noise_sweep");
""")

md(r"""
### 5.4 Stability does not imply reward (Section 7.2)

DMDP with the task-derived target $\rho_\star$ and with a slow target. Both closed loops settle at a stable distribution,
but only the task-derived target yields a high return.
""")
code(r"""
def plot_targets(name, nu=0.0):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.8), gridspec_kw=dict(width_ratios=[1.3, 1.3, 0.8]))
    for m in TARGETS:
        c, mu, se = curve_stats(m, nu)
        axes[0].plot(c, mu, color=COLORS[m], lw=1.2, label=LABEL[m])
        axes[0].fill_between(c, mu - se, mu + se, color=COLORS[m], alpha=0.2, lw=0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(S_K)
    axes[0].set_ylabel(S_D_RHOK)
    axes[0].legend(frameon=False)
    grid = np.linspace(-1.5, 6, 400)
    for m in TARGETS:
        v = pooled_late(m, nu)
        cfg = load_config(ROOT, ENV, m, SEEDS[0])
        axes[1].hist(v, bins=np.linspace(-1.5, 6, 61), density=True, color=COLORS[m], alpha=0.5, lw=0)
        axes[1].plot(grid, norm.pdf(grid, cfg["mu_star"], cfg["sd_star"]), color=COLORS[m], ls="--", lw=1)
    axes[1].axvline(spec["v_limit"], color="k", lw=0.8, ls=":")
    axes[1].set_xlabel(S_V)
    axes[1].set_ylabel("density")
    axes[1].set_title(S_RHOHAT + r" (filled) and " + S_RHOSTAR + r" (dashed)", fontsize=8.5)
    mu_r = [mean.loc[(m, nu), "ret"] for m in TARGETS]
    se_r = [sem.loc[(m, nu), "ret"] for m in TARGETS]
    axes[2].bar(range(len(TARGETS)), mu_r, yerr=se_r, color=[COLORS[m] for m in TARGETS], capsize=2)
    axes[2].set_xticks(range(len(TARGETS)), [r"task $\rho_\star$", r"slow $\rho_\star$"])
    axes[2].set_ylabel("episodic return")
    fig.tight_layout(w_pad=0.8)
    save_figure(fig, name)
    return fig


plot_targets("targets");
""")

md("### 5.5 Ablation: distribution-parameter input and dissimilarity term (Section 7.4, Appendix D)")
code(r"""
plot_curves(ABLATION, [0.0, 0.2], "ablation_curves");
plot_noise(ABLATION, "ablation_noise");
""")

md(r"""
### 5.6 Training curves (Appendix D)

Episodic return, episodic cost, and training dissimilarity during training, smoothed over ten updates.
""")
code(r"""
def plot_training(methods, name):
    keys = [("ep_return", "episodic return"), ("ep_cost", "episodic cost"),
            ("ep_dissimilarity", r"$\mathbb{D}(\rho_{\bar{\omega}_t}\|\rho_\star)$ (episode mean)")]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.8))
    for m in methods:
        runs = [load_training(ROOT, ENV, m, s) for s in SEEDS]
        runs = [r for r in runs if r is not None]
        n = min(len(r) for r in runs)
        x = runs[0]["step"].values[:n] / 1e6
        for ax, (key, lab) in zip(axes, keys):
            Y = np.array([r[key].values[:n] for r in runs])
            Y = pd.DataFrame(Y.T).rolling(10, min_periods=1).mean().values.T
            mu, se = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y))
            ax.plot(x, mu, color=COLORS[m], lw=1.1, label=LABEL[m])
            ax.fill_between(x, mu - se, mu + se, color=COLORS[m], alpha=0.2, lw=0)
            ax.set_title(lab, fontsize=8.5)
            ax.set_xlabel(r"environment steps ($\times 10^6$)")
    axes[1].axhline(COST_LIMIT, color="k", lw=0.6, ls="--")
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].set_ylim(bottom=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.6)
    fig.legend(h, l, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    save_figure(fig, name)
    return fig


plot_training(ALL, "training_curves");
""")

md(r"""
## 6 Tables

Mean ± standard error over the five seeds. The LaTeX tables are written to the output folder (`figures/` by default) and are included in the paper with `\input`:
`table_main.tex` (Section 7.2); `table_ablation.tex`, `table_shift.tex`, `table_beta.tex`, and `table_floor.tex` (Appendix D).
""")
code(r"""
def fmt_pm(m, s, d=2):
    return f"${m:.{d}f}\\pm{s:.{d}f}$"


def latex_table(methods, nus, name):
    rows = []
    for m in methods:
        for nu in nus:
            mu, se = mean.loc[(m, nu)], sem.loc[(m, nu)]
            rows.append(" & ".join([
                LABEL[m] if nu == nus[0] else "", f"${nu:g}$",
                fmt_pm(mu.ret, se.ret, 0), fmt_pm(mu.cost, se.cost, 1),
                fmt_pm(mu.D_early, se.D_early), fmt_pm(mu.D_late, se.D_late, 3),
                fmt_pm(mu.D_target, se.D_target)]) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    head = (r"\begin{tabular}{llccccc}" "\n" r"\toprule" "\n"
            r"Method & $\nu$ & return $\uparrow$ & cost ($\bar{c}=25$) & $\mathbb{D}(\rho_k\|\hat{\rho})$, $k<200$ & "
            r"$\mathbb{D}(\rho_k\|\hat{\rho})$, $k\geq500$ & $\mathbb{D}(\hat{\rho}\|\rho_\star)$ \\" "\n" r"\midrule")
    tex = head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    save_table(name, tex)
    return tex


print(latex_table(MAIN + ["DMDP-slow"], [0.0, 0.2], "table_main.tex"))
print(latex_table(ABLATION, [0.0, 0.2], "table_ablation.tex"))

# shift of the stable distribution caused by sensor noise, D(rho_hat(nu) || rho_hat(0))
lines = [" & ".join([LABEL[m]] + [fmt_pm(mean.loc[(m, nu), "D_shift"], sem.loc[(m, nu), "D_shift"])
                                  for nu in [0.1, 0.2, 0.3, 0.4]]) + r" \\"
         for m in ALL]
tex = (r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
       r"Method & $\nu=0.1$ & $\nu=0.2$ & $\nu=0.3$ & $\nu=0.4$ \\" "\n" r"\midrule" "\n"
       + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
save_table("table_shift.tex", tex)
print(tex)
""")

md(r"""
### 6.1 Sensitivity to the weight $\beta$ (Section 7.4, Appendix D)

DMDP with $\beta\in\{0.25,1,4\}$ (five seeds each). $\beta$ trades the task reward against the dissimilarity to $\rho_\star$.
""")
code(r"""
rows = []
beta_roots = [(0.25, os.path.join(HERE, "results", BETA_TAGS[0.25])), (1.0, ROOT),
              (4.0, os.path.join(HERE, "results", BETA_TAGS[4.0]))]
for beta, root in beta_roots:
    d, _ = summarize(root, ENV, ["DMDP"], SEEDS, k0=K0, block=BLOCK, k_early=K_EARLY)
    d["beta"] = beta
    rows.append(d)
dfb = pd.concat(rows)
mb = dfb.groupby(["beta", "nu"]).mean(numeric_only=True)
sb = dfb.groupby(["beta", "nu"]).sem(numeric_only=True)
lines = []
for beta in [0.25, 1.0, 4.0]:
    for nu in [0.0, 0.2]:
        if (beta, nu) not in mb.index:
            continue
        mu, se = mb.loc[(beta, nu)], sb.loc[(beta, nu)]
        lines.append(" & ".join([
            f"${beta:g}$" if nu == 0.0 else "", f"${nu:g}$", fmt_pm(mu.ret, se.ret, 0), fmt_pm(mu.cost, se.cost, 1),
            fmt_pm(mu.D_early, se.D_early), fmt_pm(mu.D_late, se.D_late, 3), fmt_pm(mu.D_target, se.D_target),
            fmt_pm(mu.mu_hat, se.mu_hat), fmt_pm(mu.sd_hat, se.sd_hat)]) + r" \\")
tex = (r"\begin{tabular}{llccccccc}" "\n" r"\toprule" "\n"
       r"$\beta$ & $\nu$ & return & cost & $\mathbb{D}(\rho_k\|\hat{\rho})$, $k<200$ & $\mathbb{D}(\rho_k\|\hat{\rho})$, $k\geq500$ & "
       r"$\mathbb{D}(\hat{\rho}\|\rho_\star)$ & mean of $\hat{\rho}$ & std of $\hat{\rho}$ \\" "\n" r"\midrule" "\n"
       + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
save_table("table_beta.tex", tex)
print(tex)
""")

md(r"""
## 7 Sanity checks

**Sampling floor of the evaluation metric (Appendix D).** With 32 episodes per half and blocks of 10 steps, even a perfectly stationary process has
$\mathbb{D}(\rho_k\|\hat{\rho})>0$ because of sampling. `metrics.sampling_floor` estimates this floor by applying the same estimator to a surrogate
that is stationary by construction but keeps the differences between episodes: every step of episode $i$ is an independent draw from the late-time
samples ($k\ge500$) of the same episode $i$. A ratio `D_late / floor` close to one means that the closed loop is stationary in the late steps up to sampling error.
""")
code(r"""
rng = np.random.default_rng(0)                  # one generator shared by all calls, in this fixed order
rows = []
for m in ALL:
    for nu in [0.0, 0.2, 0.4]:
        floors = [sampling_floor(load_evaluation(ROOT, ENV, m, s)[nu]["velocity"], K0, BLOCK, rng) for s in SEEDS]
        rows.append(dict(method=m, nu=nu, floor=np.mean(floors), D_late=mean.loc[(m, nu), "D_late"]))
floor = pd.DataFrame(rows).set_index(["method", "nu"])
floor["ratio"] = floor["D_late"] / floor["floor"]
lines = []
for m in ALL:
    cells = []
    for nu in [0.0, 0.2, 0.4]:
        r = floor.loc[(m, nu)]
        cells += [f"${r.floor:.3f}$", f"${r.D_late:.3f}$", f"${r.ratio:.2f}$"]
    lines.append(" & ".join([LABEL[m]] + cells) + r" \\")
tex = (r"\begin{tabular}{lccccccccc}" "\n" r"\toprule" "\n"
       r" & \multicolumn{3}{c}{$\nu=0$} & \multicolumn{3}{c}{$\nu=0.2$} & \multicolumn{3}{c}{$\nu=0.4$} \\" "\n"
       r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}" "\n"
       r"Method & floor & late & ratio & floor & late & ratio & floor & late & ratio \\" "\n" r"\midrule" "\n"
       + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
save_table("table_floor.tex", tex)
floor.round(3)
""")
md(r"""
**Per-seed details quoted in Section 7.2 and Section 7.3.** Noise-free stable distribution per seed, its shift at $\nu=0.4$, and the decomposition
of the spread of $v_k$ ($k\ge500$, $\nu=0$) into the average within-episode standard deviation and the standard deviation of the episode means.
""")
code(r"""
rows = []
for m in ALL:
    for s in SEEDS:
        v = load_evaluation(ROOT, ENV, m, s)[0.0]["velocity"][:, K0:]
        r0 = df[(df.method == m) & (df.seed == s) & (df.nu == 0.0)].iloc[0]
        r4 = df[(df.method == m) & (df.seed == s) & (df.nu == 0.4)].iloc[0]
        rows.append(dict(method=m, seed=s, mu_hat=r0.mu_hat, sd_hat=r0.sd_hat, shift_nu04=r4.D_shift,
                         within_sd=np.mean(np.std(v, axis=1)), between_sd=np.std(np.mean(v, axis=1))))
per_seed = pd.DataFrame(rows).set_index(["method", "seed"]).round(2)
per_seed.loc[["PPO-Lag", "DMDP", "DMDP-slow"]]
""")

md(r"""
**Episode survival.** HalfCheetah has no termination, so all evaluation episodes last $H=1000$ steps; the check below confirms this.
""")
code(r"""
df.groupby("method")["survival"].min()
""")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "torch_safety", "display_name": "torch_safety", "language": "python"}
    nb.cells = CELLS
    nbf.write(nb, NOTEBOOK)
    print(f"wrote {os.path.basename(NOTEBOOK)} with {len(CELLS)} cells")


if __name__ == "__main__":
    main()
