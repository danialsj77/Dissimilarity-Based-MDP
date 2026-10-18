# Dissimilarity-Based Markov Decision Processes for Stabilizing State Distributions

This repository contains the code, the trained policies, and the analysis notebook for the empirical validation of the paper
*Dissimilarity-Based Markov Decision Processes for Stabilizing State Distributions*.
It covers Section 4.5 (Practical Implementation), Section 5 (Empirical Validation), and Appendices E and F (Implementation Details and Additional Results) on the
Safety-Gymnasium HalfCheetah velocity task.
The theory of Sections 2–4 is not implemented for this task; `dmdp_toy/` computes its certificates for a scalar example (Appendix D, see [Computable example](#computable-example-appendix-d)).

## Overview

**The idea.** A Dissimilarity-based Markov decision process (DMDP) controls the *state distribution* $\rho_k$ of a stochastic
system, not a single state $\mathbf{s}_k$.
The stage cost is a dissimilarity $\mathbb{D}(\rho_k\Vert\rho_\star)$ between $\rho_k$ and a target distribution $\rho_\star$, for example a
Kullback–Leibler divergence or a Wasserstein distance.
The policy $\mathbf{a}_k=\psi(\rho_k)$ takes the distribution as input (*distribution feedback*).
For a parameterized family $\rho_\omega$, it takes the finite-dimensional parameter $\omega_k$ as input (*parameter feedback*).
Stability is defined on the space of probability measures.
The closed loop is $\mathbb{D}$-asymptotically stable with respect to $\rho_\star$ if it is $\mathbb{D}$-stable and $\mathbb{D}(\rho_k\Vert\rho_\star)\to0$.
Sections 2–4 give sufficient conditions, based on Lyapunov certificates, under which distribution or parameter feedback
makes the distribution dynamics $\mathbb{D}$-asymptotically stable.
These results cover the optimal policy and policies that are learned offline, deployed online with estimated parameters, or learned without a model.

**What this repository contains.** This repository contains the practical implementation of Section 4.5 and Appendix E and its evaluation.
None of the Lyapunov certificates is computed or verified for this implementation.
The experiments measure distributional stability empirically.

| Component | Practical implementation |
|---|---|
| Task | `SafetyHalfCheetahVelocity-v1` of Safety-Gymnasium, used through its Gymnasium interface `SafetyHalfCheetahVelocityGymnasium-v1`. Episodes last $H=1000$ steps, and the cost is $c_t=\mathbf{1}[v_t>v_{\mathrm{lim}}]$ (Appendix E, Table 7) |
| Regulated distribution | $\rho_k$ is the joint distribution of two task features across episodes: the forward velocity $v_k$ and the pitch angle $\theta_k$ of the torso (Appendix E) |
| Target | $\rho_\star$ is the product of $U[0.70\,v_{\mathrm{lim}},0.95\,v_{\mathrm{lim}}]=U[2.25,3.05]$ for the velocity (fast, with a margin of 5% below the threshold, so the cost under $\rho_\star$ is zero; `TARGET_BAND`) and $U[-0.2,0.2]$ rad for the pitch (upright torso; `PITCH_HALF_WIDTH`) (Appendix E) |
| Parameterization | product of the location-scale families of the two factors, $\omega=(\mu^v,\varsigma^v,\mu^\theta,\varsigma^\theta)$ |
| Dissimilarity | $\mathbb{D}_t=W_2^2(\text{last }n_{\mathrm{w}}\text{ true velocities},\rho^v_\star)+W_2^2(\text{last }n_{\mathrm{w}}\text{ true pitch angles},\rho^\theta_\star)$, exact squared 2-Wasserstein distances of the empirical window distributions (`dmdp_sg/targets.py`, `envs.DistributionEstimator`) |
| Policy input | $\hat\omega_t$, the window means and standard deviations of the last $n_{\mathrm{w}}=10$ noisy readings of the velocity and of the pitch of one trajectory. The DMDP policy, also used by Implicit DMDP (PPO), Implicit DMDP-Lag (PPO), Implicit DMDP (SAC), and Implicit DMDP-Lag (SAC), is $\phi(\hat\omega_t,\mathbf{o}_t;\theta)$ (Section 4.5) |
| Sensor noise | $\mathbf{o}_t=\mathbf{y}_t+\eta\boldsymbol{\xi}_t$, where the entries of $\boldsymbol{\xi}_t$ are independent draws from the bimodal Gaussian mixture $\tfrac12\mathcal{N}(-0.8,0.6^2)+\tfrac12\mathcal{N}(0.8,0.6^2)$ (mean 0, variance 1), with $\eta\sim\mathcal{U}[0,0.3]$ drawn for each training episode and $\eta\in\lbrace0,0.1,\ldots,0.5\rbrace$ in the evaluation (Section 4.5, Table 7) |
| Rewards | Every method is trained on the benchmark reward $r^{\mathrm{b}}_t=v_t-0.1\Vert\mathbf{a}_t\Vert^2$ (`cost_penalty` $=0$ in `METHODS`). PPO, Distributional PPO, Implicit DMDP (PPO), SAC, Distributional SAC, and Implicit DMDP (SAC) use $r^{\mathrm{b}}_t$ alone; PPO-Lag, Distributional PPO-Lag, Implicit DMDP-Lag (PPO), SAC-Lag, Distributional SAC-Lag, and Implicit DMDP-Lag (SAC) add $-\lambda c_t$; the two DMDP methods add the threshold reward $r_{t\mathbb{D}}$ and the dissimilarity term $-\beta\mathbb{D}_t$ (and $-\lambda c_t$ with the cost constraint). The reported return $R$ is the return of the reward each method is trained on, without $-\beta\mathbb{D}_t$ and $-\lambda c_t$: $R=\sum_t r^{\mathrm{b}}_t$ for the baselines and $R=\sum_t(r^{\mathrm{b}}_t+r_{t\mathbb{D}})$ for the DMDP methods (`analysis.episode_returns`); `analysis.benchmark_returns` gives the benchmark return $\sum_t r^{\mathrm{b}}_t$ of every method (Section 5 and Appendix E). Only the design pilot was trained with the cost penalty $(2c_t)^2$ subtracted from the reward (`envs.CostPenalty`, `COST_PENALTY` $=2$; every run in `results/pilot/` except `baselines_c10_benchmark_reward/`) |
| Threshold reward | $r_{t\mathbb{D}}=+d$ if $\mathbb{D}_t\le\mathbb{D}_{\mathrm{thr}}$ and $0$ otherwise, $d=10$ (`envs.ThresholdReward`, `D_BONUS`), $\mathbb{D}_{\mathrm{thr}}=0.1$ (`D_THRESHOLD`; 0.14 in the design pilot) for every window length; only for the two DMDP methods (Appendix E) |
| Objective | DMDP maximizes the return of $r^{\mathrm{b}}_t+r_{t\mathbb{D}}-\beta\mathbb{D}_t$ with $\beta=0.5$; it has no cost constraint, so the safety requirement enters only through the margin of the velocity target. DMDP-Lag maximizes the return of $r^{\mathrm{b}}_t+r_{t\mathbb{D}}-\beta\mathbb{D}_t-\lambda c_t$ and enforces the episodic cost constraint $J_c\le\bar c$ with $\bar c=10$ (Sections 4.5 and 5, Table 8). The -Lag baselines maximize the return of $r^{\mathrm{b}}_t-\lambda c_t$ under the same constraint |
| Learner | PPO with separate reward and cost critics, extended to PPO-Lagrangian for the methods with the cost constraint (SAC as the second learner: see "Second learner: SAC"). The training reward of every method is divided by the running standard deviation of its discounted return (`train.ReturnScaler`), and the policy is updated with the advantage $(\hat A^r_t-\lambda\hat A^c_t)/(1+\lambda)$ with a standardized reward advantage (`train.combine_advantages`); the standard deviation of the actions is initialized at $e^{-1}\approx0.37$ for every method (`INIT_LOG_STD`; with the PPO default of 1, the first rollouts flip the torso); the values are bootstrapped at the time limit, and the gradient norm is clipped separately for each network (Appendix E, Table 8). The Distributional and Implicit DMDP methods of both learners, with and without -Lag, have distributional reward critics that learn 32 quantiles of the return by quantile regression (`N_QUANTILES`, `agent.quantile_huber_loss`); their policies use the mean of the quantiles, and the cost critic of their -Lag variants learns the expected cost |
| Evaluation | $\mathbb{D}(\rho_k\Vert\hat\rho)$, the 1-Wasserstein distance between $\rho_k$ at a single step $k$ and the stable distribution $\hat\rho$. Both are estimated from 64 evaluation episodes, with $\rho_k$ and $\hat\rho$ taken from disjoint halves of the episodes (Section 5) |

The main study compares eight methods (`dmdp_sg/config.py`, `METHODS`; SAC, SAC-Lag, Distributional SAC, Distributional SAC-Lag, Implicit DMDP (SAC), and Implicit DMDP-Lag (SAC) belong to the SAC study, see "Second learner: SAC").
In the main study, all methods use the same PPO learner and hyperparameters.

| Code name | Name in the paper | Input of policy and critics | Reward | Critics |
|---|---|---|---|---|
| `PPO` | PPO | $\mathbf{o}_t$ | $r^{\mathrm{b}}_t$ | $V_r$ |
| `PPO-Lag` | PPO-Lag | $\mathbf{o}_t$ | $r^{\mathrm{b}}_t-\lambda c_t$ | $V_r$, $V_c$ |
| `Dist-PPO` | Distributional PPO | $\mathbf{o}_t$ | $r^{\mathrm{b}}_t$ | $V_r$ (mean of 32 quantiles of the return) |
| `Dist-PPO-Lag` | Distributional PPO-Lag | $\mathbf{o}_t$ | $r^{\mathrm{b}}_t-\lambda c_t$ | $V_r$ (mean of 32 quantiles of the return), $V_c$ |
| `Implicit-PPO` | Implicit DMDP (PPO) | $(\hat\omega_t,\mathbf{o}_t)$ | $r^{\mathrm{b}}_t$ | $V_r$ (mean of 32 quantiles of the return) |
| `Implicit-PPO-Lag` | Implicit DMDP-Lag (PPO) | $(\hat\omega_t,\mathbf{o}_t)$ | $r^{\mathrm{b}}_t-\lambda c_t$ | $V_r$ (mean of 32 quantiles of the return), $V_c$ |
| `DMDP` | DMDP (PPO) | $(\hat\omega_t,\mathbf{o}_t)$ | $r^{\mathrm{b}}_t+r_{t\mathbb{D}}-\beta\mathbb{D}_t$ | $V_r$ |
| `DMDP-Lag` | DMDP-Lag (PPO) | $(\hat\omega_t,\mathbf{o}_t)$ | $r^{\mathrm{b}}_t+r_{t\mathbb{D}}-\beta\mathbb{D}_t-\lambda c_t$ | $V_r$, $V_c$ |

$r^{\mathrm{b}}_t=v_t-0.1\Vert\mathbf{a}_t\Vert^2$ is the benchmark reward, the base of the reward of every method; $\mathbb{D}_t=\mathbb{D}(\bar\rho_t\Vert\rho_\star)$ is the window dissimilarity; $\beta=0.5$ (`BETA` in `dmdp_sg/config.py`).
The column Critics lists the critics whose advantages enter the policy update: the reward critic $V_r$ estimates the return of the reward of the method without the term $-\lambda c_t$, the cost critic $V_c$ that of $c_t$.
`Dist-PPO` is a baseline from distributional reinforcement learning: its reward critic outputs 32 quantiles of the return at the levels $(2i-1)/64$, trained with the quantile Huber loss ($\kappa=1$) toward $\lambda$-returns computed quantile by quantile (`train.compute_quantile_returns`), and $V_r$ is their mean, so the advantage is formed as in PPO.
`Implicit-PPO` is `Dist-PPO` with the policy input of the DMDP methods, $\phi(\hat\omega_t,\mathbf{o}_t)$ (21 inputs: the window means and standard deviations of the velocity and pitch readings and the 17 entries of $\mathbf{o}_t$): distribution feedback on the benchmark reward, without a target (no $r_{t\mathbb{D}}$, no $\mathbb{D}_t$) and without a cost constraint. It shows what distribution feedback contributes when the target $\rho_\star$ is unknown.
`Dist-PPO-Lag` and `Implicit-PPO-Lag` are `Dist-PPO` and `Implicit-PPO` with the cost constraint of `PPO-Lag`: they are trained on $r^{\mathrm{b}}_t-\lambda c_t$, the multiplier is updated as in `PPO-Lag`, and their cost critic learns the expected cost.

In every -Lag method (`PPO-Lag`, `Dist-PPO-Lag`, `Implicit-PPO-Lag`, `DMDP-Lag`, and their counterparts in the SAC study), the Lagrange multiplier $\lambda\ge0$ is adapted during training to enforce the cost limit $\bar c=10$ (`COST_LIMIT`).
`PPO`, `Dist-PPO`, `Implicit-PPO`, and `DMDP` are unconstrained: $\lambda=0$ throughout training, so their cost critic is trained but does not affect the policy.
Every method is trained on the benchmark reward $r^{\mathrm{b}}_t$, so the cost enters the training reward only through $-\lambda c_t$ of the -Lag methods; the cost penalty $(2c_t)^2$ was used only in the design pilot (`results/pilot/`).
`PPO` has no cost limit and serves as the unconstrained reference.
What DMDP receives and the baselines do not: the target $\rho_\star$ (a fast, safe velocity band and an upright torso), through the dissimilarity and the threshold reward. Its policy input, the window statistics $\hat\omega_t$, is shared with `Implicit-PPO` and `Implicit-PPO-Lag`, which receive distribution feedback without a target.

**How the setting was chosen.** An audit of the earlier designs found no correctness bug, but three reasons for the low benchmark return of DMDP: the earlier target $U[0,3.29]$ had mean velocity 1.65, which capped the benchmark return near 1250; most runs of every method learned to run flipped (on the back), a gait whose velocity oscillates between 0 and $v_{\mathrm{lim}}$; and the unscaled reward kept the critic from fitting the large DMDP rewards. The present setting (velocity band, pitch target, dense term in both DMDP methods, reward normalization, normalized Lagrangian advantage) was developed in pilot runs with Gaussian noise $\boldsymbol{\xi}_t\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ on the seeds 10, 11, and 12, which are not used in the reported study. With the noise of the study, $\beta$ and $\bar c$ were then chosen in a pilot of DMDP-Lag on the same seeds, by rules fixed before the pilot was run: $\beta=0.5$ as the one of $\lbrace0.5,1,2\rbrace$ (with $\bar c=10$) with the highest return without noise, and $\bar c=10$ as the largest of $\lbrace10,5,2.5\rbrace$ at this $\beta$ for which the pilot cost of DMDP-Lag was below that of PPO and PPO-Lag ($\bar c=10$) both without noise and with $\eta=0.2$ (`results/pilot/`). In this pilot, DMDP-Lag, PPO, and PPO-Lag were all trained with the cost penalty $(2c_t)^2$ subtracted from the reward (`results/pilot/baselines_c10/` for PPO and PPO-Lag). Every reported method is trained on the benchmark reward (the DMDP methods on $r^{\mathrm{b}}_t+r_{t\mathbb{D}}-\beta\mathbb{D}_t$), and $\beta=0.5$ and $\bar c=10$ were kept from the pilot. With the same runs of PPO and PPO-Lag on the benchmark reward (`results/pilot/baselines_c10_benchmark_reward/`), the rule would give $\bar c=5$ (notebook section 6.1 prints both). The pilot used the PPO default initial action std of 1. A first run on the reported seeds with Gaussian noise and this std had PPO and PPO-Lag flipped in most seeds, which confounds the comparison, so the reported study initializes the std at 0.37 for every method. The runs with Gaussian noise and the earlier pilot waves are archived outside this repository.

## Second learner: SAC (paper Section 5.1)

The eight methods are also trained with soft actor-critic (SAC, `dmdp_sg/sac.py`; hyperparameters in Table 8 of the paper): **SAC**, **SAC-Lag**, **Distributional SAC** (`Dist-SAC`), **Distributional SAC-Lag** (`Dist-SAC-Lag`), **Implicit DMDP (SAC)** (`Implicit-SAC`), and **Implicit DMDP-Lag (SAC)** (`Implicit-SAC-Lag`) take the places of PPO, PPO-Lag, Distributional PPO, Distributional PPO-Lag, Implicit DMDP (PPO), and Implicit DMDP-Lag (PPO) and, like them, are trained on the benchmark reward, and **DMDP** and **DMDP-Lag** are trained on the same rewards as with PPO. Environment, target, rewards, $\beta=0.5$, $\bar c=10$, $d=10$, $\mathbb{D}_{\mathrm{thr}}=0.1$, training noise, seeds, and evaluation are those of the PPO study; the multiplier uses the PPO rule and learning rate ($0.035$, every 2048 steps).

| Setting | Value |
|---|---|
| steps per run | $10^6$ (`SAC_TOTAL_STEPS`), 4 parallel environments, 4 gradient steps after every step of the 4 environments (one per environment step) |
| networks | actor and critics with two hidden ReLU layers of 256 units; twin reward critics with target networks ($\tau=0.005$); cost critic for the constrained methods |
| exploration | 5000 steps of uniformly random actions, then the tanh-Gaussian policy with automatic temperature (target entropy $-6$) |
| reward scaling | the replay buffer stores the unscaled training reward; it is divided by the running std of its discounted return when a minibatch is drawn |
| constraint | actor loss $(\alpha\log\pi-\min_i Q_i+\lambda Q_c)/(1+\lambda)$ (SAC-Lagrangian) |
| distributional critics (Distributional SAC and Implicit DMDP (SAC), with and without -Lag) | $Q_1$ and $Q_2$ output 32 quantiles of the return, trained with the quantile Huber loss toward $r+\gamma(\theta_j(s',a')-\alpha\log\pi(a'\vert s'))$, where $\theta_j$ are the quantiles of the target critic with the smaller mean; the means replace $Q_1$ and $Q_2$ in the actor loss |

The learner is in `dmdp_sg/sac.py` and its tests in `tests/test_sac.py` (the distributional critics of both learners in `tests/test_distributional.py`). The results of the SAC study are in `results/sac/HalfCheetah/<method>/seed<s>/`, with the files of the PPO runs (`training.csv` has an extra column `alpha`).
The figures and tables with the prefix `sac_` / `table_sac_` are produced by section 8 of the notebook; the commands that train the runs are listed under [Reproducing the paper](#reproducing-the-paper).

## Repository structure

```
.
├── dmdp_sg/                       Python package
│   ├── __init__.py                module-to-section map
│   ├── config.py                  task, target distribution, methods, and TrainConfig (Section 4.5 and Appendix E, Tables 6-8)
│   ├── targets.py                 target families of rho_star and the exact W2 of a window to rho_star (Appendix E)
│   ├── envs.py                    cost penalty (design pilot only), sensor noise, window estimate of the policy input, training dissimilarity, threshold reward (Section 4.5 and Appendix E)
│   ├── agent.py                   Gaussian actor with reward and cost critics, quantile Huber loss, input normalization (PPO learner, Appendix E)
│   ├── train.py                   PPO-Lagrangian training with the shaped reward, distributional critic of Distributional PPO and Implicit DMDP (PPO), with and without -Lag (Appendix E)
│   ├── sac.py                     SAC and SAC-Lagrangian training of the eight methods of the SAC study, the second learner (Appendix E and Section 5.1)
│   ├── evaluate.py                evaluation rollouts at fixed sensor-noise levels (Section 5)
│   ├── metrics.py                 stability metric, its stationary reference value (`sampling_floor`), distance to the target (Section 5 and Appendix F.2)
│   └── analysis.py                loading and aggregation of the stored results (Section 5)
├── run_all.py                     parallel launcher: trains and evaluates every (method, seed) run
├── build_notebook.py              generates DMDP_SafetyGymnasium.ipynb
├── build_code_notebooks.py        generates DMDP_PPO.ipynb and DMDP_SAC.ipynb from the sources in dmdp_sg/ (cutting and texts in code_notebooks/)
├── code_notebooks/                slicing.py cuts the modules at their top-level statements; shared.py, ppo.py, sac.py, and examples.py hold the texts and the example cells
├── plot_halfcheetah.py            draws the robot with its action, forward velocity, and pitch from two stored policies (Appendix E)
├── dmdp_toy/                      computable example of Sections 2-4 (Appendix D): system, grids, value iteration, GP, offline and online learners, learner from an estimated Lyapunov decrease, certificates, figure and tables
├── run_toy.py                     stage-by-stage launcher of the computable example (Appendix D)
├── results_toy/                   data of the computable example: true_value.npz, offline/L<l>/run_<i>.npz, online.npz, modelfree/run_<i>.npz, summary.json
├── results_toy_quick/             output of `run_toy.py --quick` (not in git)
├── DMDP_SafetyGymnasium.ipynb     analysis notebook: every figure and table of Section 5 and Appendix F (see Findings)
├── DMDP_PPO.ipynb                 the complete code of the PPO learner with explanations, small examples, and the stored results of results/main (see Code notebooks)
├── DMDP_SAC.ipynb                 the same for the SAC learner and results/sac
├── figures/                       figures (PDF and PNG) and LaTeX tables written by the notebook
├── results/
│   ├── main/HalfCheetah/<method>/seed<s>/              main study, PPO: 8 methods x 5 seeds (PPO, PPO-Lag, Dist-PPO, Dist-PPO-Lag, Implicit-PPO, Implicit-PPO-Lag, DMDP, DMDP-Lag), beta = 0.5, c_bar = 10
│   ├── sac/HalfCheetah/<method>/seed<s>/               SAC study: 8 methods x 5 seeds (SAC, SAC-Lag, Dist-SAC, Dist-SAC-Lag, Implicit-SAC, Implicit-SAC-Lag, DMDP, DMDP-Lag)
│   ├── pilot/<setting>/HalfCheetah/DMDP-Lag/seed<s>/   design pilot of DMDP-Lag on seeds 10-12 (beta, c_bar; initial std 1), trained with the cost penalty
│   ├── pilot/baselines_c10/HalfCheetah/<method>/seed<s>/  PPO and PPO-Lag (c_bar = 10) on the pilot seeds, trained with the cost penalty: the reference of the rule for c_bar
│   ├── pilot/baselines_c10_benchmark_reward/HalfCheetah/<method>/seed<s>/  the same runs on the benchmark reward (with them the rule would give c_bar = 5)
│   └── *.log                                           launcher logs (not in git)
├── tests/                         pytest suite (configuration, wrappers, metrics, analysis, both learners, distributional critics, launcher, stored runs, code notebooks, computable example)
├── requirements.txt               pinned Python dependencies
└── environment.yml                conda environment (Python 3.10 + requirements.txt)
```

Each run directory contains the following files:

| File | Content | In git |
|---|---|---|
| `config.json` | the `TrainConfig` of the run | yes |
| `training.csv` | PPO: one row per update, from the first update in which a training episode was completed: `step`, `ep_return`, `ep_task_return`, `ep_cost`, `ep_dissimilarity`, `ep_dissimilarity_median`, `ep_within`, `ep_pitch`, `ep_flipped`, `ep_length`, `lagrange_multiplier`, `seconds`. SAC: the same columns plus `alpha`, one row each time a batch of episodes is completed. The episode statistics are means over the most recent completed training episodes; `ep_return` is the return of $r^{\mathrm{b}}_t+r_{t\mathbb{D}}$ (without the term $-\beta\mathbb{D}_t$ of the training reward) and `ep_task_return` that of $r^{\mathrm{b}}_t$, the benchmark reward (the pilot runs with the cost penalty subtract $(2c_t)^2$ from both); they differ only for the two DMDP methods. The notebook plots `ep_return`, the reported return $R$. `ep_dissimilarity_median` is the mean of the episode medians of $\mathbb{D}_t$, which a minority of tilted or flipped episodes pulls far above the typical episode, so the training curves use `dis_median` of `episodes.csv` instead | yes |
| `episodes.csv` | one row per completed training episode: `step` (PPO: at the end of the update in which the episode was completed), `ret` (return of $r^{\mathrm{b}}_t+r_{t\mathbb{D}}$, without $-\beta\mathbb{D}_t$), `task_ret` (return of $r^{\mathrm{b}}_t$; in the pilot runs with the cost penalty, of $r^{\mathrm{b}}_t-(2c_t)^2$), `cost`, `dis` (mean window dissimilarity per step), `dis_median` (median of $\mathbb{D}_t$ over the steps of the episode; the third panel of the training curves shows its median and interquartile range over the recent training episodes of all seeds), `within` (fraction of steps within $\mathbb{D}_{\mathrm{thr}}$), `pitch` (mean absolute pitch), `flipped` (fraction of steps with the torso flipped, $|\theta_t|>\pi/2$). `ret`, the reported return $R$, is used for the reliability of the return at the end of training (the last 10% of the training steps); `task_ret`, the benchmark return, is used for the end-of-training episodes with a negative benchmark return (`negative` in the notebook) | yes |
| `policy.pt` | network weights, input-normalization statistics, configuration, and input/output dimensions | yes |
| `evaluation.npz` | arrays `velocity`, `reward`, `cost`, `dissimilarity`, `bonus` ($r_{t\mathbb{D}}$, zero for the baselines), and `pitch` for each noise level, with keys such as `nu0.20/velocity`, each of shape (64 episodes, 1000 steps) | no, regenerated from `policy.pt` (see [below](#2-evaluate-the-stored-policies)) |

Each `evaluation.npz` is about 11 MB (1.29 GB for the 113 runs in `results/`; about 0.46 GB each for the main PPO study and the SAC study), so these files are not stored in git.

## Installation

The code runs on the CPU.
It was developed and tested on macOS (Apple silicon) with Python 3.10.

```bash
git clone https://github.com/danialsj77/Dissimilarity-Based-MDP.git
cd Dissimilarity-Based-MDP
conda env create -f environment.yml          # creates the environment "dmdp" (Python 3.10 + requirements.txt)
conda activate dmdp
```

Alternatively, you can create the environment manually:

```bash
conda create -n dmdp python=3.10
conda activate dmdp
pip install -r requirements.txt
```

* `requirements.txt` pins the exact versions used for the paper, including PyTorch 2.4.1, Gymnasium 0.28.1, and MuJoCo 2.3.3.
  Safety-Gymnasium is installed from GitHub at commit `ae966e5`, so `git` must be available.
* MuJoCo comes with the `mujoco` wheel, so no separate MuJoCo installation is needed.
* The notebook metadata refers to the Jupyter kernel `torch_safety`.
  Register the environment under this name:
  ```bash
  python -m ipykernel install --user --name torch_safety --display-name torch_safety
  ```
* **LaTeX is optional.** It is used only for the fonts of the figures.
  If a `latex` executable is found, the notebook typesets the figure text with LaTeX in Latin Modern, as in the paper.
  This requires `latex`, `dvipng`, and the packages `lmodern`, `amsmath`, `amssymb`, and `bm`, which are included in MacTeX and TeX Live.
  If the MacTeX folder `/Library/TeX/texbin` exists, the notebook adds it to `PATH`.
  Without LaTeX, the notebook uses the mathtext renderer of matplotlib, and all numbers and tables are the same.
* Importing Safety-Gymnasium prints a warning about the AdroitHand environments of `gymnasium-robotics`.
  This warning does not affect the HalfCheetah task.

## Quick start

Run the tests from the repository root (less than a minute):

```bash
python -m pytest
```

The stored `evaluation.npz` files are not in git, so in a fresh clone the parts of `tests/test_results.py` that read them are skipped.
After [step 2](#2-evaluate-the-stored-policies), all 224 tests run.

Run a short training and evaluation of two methods (less than a minute):

```bash
python run_all.py --methods PPO DMDP --seeds 0 --steps 20000 --tag smoke --workers 2
```

This command writes `config.json`, `training.csv`, `episodes.csv`, `policy.pt`, and `evaluation.npz` to `results/smoke/HalfCheetah/{PPO,DMDP}/seed0/`.
`--steps` is rounded down to a multiple of 2048, the number of environment steps per PPO update (8 environments × 256 steps).
The `results/smoke/` directory is ignored by git.

The package can also be used from Python:

```python
from dmdp_sg.config import make_config
from dmdp_sg.evaluate import evaluate_policy
from dmdp_sg.train import train

cfg = make_config("HalfCheetah", "DMDP", seed=0, total_steps=50_000)   # keyword arguments override TrainConfig fields
train(cfg, "results/smoke/example")                                     # writes policy.pt, training.csv, episodes.csv, config.json
ev = evaluate_policy("results/smoke/example/policy.pt", noises=[0.0, 0.2], episodes=8)
ev["nu0.20/velocity"].shape                                             # (8, 1000): velocity v_k per episode and step
```

## Code notebooks

`DMDP_PPO.ipynb` and `DMDP_SAC.ipynb` show the complete code of the PPO and of the SAC learner, section by section: the target distributions, the configuration, the environment wrappers, the networks and the training loop of the learner, the evaluation, the metrics, and the analysis (Section 4.5 and Appendices E and F of the paper for PPO, Section 5.1 and Appendix E for SAC).
Each section explains its module in a few lines, and the code cells are copied without change from `dmdp_sg/`.
Section 10 of each notebook calls the code with small settings (the window dissimilarity, the wrappers, the distributional loss, the Lagrange multiplier, and a tiny training run with its evaluation), and Section 11 reads the stored runs of `results/main` or `results/sac` and recomputes a table from the training logs and the evaluation tables of the paper (the latter need the files `evaluation.npz`, see [below](#2-evaluate-the-stored-policies)).

The notebooks do not import `dmdp_sg`: their cells define everything, so each notebook runs on its own in the environment of `environment.yml`.

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 DMDP_PPO.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 DMDP_SAC.ipynb
```

After a change in `dmdp_sg/`, `python build_code_notebooks.py` rebuilds both notebooks without outputs, and the commands above execute them again.
`tests/test_code_notebooks.py` fails if a stored notebook differs from the current build, and it checks that the code of a notebook trains and evaluates a policy exactly as the package does.

## Reproducing the paper

`run_all.py` skips finished runs.
Training is skipped if `policy.pt` exists, and evaluation is skipped for the noise levels already stored in `evaluation.npz`.
A stored run whose `config.json` differs from the requested configuration (for example after a change of `METHODS`) is reported as failed instead of being reused, and `run_all.py` exits with an error if any run failed.
Because the policies of all runs are stored in `results/`, the commands in step 1 only evaluate them (step 2) unless the stored runs are removed first.
To regenerate the numbers of the paper from the stored policies, go to step 2.

### 1. Train and evaluate all runs

Each command writes the runs of one results folder; `--workers` is the number of runs in parallel and does not change the results.

Main study (PPO; Appendix F.2): 8 methods (`PPO`, `PPO-Lag`, `Dist-PPO`, `Dist-PPO-Lag`, `Implicit-PPO`, `Implicit-PPO-Lag`, `DMDP`, `DMDP-Lag`, the default of `run_all.py`) × 5 seeds × $3\times10^6$ environment steps, written to `results/main/`:

```bash
python run_all.py --workers 18
```

Second learner (SAC; Section 5.1 and Appendix F.1): 8 methods × 5 seeds × $10^6$ environment steps, written to `results/sac/`:

```bash
python run_all.py --learner sac --methods SAC SAC-Lag Dist-SAC Dist-SAC-Lag Implicit-SAC Implicit-SAC-Lag DMDP DMDP-Lag --tag sac --workers 20
```

Design pilot of DMDP-Lag on the seeds 10 to 12 (initial std 1), trained with the cost penalty $(2c_t)^2$ subtracted from the reward (`--cost-penalty 2`), written to `results/pilot/<setting>/` (notebook section 6.1):

```bash
python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta0.5_c10 --beta 0.5 --cost-limit 10 --init-log-std 0 --cost-penalty 2 --workers 3
python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta1_c10 --beta 1 --cost-limit 10 --init-log-std 0 --cost-penalty 2 --workers 3
python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta2_c10 --beta 2 --cost-limit 10 --init-log-std 0 --cost-penalty 2 --workers 3
python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta1_c5 --beta 1 --cost-limit 5 --init-log-std 0 --cost-penalty 2 --workers 3
python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta1_c2.5 --beta 1 --cost-limit 2.5 --init-log-std 0 --cost-penalty 2 --workers 3
python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta0.5_c5 --beta 0.5 --cost-limit 5 --init-log-std 0 --cost-penalty 2 --workers 3
python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta0.5_c2.5 --beta 0.5 --cost-limit 2.5 --init-log-std 0 --cost-penalty 2 --workers 3
# reference of the rule for c_bar: PPO and PPO-Lag (c_bar = 10) on the same seeds, trained, as in the pilot that chose
# c_bar = 10, with the cost penalty of the pilot runs of DMDP-Lag
python run_all.py --methods PPO PPO-Lag --seeds 10 11 12 --tag pilot/baselines_c10 --cost-limit 10 --init-log-std 0 --cost-penalty 2 --workers 6
# the same runs with the benchmark reward of the reported baselines (with them the rule would give c_bar = 5; c_bar = 10 was kept)
python run_all.py --methods PPO PPO-Lag --seeds 10 11 12 --tag pilot/baselines_c10_benchmark_reward --cost-limit 10 --init-log-std 0 --workers 6
```

* Each run uses one CPU core.
* The study ran on an 18-core Apple M5 Pro; the column `seconds` of `training.csv` gives the training time of each run.
  The runs of PPO, PPO-Lag, SAC, and SAC-Lag (and of the pilot on the benchmark reward) were trained with 7 runs in parallel; there, the training of one PPO or PPO-Lag run took 10 to 11 minutes, that of one SAC run 56 to 58 minutes, and that of one SAC-Lag run 74 to 83 minutes.
  The Distributional PPO and Distributional SAC runs were trained after them, with 10 runs in parallel at first; the training of one Distributional PPO run took about 17 minutes and that of one Distributional SAC run about 128 minutes.
  The runs of Implicit DMDP (PPO) and Implicit DMDP (SAC) were trained next, with 15 runs in parallel (together with earlier runs of the DMDP methods), first the runs with PPO and then those with SAC; the training of one run took about 20 minutes for Implicit DMDP (PPO) and about 180 minutes for Implicit DMDP (SAC).
  The runs of Distributional PPO-Lag, Implicit DMDP-Lag (PPO), Distributional SAC-Lag, and Implicit DMDP-Lag (SAC) were trained last, with 10 runs in parallel, first the 10 runs with PPO and then the 10 runs with SAC. With PPO, the training of one run took about 18 minutes for Distributional PPO-Lag and about 19 minutes for Implicit DMDP-Lag (PPO); with SAC, it took 181 to 190 minutes for Distributional SAC-Lag and 186 to 192 minutes for Implicit DMDP-Lag (SAC).
  The reported runs of DMDP and DMDP-Lag, with $\mathbb{D}_{\mathrm{thr}}=0.1$, were trained after all baselines, with 10 runs in parallel, first the 10 runs with PPO and then the 10 runs with SAC (with `--d-threshold 0.1`, before 0.1 became the default `D_THRESHOLD`); the training of one run took about 14 minutes with PPO and, with SAC, about 77 minutes for DMDP and 93 minutes for DMDP-Lag. The earlier runs of the two DMDP methods with $\mathbb{D}_{\mathrm{thr}}=0.14$ (and a PPO trial with $0.07$) are kept in `results/archive_dthr_2026-10-09/`.
* `--beta` changes $\beta$ only for the methods with $\beta>0$.
  `--window` changes the window length $n_{\mathrm{w}}$.
  `--d-threshold` changes the threshold $\mathbb{D}_{\mathrm{thr}}$ of the threshold reward.
  `--steps` changes the training length.
  Run `python run_all.py --help` for all options.
* The notebook can also launch the training of the main study (set `RUN_TRAINING = True` in its section 3).
* To retrain from scratch, move or delete the stored runs first, or use a new `--tag` and set `TAG` in the notebook accordingly.
  Training is deterministic for a given seed on a fixed software and hardware stack.
  On other platforms, floating-point differences can change individual runs.

### 2. Evaluate the stored policies

The commands of step 1 also write the missing `evaluation.npz` files from the stored `policy.pt` files without retraining; with all policies stored, they only evaluate.

* Each run evaluates the deterministic policy on 64 episodes at each noise level $\eta\in\lbrace0,0.1,\ldots,0.5\rbrace$.
  Adding a noise level to `EVAL_NOISES` evaluates only the new level; the stored levels are reused, which reproduces a full evaluation because the episodes of one level depend only on the seeds and on the number of episodes.
  During the study, the evaluation of one run took less than 3 minutes on one core.
* The evaluation seeds are fixed.
  On the machine used for the paper, re-evaluating the stored policies at every noise level reproduced the stored arrays bit for bit.
* To re-evaluate a saved policy at every noise level, delete its `evaluation.npz` and run the command again.

### 3. Regenerate the figures and tables

```bash
jupyter nbconvert --to notebook --execute --inplace DMDP_SafetyGymnasium.ipynb
```

The notebook reads `results/main`, `results/sac`, and the pilot settings in `results/pilot/`.
It writes every figure (PDF and 200-dpi PNG) and LaTeX table to `figures/`, or to the directory given by the environment variable `DMDP_FIG_DIR`.
It takes about 10 minutes on the machine used for the paper.
`python build_notebook.py` rebuilds the notebook from its source, without outputs.
`python plot_halfcheetah.py` draws the robot of Appendix E (Figure 8) (`figures/halfcheetah`) from the stored policies `results/main/HalfCheetah/DMDP-Lag/seed0` and `results/sac/HalfCheetah/SAC/seed0`.

| Output in `figures/` | Paper |
|---|---|
| `halfcheetah` (`plot_halfcheetah.py`) | Appendix E (Figure 8) |
| `training_curves`, `stability_curves`, `stable_distributions`, `table_main.tex`, `table_reliability.tex`, `table_shift.tex` | Appendix F.2 |
| `table_floor.tex` (both learners) | Appendix F.2 |
| `sac_training_curves`, `sac_stability_curves`, `sac_stable_distributions`, `table_sac_main.tex`, `table_sac_reliability.tex`, `table_sac_shift.tex` | Section 5.1; `sac_training_curves` and `table_sac_shift.tex`: Appendix F.1 |
| `table_pilot.tex` | not in the paper (design pilot, notebook section 6.1) |

## Computable example (Appendix D)

None of the Lyapunov certificates of Theorems 2–5 is computed for the HalfCheetah implementation.
`dmdp_toy/` computes them for a scalar linear system with Gaussian noise, in which every hypothesis of these theorems can be checked:
$s_{k+1}=a_{k,1}s_k+(1-a_{k,1})a_{k,2}+\epsilon_k$ with $\epsilon_k\sim\mathcal{N}(0,0.5^2)$ and the action $\mathbf{a}_k=(a_{k,1},a_{k,2})\in[0.70,0.95]\times[0.5,1.5]$, common to all realizations.
The Gaussian family $\rho_\omega=\mathcal{N}(\mu,\varsigma^2)$, $\omega=(\mu,\varsigma)$, is closed under these dynamics, $f(\omega,\mathbf{a})=(a_1\mu+(1-a_1)a_2,\sqrt{a_1^2\varsigma^2+0.25})$, and $\Omega_{\mathsf{com}}=[-1,3]\times[0.700,1.601]$ is positively invariant under every action.
The target is $\rho_\star=\mathcal{N}(1,0.8^2)$, the dissimilarity $\mathbb{D}=W_2^2=\Vert\omega-\omega_\star\Vert^2$, the stage cost $\ell=\mathbb{D}$ with $\ell_0=0$, the storage $\lambda=0$, and $\gamma=1$.
The code computes $U_\star^{\mathsf{R}}$ by value iteration with a numerical error bound, runs the offline learner (Theorem 3), its plug-in deployment (Theorem 4), and a learner from an estimated Lyapunov decrease (Theorem 5), each 50 times (49 runs calibrate the residuals, run 49 is deployed), and writes the figure and the two tables of Appendix D.
The learners see only their data and the known quantities $\omega_\star$, $\mathbb{D}$, $\mathcal{A}$, $\Omega_{\mathsf{com}}$, and $\ell_0$; the true $f$ is used only to evaluate the certificates.
The example says nothing about the HalfCheetah task.

Run one command at a time; each takes 6 minutes or less on the machine used for the paper (one process, 2 threads, peak memory below 0.6 GB). `run_toy.py` sets the thread variables `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to 2 itself.
Every stage skips the run files that already exist, so an interrupted command can be repeated. `online.npz` records a checksum of the deployed offline policy and of $\tilde U$; `--stage online` and `--stage outputs` stop if these files changed afterwards (delete `online.npz` and rerun `--stage online`).

```bash
nice -n 10 python run_toy.py --stage true                              # about 0.5 min: U~ and its error indicators
nice -n 10 python run_toy.py --stage offline --levels 1 2 3            # about 5 min: Theorem 3, 150 runs
nice -n 10 python run_toy.py --stage offline --levels 4 --runs 0-24    # about 3.5 min
nice -n 10 python run_toy.py --stage offline --levels 4 --runs 25-49   # about 3.5 min
nice -n 10 python run_toy.py --stage online                            # about 0.5 min: Theorem 4
nice -n 10 python run_toy.py --stage modelfree --runs 0-24             # about 3.5 min: Theorem 5
nice -n 10 python run_toy.py --stage modelfree --runs 25-49            # about 3.5 min
nice -n 10 python run_toy.py --stage outputs                           # about 0.5 min: summary, figure, tables
```

`python run_toy.py --quick` runs every stage of a reduced setting in about 10 seconds and writes its data, figure, and tables to `results_toy_quick/`.
The figure and the tables go to `../Manuscript/figures/` if that folder exists and to `figures/` otherwise (`--figdir` changes it).

| Output | Content | Paper |
|---|---|---|
| `results_toy/true_value.npz` | $\tilde U\approx U_\star^{\mathsf{R}}$ as the ratio $\tilde U/\mathbb{D}$ on a graded grid, a medium reference grid, the error indicators and $\varepsilon_{\mathrm{num}}$, optimal trajectories, particle check | Lemma (lemma:toy_facts), Theorem 2 |
| `results_toy/offline/L<l>/run_<i>.npz` | per run: $\Delta_U$, $\Delta_f$, $\Delta$, $\hat c$, $e_f$, offset, limit distance, $\Delta_{\min}$, score; for run 49 also the data, the GP states, $\widehat U^{\mathsf{R}}_{\mathsf{off}}$, the policy, and the same maxima on an independent check set of 80,000 points | Theorem 3, Proposition 1 |
| `results_toy/online.npz` | $M_t$, $\Delta_{\mathsf{on}}(\delta)$, 200 plug-in runs of 40 steps, checksum of the deployed policy and $\tilde U$ | Theorem 4 |
| `results_toy/modelfree/run_<i>.npz` | per run and checkpoint $n$: $\Delta_{\mathsf{dec},n}$ (capped and raw), $e_n$, $\Delta_{\min,n}$, limit distance; for run 49 also $\Delta_{\omega,n}(0.05)$, the plug-in runs, the GP states, the policy, and the maxima on the independent check set | Theorem 5 |
| `results_toy/summary.json` | every number of the tables, the calibration, the checks, the wall-clock time of every stage, versions, and seeds (paths relative to this folder) | Appendix D |
| `toy_certificates.pdf`, `.png` | figure | Figure fig:toy_certificates |
| `table_toy_hypotheses.tex`, `table_toy_results.tex` | hypothesis table, certificate table | Tables tab:toy_hypotheses, tab:toy_results |

| Paper | Code | Meaning |
|---|---|---|
| $s_k$, $\mathbf{a}_k=(a_{k,1},a_{k,2})$, $\epsilon_k$ | `system.step_particles` | scalar state, action (persistence, set point), noise |
| $\omega=(\mu,\varsigma)$, $f$, $\omega_\star$, $\mathbf{a}_\star$ | `system.f`, `OMEGA_STAR`, `A_STAR` | parameter of $\mathcal{N}(\mu,\varsigma^2)$, exact parameter dynamics, target parameter $(1,0.8)$, steady-state action $(\sqrt{39/64},1)$ |
| $\Omega_{\mathsf{com}}$, $r$, $\mathbb{D}$ | `MU_LO`, `MU_HI`, `VS_LO`, `VS_HI`, `system.project`, `system.dist2` | operating set, $r=\Vert\omega-\omega_\star\Vert$, $\mathbb{D}=W_2^2=r^2$ |
| $\mathcal{R}$ | `system.deadbeat` | set from which $\rho_\star$ is reached in one step; there $U_\star^{\mathsf{R}}=\mathbb{D}$ |
| $\tilde U$, $\varepsilon_{\mathrm{num}}$ | `vi.ValueFunction`, `eps_num` | computed $U_\star^{\mathsf{R}}$ and its numerical error |
| $B$, $T$, $M$, $N=BTM$, $n_{\mathrm{tr}}=BT$ | `offline.LEVELS`, `T_STEPS` | datasets, steps, samples per step, samples, transitions |
| $\hat c$, $\bar\Delta$, $\bar r$, $\Delta_{\min}$ | `c_hat`, `Delta_bar`, `radius`, `Delta_min` | learned margin coefficient, calibrated residual, residual radius, smallest residual that holds |
| $M_t$, $p_t$ | `online.sample_size_schedule` | online sample size and its failure probability $\alpha_{\mathsf{on}}/((t+1)(t+2))$ |
| $\widehat W^{+}_n$, $\widehat{\mathcal{D}}_n$, $e_n$ | `modelfree.run_learner`, `modelfree.decrease_quantities`, `e_n` | learned next value, empirical decrease, raw violation of the empirical decrease |

## Notation

| Paper | Code | Meaning |
|---|---|---|
| $\mathbf{y}_t$ | observation of the benchmark (17 entries) | observation vector (the full state $\mathbf{s}_k$ of the paper also contains the horizontal position of the torso) |
| $\mathbf{o}_t=\mathbf{y}_t+\eta\boldsymbol{\xi}_t$ | `envs.SensorNoise`, `NOISE_OFFSET`, `NOISE_SD` | noisy reading; the entries of $\boldsymbol{\xi}_t$ are independent draws from $\tfrac12\mathcal{N}(-m,s^2)+\tfrac12\mathcal{N}(m,s^2)$ with $m=0.8$ and $s=0.6$, so $\boldsymbol{\xi}_t$ has mean $\mathbf{0}$ and covariance $\mathbf{I}$ (Section 4.5, Table 7) |
| $\eta$ | `nu` (variable name in the code), `train_noise_max` (training), `EVAL_NOISES` (evaluation), keys `nu<level>/...` of `evaluation.npz` | sensor-noise level |
| $v_t$ | `info["x_velocity"]` | true forward velocity averaged over step $t$ (displacement of the torso divided by the step duration), the first regulated feature (Appendix E) |
| $\theta_t$ | `info["pitch"]`, array `pitch` in `evaluation.npz`, columns `pitch`, `flipped` | true pitch angle of the torso in $(-\pi,\pi]$, the second regulated feature; the torso is flipped (upside down) if $|\theta_t|>\pi/2$ |
| $o^v_t$, $o^\theta_t$ | `obs[vel_index]`, `obs[pitch_index]` (`vel_index = 8`, `pitch_index = 1`) | noisy readings of the instantaneous forward velocity and of the pitch angle of the torso at the start of step $t$ (Appendix E) |
| $v_{\mathrm{lim}}$, $c_t=\mathbf{1}[v_t>v_{\mathrm{lim}}]$ | `v_limit`, `info["cost"]` | velocity threshold and safety cost |
| $\bar c$, $H$ | `COST_LIMIT` (`cost_limit`), `EPISODE_LENGTH` | cost limit of every -Lag method (10) and episode length (1000) |
| $r^{\mathrm{b}}_t=v_t-0.1\Vert\mathbf{a}_t\Vert^2$ | `info["benchmark_reward"]`, `cost_penalty` $=0$ | benchmark reward, the training reward of PPO, SAC, Distributional PPO and SAC, and Implicit DMDP (PPO and SAC) (with $-\lambda c_t$, of every -Lag baseline: PPO-Lag, SAC-Lag, Distributional PPO-Lag and SAC-Lag, and Implicit DMDP-Lag (PPO and SAC)), and the base of the rewards of DMDP and DMDP-Lag (Appendix E); only the design pilot subtracted the cost penalty $(2c_t)^2$ (`envs.CostPenalty`, `COST_PENALTY` $=2$) |
| $r^{\mathrm{eva}}_t$, $r_{\mathrm{pen}}$ | `analysis.evaluation_rewards`, `analysis.evaluation_penalty`, `analysis.evaluation_returns`, column `ret_eva` | evaluation reward, the same for every method: $r^{\mathrm{eva}}_t=r^{\mathrm{b}}_t+r_{t\mathbb{D}}$ if $v_t\le v_{\mathrm{lim}}$ ($c_t=0$) and $r^{\mathrm{eva}}_t=-r_{\mathrm{pen}}$ with $r_{\mathrm{pen}}=v_{\mathrm{lim}}$ if $v_t>v_{\mathrm{lim}}$ ($c_t=1$), where the threshold reward $r_{t\mathbb{D}}$ is computed from the recorded window dissimilarity for every method (`analysis.threshold_reward`), also for the baselines, which are trained without it; it contains neither $-\beta\mathbb{D}_t$ nor $-\lambda c_t$. The evaluation tables report its episodic return (column `ret_eva`); the episodic return of $r^{\mathrm{b}}_t$ is the column `ret_benchmark` of the notebook (`analysis.benchmark_returns`) |
| $R$, $R_{\mathrm{Thr}}$, $\mathbb{P}[R\ge x]$ | `analysis.episode_returns`, columns `ret` (`episodes.csv`) and `ep_return` (`training.csv`), `R_THR` (notebook), `metrics.exceedance_probability` | episodic return of the reward each method is trained on, without $-\beta\mathbb{D}_t$ and $-\lambda c_t$ ($\sum_t r^{\mathrm{b}}_t$ for the baselines, $\sum_t(r^{\mathrm{b}}_t+r_{t\mathbb{D}})$ for the DMDP methods), used by the training curves and the reliability analysis, return threshold (5000, an average reward of 5 per step), and the fraction of episodes whose return reaches $x$ (Section 5) |
| $\rho_k$ | `velocity[:, k]` in `evaluation.npz` | distribution of $v_k$ across evaluation episodes at step $k$ |
| $\rho_\star=\rho^v_\star\otimes\rho^\theta_\star$, $\omega_\star$ | `targets.Target`, `mu_star`, `sd_star`, `config.band_target`, `TARGET_BAND`, `pitch_mu_star`, `pitch_sd_star`, `PITCH_HALF_WIDTH` | target: $U[0.70\,v_{\mathrm{lim}},0.95\,v_{\mathrm{lim}}]$ for the velocity and $U[-0.2,0.2]$ rad for the pitch, and its parameter (means and standard deviations) (Appendix E) |
| $n_{\mathrm{w}}$ | `window` | window length (10) |
| $\hat\omega_t$ | `policy_input="hybrid"` in `envs.DistributionEstimator` | window means and standard deviations of the readings $o^v_t$ and $o^\theta_t$, used as policy input (Section 4.5) |
| $\bar\rho_t$ | `envs.DistributionEstimator` | empirical distributions of the true velocities and pitch angles in the window, used only for training (Appendix E) |
| $\phi(\hat\omega_t,\mathbf{o}_t;\theta)$, $\pi(\mathbf{o}_t)$ | `policy_input="hybrid"`, `policy_input="state"` | policy of the DMDP methods and of Implicit DMDP (PPO and SAC), with and without -Lag; state-feedback policy of the other baselines |
| quantiles of the return | `n_quantiles` (`N_QUANTILES` $=32$), `agent.quantile_huber_loss`, `train.compute_quantile_returns` | distributional reward critics of Distributional PPO and SAC and Implicit DMDP (PPO and SAC), with and without -Lag, at the levels $(2i-1)/64$; their mean is the value |
| $\mathbb{D}_t=\mathbb{D}(\bar\rho_t\Vert\rho_\star)$ | `targets.empirical_w2_squared`, `info["dissimilarity"]`, `info["dissimilarity_velocity"]`, `info["dissimilarity_pitch"]`, `velocity_weight`, `pitch_weight` | training dissimilarity: sum of the squared 2-Wasserstein distances of the velocity window and of the pitch window to their targets, exact for the ordered windows (Appendix E) |
| $\mathbb{D}_{\mathrm{thr}}$ | `info["d_threshold"]`, `D_THRESHOLD`, `d_threshold` | threshold of the window dissimilarity, 0.1 for every window length (Appendix E) |
| $r_{t\mathbb{D}}$, $d$ | `envs.ThresholdReward`, `d_bonus` (`D_BONUS` $=10$), `info["r_tD"]`, array `bonus` in `evaluation.npz` | threshold reward of DMDP and DMDP-Lag: $+d$ if $\mathbb{D}(\bar\rho_t\Vert\rho_\star)\le\mathbb{D}_{\mathrm{thr}}$, $0$ otherwise |
| $\beta$ | `beta`, `BETA` | weight of the dissimilarity term $-\beta\mathbb{D}_t$ of DMDP and DMDP-Lag (0.5; Appendix E) |
| $\mathbb{P}[\mathbb{D}\le\mathbb{D}_{\mathrm{thr}}]$ | column `within` | fraction of the evaluation steps $k\ge500$ whose window dissimilarity is at most $\mathbb{D}_{\mathrm{thr}}$ |
| $\lambda$ | `lagrange_multiplier` | Lagrange multiplier of the cost constraint of the -Lag methods (Section 5) |
| $M$ | `EVAL_EPISODES` | evaluation episodes per noise level (64) |
| $k_0$ | `k0` (`K0` in the notebook) | first step used to estimate $\hat\rho$ (500) |
| $\hat\rho$ | `metrics.estimate_rho_hat` | stable distribution: $v_k$ pooled over $k\ge k_0$ (Section 5) |
| $\mathbb{D}(\rho_k\Vert\hat\rho)$ | `metrics.stability_curve`, columns `D_early` ($k<200$) and `D_late` ($k\ge500$) | evaluation dissimilarity at every single step, 1-Wasserstein, cross-fitted (Section 5) |
| $\mathbb{D}(\hat\rho\Vert\rho_\star)$ | `metrics.distance_to_target`, column `D_target` | distance of the stable distribution to the target, 1-Wasserstein (not the squared 2-Wasserstein distance of training) |
| $\mathbb{D}(\hat\rho_\eta\Vert\hat\rho_0)$ | column `D_shift` | shift of the stable distribution caused by sensor noise (Appendices F.1 and F.2) |
| stationary reference value | `metrics.sampling_floor` | value of $\mathbb{D}(\rho_k\Vert\hat\rho)$ for $k\ge500$ for a surrogate whose distribution at a single step does not depend on $k$: the segment $k\ge500$ of every episode is shifted circularly in time by an independent random amount (Appendix F.2) |

## Findings

The numbers are in `figures/table_*.tex` and in the notebook; this section states the trends.
The notebook holds the per-seed values in the data frames `main.df` and `main.rel` (PPO study), `sac.df` and `sac.rel` (SAC study), `decomposition`, `per_seed`, `negative`, `orientation`, and `orientation_sac`, the design pilot in `pilot`, `pilot_baselines`, and `pilot_benchmark`, the per-run ratios to the stationary reference value in `tests` (PPO study) and `sac_tests` (SAC study), and the synchronized runs in `phase` and `sac_phase`.
The Lagrange multipliers are read from the column `lagrange_multiplier` of `training.csv`.
The evaluation tables report the episodic return of the evaluation reward $r^{\mathrm{eva}}_t$, which is the same for every method: it equals $r^{\mathrm{b}}_t+r_{t\mathbb{D}}$ at a safe step ($v_t\le v_{\mathrm{lim}}$), with the benchmark reward $r^{\mathrm{b}}_t=v_t-0.1\Vert\mathbf{a}_t\Vert^2$ and the threshold reward $r_{t\mathbb{D}}$ computed from the window dissimilarity for every method, and $-v_{\mathrm{lim}}$ at an unsafe step.
The benchmark reward grows with the velocity and thus rewards violating the limit, whereas the evaluation reward credits only safe steps and, through $r_{t\mathbb{D}}$, the closeness of the window distribution to the target.
In the tables, bold marks the best value among all methods.
The training curves and the reliability analysis use the return $R$ of the reward a method is trained on, without $-\beta\mathbb{D}_t$ and $-\lambda c_t$: $r^{\mathrm{b}}_t$ for the baselines and $r^{\mathrm{b}}_t+r_{t\mathbb{D}}$ for the DMDP methods, so the returns $R$ of the DMDP methods include the threshold reward (at most $dH=10^4$), which the baselines do not receive.
The distances $\mathbb{D}(\rho_k\Vert\hat\rho)$, $\mathbb{D}(\hat\rho\Vert\rho_\star)$, and $\mathbb{D}(\hat\rho_\eta\Vert\hat\rho_0)$ are 1-Wasserstein distances for the velocity, whereas the window dissimilarity and its threshold $\mathbb{D}_{\mathrm{thr}}$ are sums of squared 2-Wasserstein distances, so the two are not directly comparable.

* **Of all methods, the stable velocity distribution of DMDP-Lag (PPO/SAC) is the closest to the target without noise, and that of DMDP-Lag (SAC) remains the closest at $\eta=0.2$.**
  With PPO, DMDP-Lag (PPO) is clearly closer than every other method without noise (`figures/table_main.tex`). At $\eta=0.2$, Distributional PPO-Lag is the closest, with PPO-Lag and DMDP (PPO) within its seed variation and DMDP-Lag (PPO) clearly farther, within the seed variation of Implicit DMDP-Lag (PPO). DMDP (PPO) is clearly closer than PPO, Distributional PPO, and Implicit DMDP (PPO), the unconstrained baselines, at both noise levels. The constrained baselines are clearly closer than their unconstrained counterparts at both noise levels, whereas DMDP (PPO), even without the cost constraint, is within the seed variation of the constrained baselines at both noise levels; DMDP-Lag (PPO) is clearly closer than DMDP (PPO) only without noise.
  With SAC, DMDP-Lag (SAC) is the closest of all methods at both noise levels, clearly closer than every other method without noise, followed by Implicit DMDP-Lag (SAC), which is within the seed variation of DMDP-Lag (SAC) at $\eta=0.2$; both are clearly closer than SAC-Lag and Distributional SAC-Lag (`figures/table_sac_main.tex`). DMDP (SAC) is closer than SAC, Distributional SAC, and Implicit DMDP (SAC) at both noise levels, clearly so at $\eta=0.2$ and within the seed variation of SAC and Distributional SAC without noise; two of its runs stay in the velocity band, whereas the other three never do. At both noise levels, SAC-Lag is clearly closer than SAC and Distributional SAC-Lag, whereas Distributional SAC and Distributional SAC-Lag are within the seed variation of SAC and of each other, and Implicit DMDP (SAC) is clearly the farthest from the target of all methods with SAC.
  Learning the distribution of the return does not bring the stable distribution closer to the target: with and without the cost constraint, the Distributional baselines of both learners are never clearly closer than the corresponding baselines whose critics learn the expected return (PPO, PPO-Lag, SAC, and SAC-Lag), and Distributional PPO at $\eta=0.2$ and Distributional SAC-Lag at both noise levels are clearly farther than PPO and SAC-Lag, respectively.
  With PPO, the window dissimilarity of DMDP-Lag (PPO) is at most $\mathbb{D}_{\mathrm{thr}}$ on almost all evaluation steps without noise, the largest fraction of all methods, with DMDP (PPO) and PPO-Lag within its seed variation; at $\eta=0.2$, the fractions of the DMDP methods and the constrained baselines are all within the seed variation of one another, whereas those of the unconstrained baselines stay near zero at both noise levels; the run of seed 2 of DMDP (PPO) never stays in the velocity band. With SAC, DMDP-Lag (SAC) has clearly the largest fraction of all methods at both noise levels, covering nearly every step without noise.
  Without noise, every run of DMDP (PPO) and of Distributional PPO-Lag is upright, whereas three runs of Implicit DMDP (PPO), two of PPO, and one each of Distributional PPO, PPO-Lag, Implicit DMDP-Lag (PPO), and DMDP-Lag (PPO) are tilted; at $\eta=0.2$, no run of the DMDP methods or of the constrained baselines is flipped, whereas the unconstrained baselines have one flipped run each, and most runs of every baseline except PPO-Lag are tilted (`orientation`; column `pitch` of `main.df` for tilted and upright).
  Without noise, the unconstrained baselines run mostly above $v_{\mathrm{lim}}$, the stable distributions of the constrained baselines peak at the upper edge of the velocity band, just below $v_{\mathrm{lim}}$, and those of the DMDP methods lie mostly within the band and spread over it, with a small tail above $v_{\mathrm{lim}}$ for DMDP (PPO) (`figures/stable_distributions.png`).
* **With PPO, DMDP-Lag (PPO) has the highest evaluation return of all methods without noise, and the two DMDP methods reach $R_{\mathrm{Thr}}$ clearly more often than the baselines.**
  The evaluation return $r^{\mathrm{eva}}$ is the column `ret_eva` (`figures/table_main.tex`). Without noise, DMDP-Lag (PPO) has the highest, within the seed variation of PPO-Lag and clearly above the other six methods; at $\eta=0.2$, Distributional PPO-Lag has the highest, and the DMDP methods and the other two constrained baselines are within its seed variation. The evaluation returns of the three unconstrained baselines are negative at both noise levels, clearly below that of DMDP (PPO), because they spend many steps above $v_{\mathrm{lim}}$ (their costs are 309 to 864 of 1000 steps).
  The reliability is $\mathbb{P}[R\ge R_{\mathrm{Thr}}]$ with $R_{\mathrm{Thr}}=5000$ (an average reward of 5 per step) at the end of training and in the evaluation (`figures/table_reliability.tex`).
  In all three sets, DMDP-Lag (PPO) reaches $R_{\mathrm{Thr}}$ the most often of all methods, within the seed variation of DMDP (PPO), and both reach it clearly more often than the baselines. Without noise, DMDP-Lag (PPO) reaches it in almost all evaluation episodes and DMDP (PPO) in the large majority of the episodes of four runs, whereas its run of seed 2 never reaches it in any of the three sets. Of the baselines, only Implicit DMDP (PPO) reaches $R_{\mathrm{Thr}}$, in a few evaluation episodes of one run without noise.
  Without noise, Distributional PPO-Lag has the narrowest 10–90% range of the return of all methods, and at the end of training and at $\eta=0.2$, PPO-Lag has the narrowest; a DMDP method has the widest in every set, as its return $R$ includes the threshold reward. In all three sets, the range of Distributional PPO is wider than that of PPO, and that of Implicit DMDP (PPO) is narrower than that of Distributional PPO, much narrower without noise, whereas that of Implicit DMDP-Lag (PPO) is wider than that of Distributional PPO-Lag.
* **With SAC, DMDP-Lag (SAC) has clearly the highest evaluation return of all methods at both noise levels and reaches $R_{\mathrm{Thr}}$ the most often, whereas the six baselines never reach $R_{\mathrm{Thr}}$.**
  At both noise levels, the runs of seeds 0 and 4 of SAC and of SAC-Lag are flipped and their other runs tilted, except the upright run of seed 2 of SAC-Lag. The Distributional methods have the most flipped runs: four of Distributional SAC-Lag at both noise levels, and three of Distributional SAC without noise and four at $\eta=0.2$; their other runs are tilted, except the upright run of seed 4 of Distributional SAC without noise. In contrast, every run of Implicit DMDP (SAC) and Implicit DMDP-Lag (SAC), which differ from Distributional SAC and Distributional SAC-Lag, respectively, only in the policy input, is upright at both noise levels; every run of the DMDP methods is upright without noise, and none is flipped at $\eta=0.2$ (`orientation_sac`; column `pitch` of `sac.df` for tilted and upright).
  Without noise, the stable distributions of SAC, SAC-Lag, and the Distributional methods are wide, with mass at low velocities from the flipped runs; that of Implicit DMDP (SAC) is also wide and lies mostly above $v_{\mathrm{lim}}$, that of Implicit DMDP-Lag (SAC) peaks sharply at the upper edge of the velocity band, just below $v_{\mathrm{lim}}$, and that of DMDP-Lag (SAC) is concentrated in the band; that of DMDP (SAC) combines a sharp peak in the band, from its two runs that stay in it, with a wide part above $v_{\mathrm{lim}}$ from its other three runs (`figures/sac_stable_distributions.png`).
  The evaluation return $r^{\mathrm{eva}}$ is the column `ret_eva` (`figures/table_sac_main.tex`). DMDP-Lag (SAC) has clearly the highest of all methods at both noise levels, followed by Implicit DMDP-Lag (SAC), which is clearly above Distributional SAC-Lag, the method from which it differs only in the policy input. Implicit DMDP (SAC) has clearly the lowest evaluation return, which is negative at both noise levels, because it spends most steps above $v_{\mathrm{lim}}$ (its cost is 726 to 845 of 1000 steps). The evaluation return of DMDP (SAC) varies much between its runs: two runs stay in the velocity band, and three never do.
  At the end of training and in the evaluation, DMDP-Lag (SAC) reaches $R_{\mathrm{Thr}}$ the most often of all methods, and the six baselines never reach it (`figures/table_sac_reliability.tex`). DMDP-Lag (SAC) reaches it in every episode of the evaluation without noise and in every episode of four of its five runs in the other two sets, clearly more often than DMDP (SAC) in all three sets; two runs of DMDP (SAC) reach it in every episode, and its three runs that never stay in the velocity band never reach it.
  Implicit DMDP-Lag (SAC) has the narrowest 10–90% range of the return of all methods without noise, closely followed by DMDP-Lag (SAC), and Distributional SAC-Lag has the narrowest at the end of training and at $\eta=0.2$. The whole range of each Implicit DMDP method lies above that of its Distributional counterpart in every set. DMDP (SAC) has the widest range of all methods in all three sets, since it spans its two runs that receive the threshold reward and its three runs that receive almost none, and DMDP-Lag (SAC) has the second widest at the end of training and at $\eta=0.2$, mainly because of the lower returns of its run of seed 4 there.
* **Distribution feedback without a target.** Implicit DMDP (PPO), Implicit DMDP-Lag (PPO), Implicit DMDP (SAC), and Implicit DMDP-Lag (SAC) differ from the corresponding Distributional methods only in that their policies also receive $\hat\omega_t$; like them, they have no target.
  With SAC, this feedback alone keeps every run upright at both noise levels, with and without the cost constraint, and with the cost constraint, it clearly raises the evaluation and benchmark returns (without the cost constraint, it raises the benchmark return but lowers the evaluation return, because the Implicit DMDP (SAC) runs mostly above $v_{\mathrm{lim}}$). With PPO, it makes the benchmark return of Implicit DMDP (PPO) vary much less between runs than that of Distributional PPO at both noise levels, and at $\eta=0.2$, Implicit DMDP (PPO) and Implicit DMDP-Lag (PPO) have clearly lower costs than Distributional PPO and Distributional PPO-Lag, respectively.
  In distance to the target, the feedback helps only together with SAC and the cost constraint: at both noise levels, Implicit DMDP-Lag (SAC) is clearly closer than Distributional SAC-Lag, whereas Implicit DMDP (PPO) is within the seed variation of Distributional PPO and clearly farther than PPO, Implicit DMDP-Lag (PPO) is clearly farther than Distributional PPO-Lag, and Implicit DMDP (SAC) is clearly the farthest of all methods with SAC.
  Knowing $\rho_\star$ takes the DMDP methods, whose policies receive the same $\hat\omega_t$, further: at both noise levels, DMDP (PPO/SAC) is clearly closer to the target than Implicit DMDP (PPO/SAC) with the same learner and has a clearly lower cost, and without noise, DMDP-Lag (PPO/SAC) is clearly closer than Implicit DMDP-Lag (PPO/SAC) with the same learner.
* **Training curves** (`figures/training_curves.png`, `figures/sac_training_curves.png`). The third panel summarizes each training episode by the median of $\mathbb{D}_t$ over its steps and shows the median (line) and the interquartile range (band) of these values over the recent training episodes of all five seeds; the line lies below $\mathbb{D}_{\mathrm{thr}}$ when most recent episodes have $\mathbb{D}_t\le\mathbb{D}_{\mathrm{thr}}$ (for the DMDP methods: receive $r_{t\mathbb{D}}$) on most of their steps (fractions at the end of training: `end_of_training_threshold` in the notebook).
  With PPO, the returns of the DMDP methods rise above $R_{\mathrm{Thr}}$ within the first $10^6$ training steps and remain there, whereas those of the baselines stay far below it. In the second half of training, the costs of the unconstrained baselines lie far above $\bar c$, those of the constrained baselines and DMDP-Lag (PPO) stay near $\bar c$, and that of DMDP (PPO) lies between these two groups, well above $\bar c$. In the same period, the median episodes of the DMDP methods lie lowest, below $\mathbb{D}_{\mathrm{thr}}$, closely followed by that of Distributional PPO-Lag; that of Implicit DMDP-Lag (PPO) stays near $\mathbb{D}_{\mathrm{thr}}$, and that of PPO-Lag lies above it and approaches it only toward the end of training, whereas those of the unconstrained baselines stay more than an order of magnitude above it.
  With SAC, the mean returns of the DMDP methods rise above $R_{\mathrm{Thr}}$ within about $2\times10^5$ training steps, whereas those of the baselines stay below it; among the baselines, Implicit DMDP (SAC) has the highest return after about $10^5$ steps, and Implicit DMDP-Lag (SAC) the second highest in the second half of training. From about $2\times10^5$ steps on, the mean return of DMDP-Lag (SAC) is the highest and keeps rising over most of the remaining training, whereas that of DMDP (SAC) levels off. In the second half of training, the costs of the four -Lag methods (SAC-Lag, Distributional SAC-Lag, Implicit DMDP-Lag (SAC), and DMDP-Lag (SAC)) stay near or below $\bar c$, whereas those of the methods without a cost constraint stay far above it, and that of Implicit DMDP (SAC) is the highest. From about $2\times10^5$ steps on, the median episode of DMDP-Lag (SAC) lies below $\mathbb{D}_{\mathrm{thr}}$, as does that of Implicit DMDP-Lag (SAC) in the second half of training, whereas that of DMDP (SAC) stays above it, since three of its five runs never stay in the velocity band; those of SAC, SAC-Lag, and Implicit DMDP (SAC) stay about an order of magnitude above it, and those of the Distributional methods, seven of whose ten runs are flipped in the evaluation without noise, stay higher still.
* **Cost.** At both noise levels, the costs of the methods without a cost constraint lie well above $\bar c$. Of all methods, DMDP-Lag (SAC) has the lowest cost with SAC without noise, within the seed variation of that of Implicit DMDP-Lag (SAC), and with PPO, PPO-Lag has the lowest without noise and Implicit DMDP-Lag (PPO) at $\eta=0.2$, where DMDP-Lag (PPO) exceeds $\bar c$, clearly above the three constrained baselines.
  With PPO, the costs of PPO, Distributional PPO, and Implicit DMDP (PPO), which have neither a cost term nor a target, are clearly higher than those of the other methods at both noise levels and within the seed variation of one another, except that, at $\eta=0.2$, the cost of Implicit DMDP (PPO) is clearly lower than that of Distributional PPO; DMDP (PPO) thus has a clearly lower cost than these three baselines at both noise levels, although its cost lies well above $\bar c$. Without noise, the costs of the -Lag methods are below $\bar c$ and within the seed variation of one another, and PPO-Lag has the lowest mean. At $\eta=0.2$, Implicit DMDP-Lag (PPO) has the lowest mean cost of all methods, below $\bar c$ and within the seed variation of that of PPO-Lag, which is at $\bar c$; its cost is clearly lower than those of Distributional PPO-Lag and DMDP-Lag (PPO), which exceed $\bar c$ (`figures/table_main.tex`).
  With SAC, each of the four -Lag methods has a clearly lower cost than each of the methods without a cost constraint at both noise levels. Without noise, DMDP-Lag (SAC) has the lowest cost of all methods, far below $\bar c$, and Implicit DMDP-Lag (SAC) the second lowest, within the seed variation of that of DMDP-Lag (SAC); both are clearly lower than those of SAC-Lag and Distributional SAC-Lag, and Distributional SAC-Lag is the only -Lag method whose cost lies above $\bar c$. At $\eta=0.2$, the costs of all four -Lag methods are below $\bar c$ and within the seed variation of one another. Implicit DMDP (SAC) has clearly the highest cost of all methods, and the costs of the four methods without a cost constraint lie far above $\bar c$, the lowest of them that of Distributional SAC, within the seed variation of that of SAC. The cost of DMDP (SAC) is clearly higher than that of SAC without noise and within its seed variation at $\eta=0.2$; it comes almost entirely from the three runs that never stay in the velocity band, whereas, without noise, the other two runs have costs far below $\bar c$ (`figures/table_sac_main.tex`).
  DMDP (PPO/SAC), which has the velocity margin of the target but no cost constraint, has a mean cost well above $\bar c$ at both noise levels.
* **Stability.** Every method reaches a stable distribution, but at low noise levels some runs are synchronized, so their $\rho_k$ oscillates; with SAC, DMDP (SAC) has the most synchronized runs, followed by Implicit DMDP (SAC) and DMDP-Lag (SAC) (`figures/table_floor.tex`).
  With PPO, the ratio to the stationary reference value is close to one for every method from $\eta=0.4$ on and at most slightly above one at $\eta=0.2$ and $0.3$. Without noise, it is above one for every method, mainly because 10 of the 40 runs are synchronized (two each of Distributional PPO-Lag, Implicit DMDP (PPO), and Implicit DMDP-Lag (PPO) and one each of DMDP (PPO), DMDP-Lag (PPO), PPO, and PPO-Lag) and one run of Distributional PPO is nearly synchronized (`phase`); the episodes of a synchronized run differ only in their initial states. One run each of PPO-Lag and Implicit DMDP (PPO) stays synchronized at $\eta=0.1$, where these two methods have the highest ratios.
  Implicit DMDP-Lag (PPO) and Distributional PPO-Lag have the smallest stability distances for $k\ge500$ of all methods at both noise levels, within the seed variation of each other, and DMDP (PPO) has a clearly smaller one than the three unconstrained baselines, which have clearly larger stability distances than all other methods. The same holds for $k<200$, where Implicit DMDP (PPO) has the largest mean at both noise levels, and Distributional PPO-Lag without noise and PPO-Lag at $\eta=0.2$ have the smallest of all methods. Without noise, the stability distance of every method drops by about an order of magnitude within the first hundred steps; that of Distributional PPO then keeps decreasing over several hundred steps, whereas those of the other methods change much less. At $\eta=0.3$ and $0.5$, it decreases over several hundred steps for most methods (figure below).
  With SAC, the ratios of DMDP (SAC) and Implicit DMDP (SAC) are far above one for $\eta\le0.1$ and still well above one at $\eta=0.2$, whereas those of the other methods are close to one from $\eta=0.2$ on; from $\eta=0.4$ on, the ratios of all methods are close to one. Without noise, 21 of the 40 runs with SAC are synchronized: all five of DMDP (SAC), four each of Implicit DMDP (SAC) and DMDP-Lag (SAC), two each of SAC-Lag, Distributional SAC-Lag, and Implicit DMDP-Lag (SAC), and one each of SAC and Distributional SAC; 14 of them stay synchronized at $\eta=0.1$, among them four of DMDP (SAC) and three each of Implicit DMDP (SAC) and DMDP-Lag (SAC) (`sac_phase`). Without noise, the oscillation of $\rho_k$ in these runs accounts for most of the stability distance of DMDP (SAC) and Implicit DMDP (SAC) for $k\ge500$.
  DMDP-Lag (SAC) has the smallest stability distance for $k\ge500$ of all methods at both noise levels, which reflects its narrow stable distribution, and Implicit DMDP-Lag (SAC) the second smallest, within the seed variation of that of DMDP-Lag (SAC); both are clearly smaller than those of the other six methods. For $k<200$, these two methods have the two smallest stability distances of all methods at both noise levels. DMDP (SAC) and Implicit DMDP (SAC) have the largest for $k\ge500$, within the seed variation of each other. At both noise levels, each -Lag method except Distributional SAC-Lag has a clearly smaller stability distance for $k\ge500$ than its counterpart without a cost constraint. The stability distance of every method drops steeply within about a hundred steps and then changes much more slowly; after this drop, the curves of DMDP-Lag (SAC) and Implicit DMDP-Lag (SAC) lie lowest up to $\eta=0.3$, and they nearly coincide except at $\eta=0.1$, where that of DMDP-Lag (SAC) lies below (`figures/sac_stability_curves.png`).
* **Sensor noise.** With PPO, sensor noise shifts the stable distributions of all methods, the more so the larger $\eta$ (`figures/table_shift.tex`). PPO-Lag has the smallest mean shift of all methods at every noise level, and DMDP (PPO), without the cost constraint, shifts within the seed variation of PPO-Lag at every noise level and clearly less than the unconstrained baselines; the constrained baselines and DMDP-Lag (PPO) also shift clearly less than the unconstrained baselines. Implicit DMDP (PPO) shifts within the seed variation of Distributional PPO at $\eta=0.1$ and $0.5$ and clearly more at the noise levels in between, whereas Implicit DMDP-Lag (PPO) shifts within the seed variation of Distributional PPO-Lag at every noise level. PPO-Lag shifts within the seed variation of Distributional PPO-Lag up to $\eta=0.3$ and clearly less than DMDP-Lag (PPO) from $\eta=0.2$ on. The two DMDP methods shift within the seed variation of each other at every noise level, and DMDP-Lag (PPO) within that of Implicit DMDP-Lag (PPO); Distributional PPO-Lag shifts clearly less than DMDP-Lag (PPO) at $\eta=0.3$ and clearly more than both DMDP methods at $\eta=0.5$.
  With SAC, Distributional SAC-Lag, four of whose runs are flipped at every noise level, has the smallest shift of all methods at every noise level, shared with DMDP-Lag (SAC) at $\eta=0.1$ (`figures/table_sac_shift.tex`). DMDP-Lag (SAC) shifts within the seed variation of Distributional SAC-Lag up to $\eta=0.4$, and SAC-Lag does so up to $\eta=0.3$, whereas Implicit DMDP-Lag (SAC) shifts clearly more than SAC-Lag and Distributional SAC-Lag at every noise level. SAC and Distributional SAC shift within the seed variation of each other at every noise level; DMDP (SAC) shifts clearly more than both at $\eta=0.2$ and $0.3$ and within the seed variation of SAC at the other noise levels. Implicit DMDP (SAC) shifts clearly more than every other method at every noise level, including the DMDP methods, whose policies receive the same $\hat\omega_t$; its stable distribution moves to lower velocities as $\eta$ grows.
* **Limitation and future work (paper Section 6).**
  The Lyapunov certificates of the theory are not computed for the learned HalfCheetah policies. In the example of Appendix D (`dmdp_toy`), computing them requires the true model, and the certified residual sets are much larger than the sets in which the learned closed loops settle. Certifying learned policies from data alone, by learning a Lyapunov function together with the policy and bounding the error terms of the certificates from data, is left for future work.
  Earlier designs (the penalty $-\beta\mathbb{D}$ alone, several threshold-reward variants with the target $U[0,3.29]$) are not reported in the paper and archived outside this repository.

![Convergence of the velocity distribution to the stable distribution](figures/stability_curves.png)

*Convergence of $\rho_k$ to the stable distribution $\hat\rho$ with PPO (Appendix F.2).*

The figure shows $\mathbb{D}(\rho_k\Vert\hat\rho)$ for $\eta\in\lbrace0,0.1,0.3,0.5\rbrace$, computed at every step and, for display, averaged over 10 consecutive steps for each seed.
Lines are means over the seeds (one color per family: PPO blue, Distributional PPO green, Implicit DMDP (PPO) violet, DMDP (PPO) orange; dashed for the -Lag variants), bands are standard errors, and the dotted line marks $k_0=500$.

## Citation

The citation will be added after the review process.

## License

License: not yet specified.
