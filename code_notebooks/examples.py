"""Example cells that call the code of the notebooks (Section 10) and the cells that read the stored results (Section 11).

The texts use the placeholders @L@ (learner in lower case, ``ppo`` or ``sac``) and @U@ (upper case) and the code cells use the
names ``LEARNER`` and ``ALL_METHODS``, which the first example cell defines.
"""

from __future__ import annotations

from .builder import Notebook
from .shared import METHOD_ORDER


def _fill(text: str, learner: str) -> str:
    return text.replace("@L@", learner).replace("@U@", learner.upper())


def _md(nb: Notebook, learner: str, text: str) -> None:
    nb.md(_fill(text, learner))


def _code(nb: Notebook, learner: str, text: str) -> None:
    nb.code(_fill(text, learner))


# ---------------------------------------------------------------------------------------------------------------------
# 10 Examples and checks
# ---------------------------------------------------------------------------------------------------------------------
def start(nb: Notebook, learner: str) -> None:
    _md(nb, learner, r"""
## 10 Examples and checks

The cells of this section call the code above with small settings: a few steps of the environment, a check of some functions, and one tiny training run with its evaluation.
The policy of the tiny run has seen a few thousand steps only, so its numbers show that the pipeline works and not how well a method performs; Section 11 shows the stored results of the full runs.

**Methods.** `ALL_METHODS` lists the methods of the @U@ learner in the order of the tables of the paper, each method followed by its -Lag variant, and the table shows how `METHODS` defines them.
""")
    order = METHOD_ORDER[learner]
    quoted = "[" + ", ".join(f'"{m}"' for m in order[:4]) + ",\n               " + ", ".join(f'"{m}"' for m in order[4:]) + "]"
    nb.code(f"""
LEARNER = "{learner}"
ALL_METHODS = {quoted}

methods = pd.DataFrame({{method_label(m, LEARNER): METHODS[m] for m in ALL_METHODS}}).T
methods = methods[["policy_input", "lagrangian", "beta", "d_bonus", "n_quantiles"]]
methods["n_quantiles"] = methods["n_quantiles"].fillna(0).astype(int)
methods
""")
    fields = {
        "ppo": ["method", "learner", "total_steps", "num_envs", "num_steps", "update_epochs", "num_minibatches", "learning_rate",
                "gamma", "gae_lambda", "clip_coef", "hidden", "init_log_std", "policy_input", "window", "beta", "d_bonus",
                "d_threshold", "cost_limit", "lagrangian_lr"],
        "sac": ["method", "learner", "total_steps", "num_envs", "sac_hidden", "sac_learning_rate", "sac_batch_size", "sac_buffer_size",
                "sac_tau", "sac_learning_starts", "sac_lambda_interval", "gamma", "policy_input", "window", "beta", "d_bonus",
                "d_threshold", "cost_limit", "lagrangian_lr"],
    }[learner]
    _md(nb, learner, r"""
**Configuration.** `make_config` builds the configuration of DMDP-Lag (@U@), the method with every component: the window estimator, the threshold reward, the dissimilarity term, and the cost constraint.
The table shows some of its fields.
""")
    wrapped, line = [], ""
    for name in fields:
        if len(line) + len(name) + 4 > 105:
            wrapped.append(line.rstrip())
            line = ""
        line += f'"{name}", '
    wrapped.append(line.rstrip().rstrip(","))
    shown = "[" + "\n         ".join(wrapped) + "]"
    nb.code(f"""
config = make_config("HalfCheetah", "DMDP-Lag", seed=0, learner=LEARNER)
shown = {shown}
pd.Series({{name: getattr(config, name) for name in shown}}, name="DMDP-Lag").to_frame()
""")


def targets_check(nb: Notebook, learner: str) -> None:
    _md(nb, learner, r"""
**Target and window dissimilarity.** The velocity target is $U[0.70\,v_{\mathsf{lim}},0.95\,v_{\mathsf{lim}}]$ and the pitch target is $U[-0.2,0.2]$.
The first check compares `empirical_w2_squared` with the integral $\int_0^1(F^{-1}_{\text{window}}(u)-F^{-1}_\star(u))^2\,\mathrm{d}u$ computed numerically, for a window of 10 samples from the target.
""")
    nb.code(r"""
velocity_target = Target("uniform", *band_target(V_LIMIT))
pitch_target = Target("uniform", 0.0, PITCH_HALF_WIDTH / 3 ** 0.5)
print("velocity target", velocity_target.label(), "   pitch target", pitch_target.label())

sample_window = velocity_target.sample(np.random.default_rng(0), 10)
u = (np.arange(200_000) + 0.5) / 200_000                                  # midpoints of a fine grid on (0, 1)
window_quantile = np.sort(sample_window)[(u * sample_window.size).astype(int)]     # quantile function of the empirical distribution
print("W2^2 from the sorted samples :", round(empirical_w2_squared(sample_window, velocity_target), 6))
print("W2^2 by numerical integration:", round(float(np.mean((window_quantile - velocity_target.quantile(u)) ** 2)), 6))
""")
    _md(nb, learner, r"""
A window at rest is far from the target, whereas a window drawn from the target is almost always within the threshold $\mathbb{D}_{\mathsf{thr}}=0.1$ (`window_within_probability`; it is estimated for the velocity target alone).
""")
    nb.code(r"""
print("window at rest, velocity term   :", round(empirical_w2_squared(np.zeros(10), velocity_target), 3))
print("window at rest, pitch term      :", round(empirical_w2_squared(np.zeros(10), pitch_target), 4))
print("P[D <= D_thr] for a window drawn from the velocity target:", round(window_within_probability(velocity_target, 10, D_THRESHOLD), 4))
""")


def env_check(nb: Notebook, learner: str) -> None:
    _md(nb, learner, r"""
**Environment stack.** The cell steps the wrapped environment of DMDP-Lag with uniformly random actions at the noise level $\eta=0.2$.
The table shows the quantities that the wrappers put into `info`: the true velocity $v_t$, the cost $c_t$, the pitch $\theta_t$, the two terms and the sum of the window dissimilarity, the threshold reward $r_{t\mathbb{D}}$, and the reward that the learner receives before the term $-\beta\,\mathbb{D}_t$ is subtracted.
The windows are short at the first steps of an episode, and a robot that hardly moves is far from the velocity target, so $\mathbb{D}_t>\mathbb{D}_{\mathsf{thr}}$ and $r_{t\mathbb{D}}=0$.
""")
    nb.code(r"""
env = make_env(nu=0.2, seed=0, **env_settings(config))
obs, _ = env.reset(seed=0)
print(f"policy input: {obs.shape[0]} entries (4 window statistics and the 17 entries of the reading)")
steps = []
for t in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    steps.append(dict(t=t, v=info["x_velocity"], c=info["cost"], pitch=info["pitch"], D_velocity=info["dissimilarity_velocity"],
                      D_pitch=info["dissimilarity_pitch"], D=info["dissimilarity"], r_tD=info["r_tD"], reward=reward))
env.close()
pd.DataFrame(steps).set_index("t").iloc[[0, 1, 2, 9, 49, 199]].round(3)
""")


def quantile_check(nb: Notebook, learner: str) -> None:
    _md(nb, learner, r"""
**Distributional critic.** The cell fits the 8 outputs of a distributional critic to samples of a Gaussian with standard deviation 5 by minimizing `quantile_huber_loss`; the outputs approximate the quantiles of the Gaussian at the levels `quantile_levels`.
""")
    nb.code(r"""
torch.manual_seed(0)
outputs = torch.zeros(1, 8, requires_grad=True)
adam = torch.optim.Adam([outputs], lr=0.05)
for _ in range(1500):
    samples = 5.0 * torch.randn(128, 64)
    loss = quantile_huber_loss(outputs.expand(128, 8), samples)
    adam.zero_grad()
    loss.backward()
    adam.step()
levels = quantile_levels(8).numpy()
pd.DataFrame({"level": levels, "learned": outputs.detach().numpy()[0], "quantile of N(0, 5^2)": 5.0 * norm.ppf(levels)}).round(2)
""")


def multiplier_check(nb: Notebook, learner: str) -> None:
    update = {"ppo": "at every policy update", "sac": "every `sac_lambda_interval` environment steps"}[learner]
    _md(nb, learner, r"""
**Lagrange multiplier.** The multiplier takes one Adam step on $-\lambda(\hat J_c-\bar c)$ """ + update + r""" and is projected onto $\lambda\geq0$.
The cell feeds `_update_lagrange_multiplier` with a cost of 40 against the limit $\bar c=10$ for 60 updates and then with a cost of 0: $\lambda$ rises while the constraint is violated, falls when it holds, and stays at 0 at the projection.
""")
    nb.code(r"""
import matplotlib.pyplot as plt

multiplier = torch.zeros(1, requires_grad=True)
adam = torch.optim.Adam([multiplier], lr=TrainConfig.lagrangian_lr)
trace = []
for k in range(300):
    episodic_cost = 40.0 if k < 60 else 0.0
    _update_lagrange_multiplier(multiplier, adam, episodic_cost, COST_LIMIT)
    trace.append(float(multiplier))

fig, ax = plt.subplots(figsize=(5.5, 2.4))
ax.plot(trace)
ax.axvspan(0, 60, alpha=0.15, label=r"cost $\hat J_c=40>\bar c$")
ax.axvspan(60, 300, alpha=0.15, color="tab:orange", label=r"cost $\hat J_c=0<\bar c$")
ax.set_xlabel("update of the multiplier")
ax.set_ylabel(r"$\lambda$")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
plt.show()
""")


# ---------------------------------------------------------------------------------------------------------------------
# tiny training run
# ---------------------------------------------------------------------------------------------------------------------
_RUN_SETTINGS = {
    "ppo": ("total_steps=4000, num_envs=2, num_steps=250,\n                   update_epochs=4, num_minibatches=5", "train",
            "The settings give 8 updates of 2 environments $\\times$ 250 steps; the first episodes end at the 4th update."),
    "sac": ("total_steps=4000, num_envs=2, sac_hidden=64, sac_batch_size=64,\n                   sac_learning_starts=500, sac_lambda_interval=500",
            "train_sac",
            "The settings give 4000 environment steps in 2 environments, with uniformly random actions for the first 500 steps and one gradient step per environment step afterwards; the first episodes end after 2000 environment steps."),
}


def tiny_run(nb: Notebook, learner: str) -> None:
    settings, trainer, note = _RUN_SETTINGS[learner]
    rows = {"ppo": "one row per update from the first update in which an episode has been completed",
            "sac": "one row each time episodes have been completed"}[learner]
    tail = r" and the temperature $\alpha$" if learner == "sac" else ""
    _md(nb, learner, r"""
**A tiny training run.** DMDP-Lag (@U@) is trained for 4000 steps and written to a temporary folder in the layout of `results/`.
@NOTE@
The log has @ROWS@: the mean over the most recent training episodes of the return $R$ of the training reward without $-\beta\,\mathbb{D}_t$ (`ep_return`), of the benchmark return (`ep_task_return`), the cost, the dissimilarity, the fraction of steps within $\mathbb{D}_{\mathsf{thr}}$, the pitch, and the fraction of flipped steps, together with the multiplier@TAIL@.
""".replace("@NOTE@", note).replace("@ROWS@", rows).replace("@TAIL@", tail))
    nb.code(f"""
import shutil
import tempfile

demo_root = tempfile.mkdtemp()
demo = make_config("HalfCheetah", "DMDP-Lag", seed=0, learner=LEARNER,
                   {settings})
demo_dir = run_dir(demo_root, demo.env, demo.method, demo.seed)
log = pd.DataFrame({trainer}(demo, demo_dir, verbose=False))
print("files of the run:", sorted(os.listdir(demo_dir)))
log.drop(columns="seconds").round(3)
""")
    _md(nb, learner, r"""
**Evaluation of the tiny run.** `evaluate_policy` runs 8 episodes at each of two noise levels and stores `evaluation.npz`, `load_evaluation` reads it back as `{noise level: {name: array}}`, and `summarize_run` computes one row per noise level, as for the stored runs in Section 11.
The columns are the returns $R$, $r^{\mathsf{b}}$, and $r^{\mathsf{eva}}$, the cost, the stability distances `D_early` and `D_late`, the target distance `D_target`, the fraction `within` of steps with $\mathbb{D}_t\leq\mathbb{D}_{\mathsf{thr}}$, and the fraction `flipped` of steps with a flipped torso.
""")
    nb.code(r"""
evaluate_policy(os.path.join(demo_dir, "policy.pt"), (0.0, 0.3), episodes=8, out_path=os.path.join(demo_dir, "evaluation.npz"))
rows, curves = summarize_run(load_evaluation(demo_root, demo.env, demo.method, demo.seed), load_config(demo_root, demo.env, demo.method, demo.seed))
shutil.rmtree(demo_root)
pd.DataFrame(rows).set_index("nu")[["ret", "ret_benchmark", "ret_eva", "cost", "D_early", "D_late", "D_target", "within", "flipped"]].round(3)
""")


# ---------------------------------------------------------------------------------------------------------------------
# 11 Stored results
# ---------------------------------------------------------------------------------------------------------------------
def stored(nb: Notebook, learner: str) -> None:
    tag = {"ppo": "main", "sac": "sac"}[learner]
    steps = {"ppo": "3 million steps per run", "sac": "1 million steps per run"}[learner]
    command = {"ppo": "python run_all.py --workers 18",
               "sac": "python run_all.py --learner sac --methods SAC SAC-Lag Dist-SAC Dist-SAC-Lag Implicit-SAC Implicit-SAC-Lag DMDP DMDP-Lag --tag sac --workers 20"}[learner]
    appendix = {"ppo": "Appendix F", "sac": "Section 5.1"}[learner]
    learner_argument = ', learner="sac"' if learner == "sac" else ""
    trainer = "train_sac" if learner == "sac" else "train"
    calls = [(f'cfg = make_config("HalfCheetah", method, seed{learner_argument})', steps),
             (f"{trainer}(cfg, out)", f"out: results/{tag}/HalfCheetah/<method>/seed<seed>")]
    width = max(len(code) for code, _ in calls) + 3
    calls = "\n".join(f"{code.ljust(width)}# {comment}" for code, comment in calls)
    _md(nb, learner, r"""
## 11 Stored results (`results/@TAG@`)

The folder `results/@TAG@/HalfCheetah/<method>/seed<s>/` holds the eight methods of the @U@ study with five seeds each: `config.json`, `training.csv`, `episodes.csv`, `policy.pt`, and `evaluation.npz`.
The command

```bash
@COMMAND@
```

runs the following for every method and seed, with `--workers` runs in parallel (`run_one` in `run_all.py`):

```python
@CALLS@
evaluate_policy(os.path.join(out, "policy.pt"), EVAL_NOISES, EVAL_EPISODES, os.path.join(out, "evaluation.npz"))
```

A run whose `policy.pt` exists is not trained again, and noise levels that `evaluation.npz` already holds are not evaluated again.
The tables below are computed from the stored files with the functions of Section 9, and the tables of @APPENDIX@ come from the same functions.
Values are the mean $\pm$ the standard error over the five seeds, and the rows follow the order of the tables of the paper.
`evaluation.npz` has about 11 MB per run and is not kept in the repository, where `run_all.py` writes it again from `policy.pt`; without it, the notebook shows the training logs only.
""".replace("@TAG@", tag).replace("@COMMAND@", command).replace("@CALLS@", calls).replace("@APPENDIX@", appendix))
    _code(nb, learner, r'''
from IPython.display import display

RESULTS = os.path.join("results", "@TAG@")


def summary_table(frame, columns):
    """Mean ± standard error over the seeds of the columns, for the methods of ALL_METHODS, in this order."""
    index, rows = [], []
    for method in ALL_METHODS:
        runs = frame[frame["method"] == method]
        if len(runs):
            index.append(method_label(method, LEARNER))
            rows.append({header: f"{runs[column].mean():.{digits}f} ± {runs[column].sem():.{digits}f}"
                         for column, (header, digits) in columns.items()})
    return pd.DataFrame(rows, index=pd.Index(index, name="method"))
'''.replace("@TAG@", tag))
    _md(nb, learner, r"""
**Training logs.** The last row of `training.csv` of every run is the mean over the most recent training episodes at the end of training, with the stochastic policy and the noise level $\eta\sim U[0,0.3]$ of training.
The return $R$ is that of the reward the method is trained on, without $-\beta\,\mathbb{D}_t$ and $-\lambda c_t$, and $r^{\mathsf{b}}$ is the benchmark return.
The columns are the cost, the mean window dissimilarity $\mathbb{D}$ per step, and the fraction of steps with a flipped torso.
""")
    _code(nb, learner, r"""
TRAIN_COLUMNS = {"ep_return": ("return R", 0), "ep_task_return": ("return r^b", 0), "ep_cost": ("cost", 1),
                 "ep_dissimilarity": ("D", 2), "ep_flipped": ("flipped", 2)}
finals = []
for method in ALL_METHODS:
    for seed in SEEDS:
        log = load_training(RESULTS, "HalfCheetah", method, seed)
        if log is not None:
            finals.append(dict(method=method, seed=seed, **log.iloc[-1][list(TRAIN_COLUMNS)]))
summary_table(pd.DataFrame(finals), TRAIN_COLUMNS) if finals else print("no stored runs found in", RESULTS)
""")
    _md(nb, learner, r"""
**Evaluation.** `summarize_run` is applied to the stored evaluation of every run at the noise levels $\eta=0$ and $\eta=0.2$ (64 episodes per run).
The columns are the return of the evaluation reward $r^{\mathsf{eva}}_t$ (Section 9), the episodic cost, the stability distance $\mathbb{D}(\rho_k\|\hat\rho)$ averaged over $k<200$ and over $k\geq500$, the distance $\mathbb{D}(\hat\rho\|\rho^{v}_\star)$ of the stable velocity distribution to the target, and the fraction of steps with $\mathbb{D}_t\leq\mathbb{D}_{\mathsf{thr}}$.
""")
    _code(nb, learner, r"""
EVAL_COLUMNS = {"ret_eva": ("return r^eva", 0), "cost": ("cost", 1),
                "D_early": ("D(ρ_k‖ρ̂), k<200", 3), "D_late": ("D(ρ_k‖ρ̂), k≥500", 3), "D_target": ("D(ρ̂‖ρ⋆)", 3),
                "within": ("P[D ≤ D_thr]", 2)}
rows = []
for method in ALL_METHODS:
    for seed in SEEDS:
        stored_evaluation = load_evaluation(RESULTS, "HalfCheetah", method, seed)
        if stored_evaluation is None:
            continue
        stored_evaluation = {nu: record for nu, record in stored_evaluation.items() if nu in (0.0, 0.2)}
        for row in summarize_run(stored_evaluation, load_config(RESULTS, "HalfCheetah", method, seed))[0]:
            rows.append(dict(row, method=method, seed=seed))
evaluated = pd.DataFrame(rows)
if evaluated.empty:
    print("no evaluation.npz found in", RESULTS, "- run_all.py writes it")
else:
    for nu in (0.0, 0.2):
        print(f"noise level eta = {nu}")
        display(summary_table(evaluated[evaluated["nu"] == nu], EVAL_COLUMNS))
""")
