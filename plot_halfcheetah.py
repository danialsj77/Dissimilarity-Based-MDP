"""Draws the HalfCheetah robot of the study with its actions and regulated features (paper Appendix E, Figure halfcheetah).

The robot is drawn from the kinematics of the simulator: the stored policies of two runs are rolled out without noise, the
joint positions qpos of one step are kept, and the capsules of the model are placed by the forward kinematics of MuJoCo
at these positions. The left panel shows an upright step of DMDP-Lag trained with PPO (results/main,
seed 0) with the six joint torques a_{t,1}, ..., a_{t,6} (the action), the forward velocity v_t and the pitch theta_t of
the torso. The right panel shows a flipped step (|theta_t| > pi/2) of SAC (results/sac, seed 0).

    python plot_halfcheetah.py            # writes figures/halfcheetah.pdf and figures/halfcheetah.png
"""

from __future__ import annotations

import os
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from matplotlib.patches import Arc, FancyArrowPatch

from dmdp_sg.envs import env_settings, make_env
from dmdp_sg.evaluate import EVAL_SEED, load_policy

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.environ.get("DMDP_FIG_DIR", os.path.join(HERE, "figures")))
UPRIGHT = os.path.join(HERE, "results", "main", "HalfCheetah", "DMDP-Lag", "seed0", "policy.pt")
FLIPPED = os.path.join(HERE, "results", "sac", "HalfCheetah", "SAC", "seed0", "policy.pt")
K0 = 500                                   # steps k >= K0 (the steps used for the stable distribution)
JOINTS = ("bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot")   # actuated joints, in the order of the action entries
LINKS = ("torso", "head", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot")
# position of the label a_{t,i} relative to its joint (x, z offset and horizontal alignment), placed in free space of the
# upright pose drawn in the left panel
LABEL_OFFSETS = {"bthigh": (-0.09, 0.06, "right"), "bshin": (0.05, -0.09, "left"), "bfoot": (-0.09, -0.02, "right"),
                 "fthigh": (0.09, -0.01, "left"), "fshin": (-0.05, -0.09, "right"), "ffoot": (0.07, -0.08, "left")}


@torch.no_grad()
def rollout_states(path: str, steps: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """qpos and forward velocity of every step of one noise-free evaluation episode of the stored policy."""
    agent, rms, cfg = load_policy(path)
    env = make_env(nu=0.0, seed=EVAL_SEED, **env_settings(cfg))
    obs, _ = env.reset(seed=EVAL_SEED)
    qpos, vel = [], []
    for _ in range(steps):
        action, _, _ = agent.act(torch.as_tensor(rms.normalize(obs[None])), deterministic=True)
        action = np.clip(action.numpy()[0], env.action_space.low, env.action_space.high)
        obs, _, terminated, truncated, info = env.step(action)
        qpos.append(env.unwrapped.data.qpos.copy())
        vel.append(info["x_velocity"])
        if terminated or truncated:
            break
    env.close()
    return np.array(qpos), np.array(vel)


def pitch(qpos: np.ndarray) -> np.ndarray:
    """Pitch theta of the torso in (-pi, pi] (qpos[2] is the rotation about the y axis)."""
    return np.arctan2(np.sin(qpos[..., 2]), np.cos(qpos[..., 2]))


def geometry(qpos: np.ndarray):
    """Capsule segments (x-z plane) of the links and the positions of the actuated joints for the joint positions qpos."""
    env = make_env(nu=0.0, seed=0, **env_settings(load_policy(UPRIGHT)[2]))
    model, data = env.unwrapped.model, env.unwrapped.data
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    links = {}
    for name in LINKS:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        radius, half = model.geom_size[g][:2]
        center, axis = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)[:, 2]
        a, b = center - half * axis, center + half * axis
        links[name] = (np.array([a[0], a[2]]), np.array([b[0], b[2]]), radius)
    joints = {}
    for name in JOINTS:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joints[name] = np.array([data.xanchor[j][0], data.xanchor[j][2]])
    torso = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")]
    env.close()
    return links, joints, np.array([torso[0], torso[2]])


def draw_robot(ax, qpos: np.ndarray, color: str, labels: bool) -> tuple[np.ndarray, float]:
    links, joints, torso = geometry(qpos)
    shift = np.array([-torso[0], 0.0])                     # torso centre at x = 0
    lw = 72.0 * 2 * 0.046 / 1.25                           # capsule diameter in points for the axis scale below
    for name, (a, b, _) in links.items():
        a, b = a + shift, b + shift
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, solid_capstyle="round", zorder=2)
    centre = torso + shift
    for i, name in enumerate(JOINTS):
        p = joints[name] + shift
        ax.plot(*p, "o", ms=3.2, mfc="white", mec="k", mew=0.6, zorder=5)
        if labels:
            # the label of the torque a_{t,i} lies outward from the torso centre, with a curved arrow around the joint
            out = p - centre
            out = out / np.linalg.norm(out)
            start = np.rad2deg(np.arctan2(out[1], out[0])) + 40.0
            ax.add_patch(Arc(p, 0.13, 0.13, theta1=start, theta2=start + 280.0, lw=0.7, color="C3", zorder=4))
            tip = np.deg2rad(start + 280.0)
            end = p + 0.065 * np.array([np.cos(tip), np.sin(tip)])
            tangent = np.array([-np.sin(tip), np.cos(tip)])
            ax.annotate("", xy=end + 0.012 * tangent, xytext=end - 0.012 * tangent,
                        arrowprops=dict(arrowstyle="-|>", color="C3", lw=0.7, mutation_scale=5), zorder=4)
            dx, dy, ha = LABEL_OFFSETS[name]
            ax.text(p[0] + dx, p[1] + dy, rf"$a_{{t,{i + 1}}}$", fontsize=7.5, color="C3", ha=ha, va="center")
    return centre, float(pitch(qpos))


def draw_pitch(ax, centre: np.ndarray, theta: float, radius: float, label_offset: tuple[float, float]):
    # horizontal reference from the torso centre and the arc from it to the torso axis
    ax.plot([centre[0], centre[0] + radius + 0.12], [centre[1], centre[1]], color="0.3", lw=0.6, ls="--", zorder=3)
    # MuJoCo's rotation about y turns the torso axis (1, 0) into (cos theta, -sin theta) in the x-z plane
    angle = -np.rad2deg(theta)
    ax.add_patch(Arc(centre, 2 * radius, 2 * radius, theta1=min(0.0, angle), theta2=max(0.0, angle), lw=0.8, color="C0",
                     zorder=3))
    mid = np.deg2rad(angle / 2)
    ax.text(centre[0] + radius * np.cos(mid) + label_offset[0], centre[1] + radius * np.sin(mid) + label_offset[1],
            r"$\theta_t$", fontsize=8.5, color="C0", ha="left", va="center")


def main():
    TEX_BIN = "/Library/TeX/texbin"
    if os.path.isdir(TEX_BIN) and TEX_BIN not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = TEX_BIN + os.pathsep + os.environ["PATH"]
    mpl.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    if shutil.which("latex"):
        mpl.rcParams.update({"text.usetex": True,
                             "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}"})
    else:
        mpl.rcParams.update({"mathtext.fontset": "cm"})

    q_up, v_up = rollout_states(UPRIGHT)
    q_fl, v_fl = rollout_states(FLIPPED)
    # upright step: the largest |theta_t| among the steps k >= K0 whose torso is within the pitch target |theta| <= 0.2,
    # so that the angle is visible; flipped step: the step k >= K0 whose |theta_t| is closest to pi
    th_up, th_fl = np.abs(pitch(q_up)), np.abs(pitch(q_fl))
    k_up = K0 + int(np.argmax(np.where(th_up[K0:] <= 0.2, th_up[K0:], -1.0)))
    k_fl = K0 + int(np.argmin(np.abs(th_fl[K0:] - np.pi)))

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.05), gridspec_kw=dict(wspace=0.05))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-0.08, 1.12)
        ax.axis("off")
        ax.plot([-1.25, 1.25], [0, 0], color="0.3", lw=0.8)
        ax.fill_between([-1.25, 1.25], [0, 0], [-0.08, -0.08], color="0.9", lw=0)

    centre, theta = draw_robot(axes[0], q_up[k_up], "0.35", labels=True)
    draw_pitch(axes[0], centre, theta, radius=0.30, label_offset=(0.04, 0.09))
    axes[0].add_patch(FancyArrowPatch(centre + np.array([0.0, 0.33]), centre + np.array([0.55, 0.33]), arrowstyle="-|>",
                                      mutation_scale=8, lw=1.0, color="C2"))
    axes[0].text(centre[0] + 0.58, centre[1] + 0.33, r"$v_t$", fontsize=8.5, color="C2", va="center")
    axes[0].set_title("upright", fontsize=9)

    centre, theta = draw_robot(axes[1], q_fl[k_fl], "0.55", labels=False)
    draw_pitch(axes[1], centre, theta, radius=0.22, label_offset=(-0.04, 0.09))
    axes[1].set_title(r"flipped, $|\theta_t|>\pi/2$", fontsize=9)

    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"halfcheetah.{ext}"), dpi=200)
    print(f"upright: step {k_up}, theta {pitch(q_up[k_up]):.3f} rad, v {v_up[k_up]:.2f}; "
          f"flipped: step {k_fl}, theta {pitch(q_fl[k_fl]):.3f} rad, v {v_fl[k_fl]:.2f}")


if __name__ == "__main__":
    main()
