"""Test sets, calibration of the residuals, and the hypothesis table of Appendix D.

Every sup over Omega_com (or Omega_com x A) in eqs. (def_delta_U_finite_sample), (def_delta_f_finite_sample),
(online_plugin_perturbation_bound), (model_free_decrease_estimation_error),
(model_free_parameter_estimation_perturbation) and (uniform_model_consistency_main) is a maximum over these test sets.
For the deployed runs the maxima are recomputed on an independent check set (``independent_points``).

Calibration: the score S_i of a run is its Delta (Theorem 3, +inf if the learned margin or the learner failed) or
Delta_dec,n (Theorem 5). With n_rep independent calibration runs and Delta_bar = max_i S_i,
P_calib[F(Delta_bar) >= q] >= 1 - q^n_rep, where F is the distribution function of the score (also with atoms and
infinite scores). So, given the calibration, an independent deployed run satisfies the theorem's event
{S <= Delta_bar} with probability at least q = 1 - alpha, with confidence 1 - q^n_rep (0.994 for n_rep = 49). The
proofs are pathwise: on every data realization the conclusion holds with the run's own Delta in place of Delta_bar.
"""

from __future__ import annotations

import numpy as np

from . import system as S


def test_points(n_mu=161, n_vs=61, n_near=2000, seed=7):
    """T_omega: uniform grid of Omega_com (boundary included), points near omega_star, and omega_star itself."""
    rng = np.random.default_rng(seed)
    mg, sg = np.meshgrid(np.linspace(S.MU_LO, S.MU_HI, n_mu), np.linspace(S.VS_LO, S.VS_HI, n_vs), indexing="ij")
    rho = 10 ** rng.uniform(-3, -0.5, n_near)
    th = rng.uniform(0, 2 * np.pi, n_near)
    mn, sn = S.project(S.MU_STAR + rho * np.cos(th), S.VS_STAR + rho * np.sin(th))
    return np.r_[mg.ravel(), mn, S.MU_STAR], np.r_[sg.ravel(), sn, S.VS_STAR]


def independent_points(n_uniform=60_000, n_near=20_000, seed=11):
    """Check set for the maxima on T_omega: uniform points of Omega_com and points omega_star + rho (cos th, sin th)
    with log10 rho ~ U[-5, -0.5], projected onto Omega_com. Drawn independently of T_omega (not nested in it), so
    the relative change of a maximum can have either sign; it reaches 100 times closer to omega_star than T_omega."""
    rng = np.random.default_rng(seed)
    mu, vs = rng.uniform(S.MU_LO, S.MU_HI, n_uniform), rng.uniform(S.VS_LO, S.VS_HI, n_uniform)
    rho = 10 ** rng.uniform(-5, -0.5, n_near)
    th = rng.uniform(0, 2 * np.pi, n_near)
    mn, sn = S.project(S.MU_STAR + rho * np.cos(th), S.VS_STAR + rho * np.sin(th))
    return np.r_[mu, mn], np.r_[vs, sn]


def model_test_points(n=20000, seed=123):
    """T_omega_a: uniform points of Omega_com x A and the 16 corners of the box."""
    rng = np.random.default_rng(seed)
    cols = [rng.uniform(lo, hi, n) for lo, hi in zip(S.LO, S.HI)]
    corners = np.array(np.meshgrid(*zip(S.LO, S.HI), indexing="ij")).reshape(4, -1)
    return tuple(np.r_[c, k] for c, k in zip(cols, corners))


def order_statistic(scores, q=0.9):
    """(Delta_bar, confidence) = (max of the calibration scores, 1 - q^n)."""
    scores = np.asarray(scores, float)
    return float(np.max(scores)), float(1.0 - q ** len(scores))


def residual_radius(bar, c):
    """Radius r_bar of the residual set {omega in Omega_com : c r^2 <= bar} of eq. (finite_sample_residual_set)."""
    return float(np.sqrt(bar / c)) if np.isfinite(bar) else np.inf


def capped_decrease(d_hat, d, c):
    """min{D_hat_n, -c D}: satisfies eq. (model_free_empirical_decrease_certificate) with sigma_n = c r^2."""
    return np.minimum(d_hat, -c * d)


def smallest_residual(true_dec, d, c):
    """Smallest residual for which the decrease inequality with sigma = c r^2 holds at the points: max[dec + c D]."""
    return float(np.max(true_dec + c * d))


def relative_change(new, old):
    """(new - old) / |old|; 0 if both are 0 and infinite if only old is 0."""
    if old == 0:
        return 0.0 if new == 0 else float(np.inf)
    return float((new - old) / abs(old))


def num(x, digits=2):
    """LaTeX math for x with two significant digits, a x 10^b below 0.01; infinity means no certificate.

    x is rounded to ``digits`` significant digits first, so that 0.0996 gives 0.10 and 9.96 gives 10."""
    if not np.isfinite(x):
        return r"\infty"
    x = float(f"{x:.{digits}g}")
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        m, e = f"{x:.{digits - 1}e}".split("e")
        return rf"{m}\times10^{{{int(e)}}}"
    return f"{x:.{max(digits - 1 - int(np.floor(np.log10(abs(x)))), 0)}f}"


def fmt(x, digits=2):
    """``num`` in math mode; None or NaN as a dash."""
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return "--"
    return f"${num(float(x), digits)}$"


def hypothesis_rows(s):
    """Rows (result, hypothesis, status, evidence) of the hypothesis table, with the statuses read from the summary."""
    t, off, mf = s["true"], s["offline"], s["modelfree"]
    levels = sorted(off, key=int)
    top = off[levels[-1]]
    exact_bounds = t["ratio_min"] >= 1 - 1e-6 and t["ratio_max"] <= S.KAPPA2 + 1e-6
    offsets = [off[l]["deployed"]["offset"] for l in levels]
    no_cert = [off[l]["n_tr"] for l in levels if not np.isfinite(off[l]["Delta_bar"])]
    e_f = [off[l]["e_f_median"] for l in levels]
    fine = s["fine_change_max"]                       # Delta_U, Delta_f, Delta, c_hat, Delta_dec,n
    sets = s["test_sets"]
    n_cal = top["n_calib"]
    conf = top["confidence"]
    lim_off = max(off[l]["deployed"]["limit"] for l in levels)
    rows = [
        ("All", r"closure; $\mathbb{D}$-compatibility with $\alpha_1(r)=\alpha_2(r)=r^2$", "exact",
         r"Gaussian AR(1); $\mathbb{D}=W_2^2=r^2$"),
        ("All", r"$\Omega_{\mathsf{com}}$ compact and positively invariant under every closed loop",
         r"exact; for $\widehat F_{\mathsf{off}}$ by construction", r"Lemma~\ref{lemma:toy_facts}; $\hat f_{\mathsf{off}}$ projected"),
        (r"Thm.~\ref{theo:parameter_stability_means_distribution_stability}", r"optimal $\phi_\star$; $U_\star^{\mathsf{R}}$ finite, continuous, Bellman identity; "
         r"$F_{\phi_\star}(\omega_\star)=\omega_\star$; $\kappa_1(r)=r^2$, $\kappa_2(r)=2.56\,r^2$",
         "exact" if exact_bounds else "exact; numerical check fails",
         rf"Lemma~\ref{{lemma:toy_facts}}; $\max\tilde U/\mathbb{{D}}={t['ratio_max']:.2f}$"),
        (r"Thms.~\ref{theo:stability_finite_samples}--\ref{theo:model_free_online_policy_learning}", r"$U_\star^{\mathsf{R}}$ and the sups over $\Omega_{\mathsf{com}}$", "numerical",
         rf"$\tilde U$ with $\varepsilon_{{\mathrm{{num}}}}={num(t['eps_num'])}$; maxima over {sets['T_omega']:,} points "
         rf"({sets['T_omega_a']:,} for $e_f$); {sets['independent']:,} independent points change $\Delta$, $\hat c$, "
         rf"$\Delta_{{\mathsf{{dec}},n}}$ of the deployed run by at most ${num(100 * fine)}\%$"),
        (r"Thm.~\ref{theo:stability_finite_samples}", r"$F_{\mathsf{off}}(\omega_\star)=\omega_\star$", "fails" if min(offsets) > 0 else "holds",
         r"offset $>0$; not used by~\eqref{eq:true_lyapunov_decrease_with_residual}" if min(offsets) > 0
         else "offset $0$"),
        (r"Thm.~\ref{theo:stability_finite_samples}", r"learned margin with $\sigma(r)=0.75\,r^2$; $\Delta\le\bar\Delta$ with probability $\ge0.9$",
         "computed" if not no_cert else "computed; fails at $n_{\\mathsf{tr}}=" + ",".join(map(str, no_cert)) + "$",
         rf"per run; maximum of {n_cal} runs, confidence ${conf:.3f}$"),
        (r"Thms.~\ref{theo:online_plugin_stability}--\ref{theo:model_free_online_policy_learning}", r"estimation event for all $t\ge0$ with probability $\ge0.95$", "exact", r"growing $M_t$"),
        (r"Thm.~\ref{theo:model_free_online_policy_learning}", r"$W_n=\mathbb{D}$: bounds and continuity", "exact", "--"),
        (r"Thm.~\ref{theo:model_free_online_policy_learning}", r"empirical decrease with $\sigma_n(r)=0.25\,r^2$; $\Delta_{\mathsf{dec},n}\le\bar\Delta_{\mathsf{dec},n}$ "
         r"with probability $\ge0.9$", "by construction; computed",
         rf"capped estimate; maximum of {n_cal} runs, with the true $f$"),
        (r"Thms.~\ref{theo:stability_finite_samples}--\ref{theo:model_free_online_policy_learning}", "asymptotic clause", "fails" if lim_off > 0 else "holds", "residual sets only"),
        (r"Prop.~\ref{prop:asymptotic_recovery_rotated_optimizers}", "uniform model consistency", "trend" if all(np.diff(e_f) < 0) else "no monotone trend",
         r"$e_f$"),
        (r"Prop.~\ref{prop:asymptotic_recovery_rotated_optimizers}", r"$\Theta$ compact, $\phi$ continuous; $J^{\mathsf{R}}$ finite and continuous",
         "exact; fails" if min(offsets) > 0 else "exact", r"node actions; $J^{\mathsf{R}}$ infinite with an offset"),
    ]
    return rows
