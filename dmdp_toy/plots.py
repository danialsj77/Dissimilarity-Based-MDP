"""Figure fig:toy_certificates and Tables tab:toy_hypotheses, tab:toy_results of Appendix D.

The tables are written as bare tabulars (booktabs, footnotesize); caption and label are set in main.tex.
"""

from __future__ import annotations

import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402

from . import system as S                            # noqa: E402
from .certify import fmt, hypothesis_rows           # noqa: E402

TEX_BIN = "/Library/TeX/texbin"
COLORS = {"opt": "#222222", "off": "#d62728", "plug": "#ff7f0e", "mf": "#1f77b4", "ref": "#555555"}
WHITE_BOX = dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8)


def style(use_tex=True):
    """Font 8, serif; LaTeX with Latin Modern if a latex executable is available."""
    if use_tex and os.path.isdir(TEX_BIN) and TEX_BIN not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = TEX_BIN + os.pathsep + os.environ["PATH"]
    use_tex = use_tex and shutil.which("latex") is not None
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                         "text.usetex": use_tex})
    if use_tex:
        plt.rcParams["text.latex.preamble"] = (r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}"
                                               r"\usepackage{amssymb}\usepackage{bm}")
    else:
        plt.rcParams["mathtext.fontset"] = "cm"
    return use_tex


def _region_outline():
    """Boundary of R = {omega in Omega_com : deadbeat(omega) in A}, in (mu, varsigma)."""
    vs = np.linspace(S.VS_LO, np.sqrt(S.VS_STAR ** 2 - S.NOISE_SD ** 2) / S.A1_LO, 60)
    a1 = np.sqrt(S.VS_STAR ** 2 - S.NOISE_SD ** 2) / vs
    lo, hi = (1 - S.A2_HI * (1 - a1)) / a1, (1 - S.A2_LO * (1 - a1)) / a1
    return np.r_[lo, hi[::-1], lo[0]], np.r_[vs, vs[::-1], vs[0]]


def _curve_points(ax, x, y, n=12):
    """Display coordinates of the polyline (x, y), with n points per segment (obstacles for the labels)."""
    p = ax.transData.transform(np.column_stack([np.ravel(x), np.ravel(y)]))
    t = np.linspace(0.0, 1.0, n, endpoint=False)[None, :, None]
    return np.vstack([(p[:-1, None] + t * (p[1:, None] - p[:-1, None])).reshape(-1, 2), p[-1:]])


def _grid(width, height, step=2.0):
    """Offsets covering a width x height box, at most ``step`` display units apart, centred at 0."""
    u = np.linspace(-width / 2, width / 2, max(2, int(np.ceil(width / step)) + 1))
    v = np.linspace(-height / 2, height / 2, max(2, int(np.ceil(height / step)) + 1))
    g = np.meshgrid(u, v)
    return np.column_stack([g[0].ravel(), g[1].ravel()])


def _box_points(x0, y0, x1, y1):
    """Points covering an axis-aligned display box (an obstacle for the labels)."""
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2]) + _grid(x1 - x0, y1 - y0)


class _Labels:
    """Places text labels clear of the data: a candidate position is used only if the label's box (rotated with the
    label) lies inside the axes and contains none of the obstacle points (data, curves, reference lines, legend, and
    the labels placed before)."""

    def __init__(self, ax, obstacles, fontsize=7):
        self.ax, self.fontsize = ax, fontsize
        self.renderer = ax.figure.canvas.get_renderer()
        self.pts = np.vstack([np.empty((0, 2))] + list(obstacles))
        self.pt = ax.figure.dpi / 72.0                                 # display units per point

    def _size(self, text):
        t = self.ax.text(0, 0, text, fontsize=self.fontsize, transform=None)
        bb = t.get_window_extent(self.renderer)
        t.remove()
        return bb.width, bb.height

    def _box(self, c, th, w, h):
        """Points covering the box of width w and height h centred at c, rotated by th (display units)."""
        e = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        return c + _grid(w, h) @ e

    def _free(self, c, th, w, h):
        a = self.ax.get_window_extent(self.renderer)
        box = self._box(c, th, w, h)
        if box[:, 0].min() < a.x0 or box[:, 0].max() > a.x1 or box[:, 1].min() < a.y0 or box[:, 1].max() > a.y1:
            return False
        d = self.pts - c
        u = d[:, 0] * np.cos(th) + d[:, 1] * np.sin(th)
        v = -d[:, 0] * np.sin(th) + d[:, 1] * np.cos(th)
        return not np.any((np.abs(u) <= w / 2 + self.pt) & (np.abs(v) <= h / 2 + self.pt))

    def _put(self, text, xy, offset_px, th, c, w, h, leader=False, **kw):
        if leader:                                          # thin line from the label to its anchor on the curve
            kw["arrowprops"] = dict(arrowstyle="-", lw=0.5, color=COLORS["ref"], shrinkA=1, shrinkB=0)
            self.pts = np.vstack([self.pts, c + np.linspace(0, 1, 40)[:, None] * (-offset_px)])
        t = self.ax.annotate(text, xy, xytext=offset_px / self.pt, textcoords="offset points", ha="center",
                             va="center", rotation=np.degrees(th), rotation_mode="anchor", fontsize=self.fontsize,
                             **kw)
        self.pts = np.vstack([self.pts, self._box(c, th, w, h)])
        return t

    def place(self, text, candidates, force=False, leader=False, **kw):
        """Horizontal label; candidates (x, y, ha, va, dx, dy): anchor in data coordinates, offset in points; with
        ``leader`` a thin line joins the label to the anchor. Returns the Text, or None if no candidate is free; with
        ``force`` the first candidate is then used, on a white box."""
        w, h = self._size(text)
        fx = {"left": 0.5, "center": 0.0, "right": -0.5}
        fy = {"bottom": 0.5, "center": 0.0, "top": -0.5}
        options = []
        for x, y, ha, va, dx, dy in candidates:
            off = self.pt * np.array([dx, dy]) + np.array([fx[ha] * w, fy[va] * h])
            options.append(((x, y), off, self.ax.transData.transform((x, y)) + off))
            if self._free(options[-1][2], 0.0, w, h):
                return self._put(text, *options[-1][:2], 0.0, options[-1][2], w, h, leader=leader, **kw)
        if force and options:
            kw.setdefault("bbox", WHITE_BOX)
            return self._put(text, *options[0][:2], 0.0, options[0][2], w, h, leader=leader, **kw)
        return None

    def along(self, text, x, y, fracs, side, gap=1.5, force=False, **kw):
        """Label parallel to the curve (x, y) (data coordinates), on ``side`` ('above' or 'below') at a gap in points,
        at the first fraction of the curve's length (in x) where it is free. The label follows the tangent or, if
        that is not free, the chord of the curve under the label. Returns the Text or None (see ``place``)."""
        w, h = self._size(text)
        p = self.ax.transData.transform(np.column_stack([x, y]))
        sgn = 1.0 if side == "above" else -1.0
        first = None
        for q in fracs:
            i = int(round(q * (len(x) - 1)))
            for mode in ("tangent", "chord"):
                if mode == "tangent":
                    j0, j1 = max(i - 1, 0), min(i + 1, len(x) - 1)
                    a, b = p[j0], p[j1]
                    mid = p[i]
                else:                                       # chord over the label's horizontal extent
                    j = np.nonzero(np.abs(p[:, 0] - p[i, 0]) <= w / 2)[0]
                    if len(j) < 2:
                        continue
                    a, b = p[j[0]], p[j[-1]]
                    mid = 0.5 * (a + b)
                th = np.arctan2(b[1] - a[1], b[0] - a[0])
                n = sgn * np.array([-np.sin(th), np.cos(th)])
                c = mid + n * (gap * self.pt + h / 2)
                xy = self.ax.transData.inverted().transform(mid)
                first = first or (xy, c - mid, th, c)
                if self._free(c, th, w, h):
                    return self._put(text, xy, c - mid, th, c, w, h, **kw)
        if force and first:
            kw.setdefault("bbox", WHITE_BOX)
            return self._put(text, *first, w, h, **kw)
        return None


def _decrease_panel(ax, r, dec, c, bound, radius, ylabel, title, sigma_label):
    color = COLORS["mf"] if c < 0.5 else COLORS["off"]
    ax.scatter(r, dec, s=1.0, color=color, alpha=0.25, lw=0, rasterized=True)
    rr = np.linspace(0, r.max(), 200)
    ax.plot(rr, -c * rr ** 2, color=COLORS["ref"], lw=0.9)
    top = dec.max()
    certified = np.isfinite(bound)                         # no curve without a certificate
    vline = certified and radius < r.max()                 # otherwise the residual set covers the test points
    if certified:
        ax.plot(rr, -c * rr ** 2 + bound, color=COLORS["ref"], lw=0.9, ls="-.")
        if vline:
            ax.axvline(radius, color=COLORS["ref"], lw=0.8, ls="--")
        top = max(top, bound)
    lo = min(dec.min(), -c * r.max() ** 2)
    ax.set_ylim(lo * 1.05, top + 0.15 * abs(lo))
    ax.set_xlim(0, 1.02 * r.max())
    ax.set_xlabel(r"$\|\omega-\omega_\star\|$")
    ax.set_ylabel(ylabel, labelpad=1)
    ax.set_title(title, fontsize=8)
    # labels of the reference lines, clear of the data and of each other: -sigma below its curve (between the curve
    # and the data), the certified bound above its curve, the residual radius beside its line
    obstacles = [ax.transData.transform(np.column_stack([r, dec])), _curve_points(ax, rr, -c * rr ** 2)]
    if certified:
        obstacles.append(_curve_points(ax, rr, -c * rr ** 2 + bound))
    if vline:
        obstacles.append(_curve_points(ax, [radius, radius], ax.get_ylim(), n=200))
    lab = _Labels(ax, obstacles)
    if not lab.along(sigma_label, rr, -c * rr ** 2, (0.5, 0.6, 0.4, 0.7, 0.8, 0.9, 0.3, 0.95), "below"):
        # no room between the curve and the data: label on the data side with a leader to the curve
        on_sigma = [(q * rr[-1], -c * (q * rr[-1]) ** 2) for q in (0.65, 0.75, 0.55)]
        lab.place(sigma_label, [(x, y, "center", "center", d * np.cos(np.radians(a)), d * np.sin(np.radians(a)))
                                for x, y in on_sigma for d in (18, 24, 32, 42, 54)
                                for a in (-135, -120, -150, -105, -165, -90, 180)], force=True, leader=True)
    if certified:
        on_bound = lambda q: (q * rr[-1], -c * (q * rr[-1]) ** 2 + bound)                 # noqa: E731
        for text in ("certified bound", "certified\nbound"):
            if lab.place(text, [(*on_bound(q), "left", "bottom", 0, 2) for q in (0.6, 0.5, 0.4, 0.3, 0.2, 0.1)]):
                break
        else:
            lab.along("certified bound", rr, -c * rr ** 2 + bound, (0.7, 0.6, 0.5, 0.4, 0.3), "above", force=True)
    if vline:
        lab.place("residual\nradius", [(radius, y, ha, "center", dx, 0) for y in np.linspace(*ax.get_ylim(), 11)[
            [6, 5, 7, 4, 8, 3, 9, 2]] for ha, dx in (("left", 2), ("right", -2))], force=True)


def fig_certificates(d, path, use_tex=True):
    """Four panels: optimal value and trajectories, decrease of Theorem 3, decrease of Theorem 5, D(rho_k||rho_star)."""
    style(use_tex)
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.4), gridspec_kw=dict(wspace=0.62))
    # (a) optimal policy
    ax = axes[0]
    mg, sg = np.meshgrid(np.linspace(S.MU_LO, S.MU_HI, 241), np.linspace(S.VS_LO, S.VS_HI, 121), indexing="ij")
    cs = ax.contour(mg, sg, d["U"](mg, sg), levels=(0.1, 0.5, 1, 2, 4, 6), colors="#aaaaaa", linewidths=0.6)
    ax.clabel(cs, levels=(0.5, 2, 6), fontsize=5, inline=True, fmt="%g", inline_spacing=1)
    for tr in d["trajs"]:
        keep = np.r_[True, np.any(np.abs(np.diff(tr, axis=0)) > 1e-9, axis=1)]
        ax.plot(tr[keep, 0], tr[keep, 1], "-o", color=COLORS["opt"], lw=0.7, ms=2)
    rm, rs = _region_outline()
    ax.plot(rm, rs, "--", color=COLORS["off"], lw=0.8)
    ax.annotate(r"$U_\star^{\mathsf{R}}=\mathbb{D}$", (rm.max(), 0.75), xytext=(1.75, 1.02), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=COLORS["off"]))
    ax.plot(S.MU_STAR, S.VS_STAR, "*", color=COLORS["off"], ms=7, zorder=5)
    ax.annotate(r"$\omega_\star$", (S.MU_STAR, S.VS_STAR), xytext=(-12, -1), textcoords="offset points", fontsize=7,
                ha="right", va="center")
    ax.set_xlim(S.MU_LO, S.MU_HI)
    ax.set_ylim(S.VS_LO, S.VS_HI)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\varsigma$", labelpad=1)
    ax.set_title("(a) optimal policy", fontsize=8)
    # (b), (c) decreases of the deployed learned policies
    _decrease_panel(axes[1], d["off_r"], d["off_dec"], 0.75, d["off_bar"], d["off_radius"],
                    r"$\tilde U(F_{\mathsf{off}}(\omega))-\tilde U(\omega)$", "(b) Theorem 3", r"$-\sigma(r)$")
    _decrease_panel(axes[2], d["mf_r"], d["mf_dec"], 0.25, d["mf_bound"], d["mf_radius"],
                    r"$\mathbb{D}(F_n(\omega))-\mathbb{D}(\omega)$", "(c) Theorem 5", r"$-\sigma_n(r)$")
    # (d) D(rho_k || rho_star) from (-1, VS_HI)
    ax = axes[3]
    floor = 1e-9
    curves = [(d["D_opt"], COLORS["opt"], r"$\phi_\star$"),
              (d["D_off"], COLORS["off"], r"$\hat\phi_{\mathsf{off}}$"),
              (d["D_plug"], COLORS["plug"], "plug-in"),
              (d["D_mf"], COLORS["mf"], rf"$\phi_{{{d['n_label']}}}$")]
    for y, col, lab in curves:
        k = np.arange(len(y))
        pos = y > 1e-24
        ax.plot(k[pos], np.maximum(y[pos], floor), color=col, lw=0.9, label=lab)
    if np.any(d["D_opt"] <= 1e-24):
        hit = int(np.argmax(d["D_opt"] <= 1e-24))
        ax.plot(hit, floor, "v", color=COLORS["opt"], ms=4, clip_on=False, zorder=5)
        ax.annotate(r"$\rho_\star$ reached", (hit, floor), xytext=(4, 2), textcoords="offset points", fontsize=7,
                    va="bottom")
    lines = sorted([(y, lab) for y, lab in ((d["off_radius"] ** 2, "Thm. 3"), (d["mf_radius"] ** 2, "Thm. 5"))
                    if np.isfinite(y)], reverse=True)                    # residual sets {r <= r_bar}
    for y, _ in lines:
        ax.axhline(y, color=COLORS["ref"], lw=0.7, ls="--")
    ax.set_yscale("log")
    ax.set_xlim(0, 40)
    ax.set_ylim(floor, 1e3)
    ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
    ax.set_xlabel(r"step $k$")
    ax.set_ylabel(r"$\mathbb{D}(\rho_k\|\rho_\star)$", labelpad=1)
    ax.set_title(r"(d) dissimilarity to $\rho_\star$", fontsize=8)
    leg = ax.legend(loc="upper right", ncol=2, frameon=False, handlelength=1.0, handletextpad=0.4, columnspacing=0.8,
                    borderaxespad=0.1, labelspacing=0.2, fontsize=6.5)
    # residual-set labels at the right end, where the curves are flat, clear of the curves and the legend
    renderer = fig.canvas.get_renderer()
    obstacles = [_curve_points(ax, np.arange(len(y))[y > 1e-24], np.maximum(y[y > 1e-24], floor)) for y, _, _ in curves]
    obstacles += [_curve_points(ax, [0, 40], [y, y], n=200) for y, _ in lines]
    obstacles += [_box_points(*b.extents) for b in (leg.get_window_extent(renderer),
                                                    *(t.get_window_extent(renderer) for t in ax.texts))]
    lab = _Labels(ax, obstacles, fontsize=6.5)
    for i, (y, text) in enumerate(lines):
        sides = [("bottom", 1.5), ("top", -1.5)] if i == 0 else [("top", -1.5), ("bottom", 1.5)]
        lab.place(text, [(x, y, "right", va, 0, dy) for va, dy in sides for x in (38, 30, 22)], force=True,
                  bbox=WHITE_BOX)
    fig.savefig(path + ".pdf", dpi=300)
    fig.savefig(path + ".png", dpi=200)
    plt.close(fig)


def _head(*lines):
    return r"\begin{tabular}[b]{@{}c@{}}" + r"\\".join(lines) + r"\end{tabular}"


def write_tables(s, figdir):
    """table_toy_hypotheses.tex and table_toy_results.tex."""
    col = r">{\raggedright\arraybackslash}p{%.2f\linewidth}"
    rows = hypothesis_rows(s)
    lines = [r"{\footnotesize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}" + "".join(col % w for w in (0.10, 0.42, 0.14, 0.30)) + "@{}}", r"\toprule",
             r"Result & Hypothesis & Status & Evidence \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}}"]
    with open(os.path.join(figdir, "table_toy_hypotheses.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    off, on, mf = s["offline"], s["online"], s["modelfree"]
    head = ["Data", _head(r"$e_f$ or", r"$\Delta_{\mathsf{dec},n}$ (median)"), _head(r"$\bar\Delta$ or",
            r"$\bar\Delta_{\mathsf{dec},n}$"), _head(r"$\Delta_{\mathsf{on}}$ or", r"$\Delta_{\omega,n}$"),
            _head("residual", r"radius $\bar r$"), _head("smallest", "residual"), _head("limit", "distance"), "failed"]
    body = [r"\multicolumn{8}{@{}l}{\emph{Thm.~\ref{theo:stability_finite_samples}, offline}} \\"]
    for l in sorted(off, key=int):
        o = off[l]
        body.append(" & ".join([rf"$n_{{\mathsf{{tr}}}}={o['n_tr']}$", fmt(o["e_f_median"]), fmt(o["Delta_bar"]), "--",
                                fmt(o["radius"]), fmt(o["deployed"]["Delta_min"]), fmt(o["deployed"]["limit"]),
                                f"{o['failed']}/{o['n_runs']}"]) + r" \\")
    body += [r"\midrule", r"\multicolumn{8}{@{}l}{\emph{Thm.~\ref{theo:online_plugin_stability}, online plug-in}} \\",
             " & ".join([rf"$n_{{\mathsf{{tr}}}}={on['n_tr']}$", "--", fmt(on["Delta_bar_off"]), fmt(on["Delta_on_used"]),
                         fmt(on["radius"]), fmt(on["Delta_min"]), fmt(on["limit_median"]),
                         f"{on['failed']}/{on['n_runs']}"]) + r" \\",
             r"\midrule", r"\multicolumn{8}{@{}l}{\emph{Thm.~\ref{theo:model_free_online_policy_learning}, estimated decrease}} \\"]
    for n in sorted(mf, key=int):
        m = mf[n]
        body.append(" & ".join([rf"$N_n={m['N_n']}$", fmt(m["Delta_dec_median"]), fmt(m["Delta_bar"]),
                                fmt(m["deployed"]["Delta_omega"]), fmt(m["radius"]),
                                fmt(m["deployed"]["Delta_min_with_plugin"]), fmt(m["deployed"]["limit"]),
                                f"{m['failed']}/{m['n_runs']}"]) + r" \\")
    lines = [r"{\footnotesize", r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccccc@{}}", r"\toprule",
             " & ".join(head) + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular*}}"]
    with open(os.path.join(figdir, "table_toy_results.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
