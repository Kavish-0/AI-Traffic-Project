"""Static evaluation figures (PNG) for experiment reports.

Colours: validated reference categorical palette, assigned to models in a FIXED
order (a model keeps its colour in every figure); actual values in primary ink.
One y-axis per panel; hairline grid; legend always present for >1 series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_ORDER = ["persistence", "seasonal_naive", "linear", "random_forest", "xgboost", "lstm", "gru", "transformer"]


def model_colors(models: list[str]) -> dict[str, str]:
    """Stable colour per model: canonical models by MODEL_ORDER slot, variants by base name."""
    out: dict[str, str] = {}
    extra = 0
    for m in models:
        base = next((b for b in MODEL_ORDER if m == b or m.startswith(b + "_")), None)
        if base is not None:
            out[m] = SERIES[MODEL_ORDER.index(base)]
        else:
            out[m] = SERIES[(len(MODEL_ORDER) + extra) % len(SERIES)]
            extra += 1
    return out


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def rmse_by_horizon(long: pd.DataFrame, target: str, unit: str, path: Path, title: str) -> Path:
    d = long[long["target"] == target]
    models = list(dict.fromkeys(d["model"]))
    colors = model_colors(models)
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=SURFACE)
    _style(ax)
    for m in models:
        r = d[d["model"] == m].sort_values("horizon_min")
        dashed = m in ("persistence", "seasonal_naive")
        ax.plot(r["horizon_min"], r["RMSE"], color=colors[m], linewidth=2, marker="o", markersize=6,
                linestyle="--" if dashed else "-", label=m)
    ax.set_xticks(sorted(d["horizon_min"].unique()))
    ax.set_xlabel("forecast horizon (min)")
    ax.set_ylabel(f"RMSE ({unit})")
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2, labelcolor=INK_2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def prediction_vs_actual(times: pd.DatetimeIndex, actual: np.ndarray, preds: dict[str, np.ndarray],
                         horizons_min: list[int], target: str, unit: str, tz: str, path: Path, title: str) -> Path:
    """actual / preds: [T, H] aligned to *target* times of one series; one panel per horizon."""
    colors = model_colors(list(preds))
    fig, axes = plt.subplots(len(horizons_min), 1, figsize=(10, 2.5 * len(horizons_min)), sharex=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    local = times.tz_convert(tz)
    for j, (ax, h) in enumerate(zip(axes, horizons_min, strict=True)):
        _style(ax)
        tt = local + pd.Timedelta(minutes=h)
        ax.plot(tt, actual[:, j], color=INK, linewidth=1.4, label="actual")
        for m, p in preds.items():
            ax.plot(tt, p[:, j], color=colors[m], linewidth=1.4, alpha=0.9, label=f"{m} forecast")
        ax.set_ylabel(f"{unit}")
        ax.set_title(f"+{h} min", color=INK_2, fontsize=9, loc="left")
    axes[0].legend(frameon=False, fontsize=8, ncol=len(preds) + 1, labelcolor=INK_2, loc="upper right")
    axes[-1].set_xlabel(f"target time ({tz})")
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return path


def _save(fig: Figure, path: Path, dpi: int = 120) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def error_profile(times: pd.DatetimeIndex, actual: np.ndarray, preds: dict[str, np.ndarray], horizons_min: list[int],
                  unit: str, tz: str, path: Path, title: str, by: str = "time", freq: str = "1D") -> Path:
    """MAE of every test window (all series) against the TARGET time, one panel per horizon.

    ``by="time"``: binned by ``freq`` over the test period (error over time / drift);
    ``by="hour"``: by local hour of day (where in the daily cycle the error sits).
    actual / preds: [N, H]; NaN targets (missing observations) are skipped.
    """
    colors = model_colors(list(preds))
    fig, axes = plt.subplots(len(horizons_min), 1, figsize=(10, 2.3 * len(horizons_min)), sharex=True, sharey=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    local = times.tz_convert(tz)
    for j, (ax, h) in enumerate(zip(axes, horizons_min, strict=True)):
        _style(ax)
        tt = local + pd.Timedelta(minutes=h)
        key = tt.hour if by == "hour" else tt.floor(freq)
        for m, p in preds.items():
            err = pd.Series(np.abs(p[:, j] - actual[:, j]), index=key).groupby(level=0).mean()
            base = m in ("persistence", "seasonal_naive")
            ax.plot(err.index, err.to_numpy(), color=colors[m], linewidth=1.4 if base else 2,
                    linestyle="--" if base else "-", marker="o" if len(err) <= 40 else None, markersize=4, label=m)
        ax.set_ylabel(f"MAE ({unit})")
        ax.set_title(f"+{h} min", color=INK_2, fontsize=9, loc="left")
    axes[0].legend(frameon=False, fontsize=8, ncol=len(preds), labelcolor=INK_2, loc="upper right")
    axes[-1].set_xlabel(f"hour of day of the target ({tz})" if by == "hour" else f"target time ({tz}), {freq} bins")
    if by == "hour":
        axes[-1].set_xticks(range(0, 24, 3))
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, path)


def score_timeline(t: pd.DatetimeIndex, scores: dict[str, tuple[np.ndarray, float]],
                   events: list[tuple[pd.Timestamp, pd.Timestamp, bool]], tz: str, path: Path, title: str) -> Path:
    """One panel per detector: score, alarm threshold, and the ground-truth incident periods."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(len(scores), 1, figsize=(10, 1.9 * len(scores) + 0.6), sharex=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    local = t.tz_convert(tz)
    for i, (ax, (name, (sc, thr))) in enumerate(zip(axes, scores.items(), strict=True)):
        _style(ax)
        for s, e, impactful in events:
            ax.axvspan(s.tz_convert(tz), e.tz_convert(tz), color=INK_2 if impactful else MUTED,
                       alpha=0.22 if impactful else 0.1, linewidth=0)
        ax.plot(local, sc, color=SERIES[i % len(SERIES)], linewidth=1.2)
        ax.axhline(thr, color=INK_2, linewidth=1, linestyle="--")
        ax.set_title(name, color=INK_2, fontsize=9, loc="left")
        ax.set_ylabel("score")
    handles = [Patch(color=INK_2, alpha=0.22, label="incident (traffic impact)"),
               Patch(color=MUTED, alpha=0.1, label="incident (no traffic impact)"),
               Line2D([], [], color=INK_2, linestyle="--", label="alarm threshold")]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, ncol=3, labelcolor=INK_2, loc="upper right")
    axes[-1].set_xlabel(f"time ({tz})")
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, path)


def pr_curves(y: np.ndarray, scores: dict[str, np.ndarray], path: Path, title: str) -> Path:
    """Precision-recall per detector (colour = fixed detector order) and the random-score base rate."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    _style(ax)
    for i, (name, sc) in enumerate(scores.items()):
        p, r, _ = precision_recall_curve(y, sc)
        ax.plot(r, p, color=SERIES[i % len(SERIES)], linewidth=2,
                label=f"{name} (AP {average_precision_score(y, sc):.3f})")
    ax.axhline(float(np.mean(y)), color=INK_2, linewidth=1, linestyle="--", label=f"base rate {np.mean(y):.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    return _save(fig, path, 130)


def confusion(cm: np.ndarray, labels: list[str], path: Path, title: str, xlabel: str = "predicted",
              ylabel: str = "actual") -> Path:
    """Counts + row share; one sequential hue (surface -> blue), text ink flips on dark cells."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE, "#9ec5f0", SERIES[0], "#0d3a70"])
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.25 * n + 2.2, 1.1 * n + 1.6), facecolor=SURFACE)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{int(v)}\n{norm[r, c]:.0%}", ha="center", va="center", fontsize=8,
                color=SURFACE if norm[r, c] > 0.55 else INK)
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    return _save(fig, path, 130)


def grouped_bars(df: pd.DataFrame, category: str, group: str, value: str, path: Path, title: str, ylabel: str,
                 fmt: str = "{:.2f}", ylim: tuple[float, float] | None = None) -> Path:
    """x = category, one bar per group (fixed colour order), value labelled above each bar."""
    cats = list(dict.fromkeys(df[category]))
    groups = list(dict.fromkeys(df[group]))
    width = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * len(cats) + 1.6 * len(groups) * 0.4), 4.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(False, axis="x")
    for i, g in enumerate(groups):
        d = df[df[group] == g].set_index(category).reindex(cats)
        x = np.arange(len(cats)) + (i - (len(groups) - 1) / 2) * width
        bars = ax.bar(x, d[value].to_numpy(), width * 0.92, color=SERIES[i % len(SERIES)], label=str(g))
        for b, v in zip(bars, d[value].to_numpy(), strict=True):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom",
                        fontsize=7, color=INK_2)
    ax.set_xticks(range(len(cats)), cats)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=min(len(groups), 3))
    fig.tight_layout()
    return _save(fig, path, 130)
