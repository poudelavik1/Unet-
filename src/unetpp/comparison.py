'Comparison of finished training runs: the optimiser and loss study.'

from __future__ import annotations
from .common import Path, csv, dataclass, json, math, np
from .losses import LOSS_TERMS, loss_label, loss_terms
from .plots import BLUE_RAMP, PLOT_INK, PLOT_STYLE, SERIES_COLOURS, read_history

# --------------------------------------------------------------------------- #
# Runs compared side by side: every figure colours a run by its optimiser and
# places it by its loss, so the same colour means the same optimiser throughout.
# --------------------------------------------------------------------------- #
OPTIMIZER_LABELS = {"adamw": "AdamW", "adam": "Adam"}
OPTIMIZER_COLOURS = {"adamw": SERIES_COLOURS[0], "adam": SERIES_COLOURS[1]}
# Test metrics shown in the figures, as (key in test_metrics.json, label).
FIGURE_METRICS = (("tol_f1", "Tolerant F1 (2 px)"), ("f1", "F1"), ("iou", "IoU"))
SUMMARY_METRICS = ("precision", "recall", "f1", "iou", "tol_precision", "tol_recall", "tol_f1")


@dataclass(eq=False)  # runs are compared by identity: their history holds arrays
class Run:
    """What the comparison reads from one run directory."""
    path: Path
    optimizer: str
    terms: tuple
    best_epoch: int
    test: dict              # the block of test_metrics.json for the chosen mode
    history: dict           # training_history.csv as {column: values}

    @property
    def loss(self) -> str:
        return loss_label(self.terms)

    @property
    def label(self) -> str:
        return f"{OPTIMIZER_LABELS.get(self.optimizer, self.optimizer)}, {self.loss}"


def run_terms(configuration: dict) -> tuple:
    """The loss terms of a run. Runs before the term list stored the name of their loss:
    "hairline" (BCE + Tversky + clDice) or "focal" (focal + clDice)."""
    loss = configuration.get("loss", "hairline")
    if loss == "focal":
        return ("focal", "cldice")
    return loss_terms(loss)


def run_directories(paths) -> list[Path]:
    """Run directories among `paths`: a path holding configuration.json is a run, any other
    directory is searched one level down (the folder of a whole study)."""
    found = []
    for path in map(Path, paths):
        if (path / "configuration.json").is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(child for child in path.iterdir() if (child / "configuration.json").is_file()))
    return found


def load_run(path: Path, mode: str = "tta") -> Run | None:
    """The run in `path`, or None while it has no test metrics (still training, or stopped)."""
    if not (path / "test_metrics.json").is_file():
        return None
    configuration = json.loads((path / "configuration.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "test_metrics.json").read_text(encoding="utf-8"))
    test = metrics.get(mode) or metrics.get("no_tta")
    return Run(path, configuration.get("optimizer", "adamw"), run_terms(configuration),
               int(metrics.get("best_epoch", 0)), test, read_history(path / "training_history.csv"))


def loss_order(terms: tuple) -> tuple:
    """Single terms first, in LOSS_TERMS order, then the combinations by their number of terms."""
    return len(terms), [LOSS_TERMS.index(term) for term in terms]


def write_summary(runs: list[Run], folder: Path) -> list[str]:
    """comparison_summary.csv (every metric, and the tolerant F1 of every source) and a Markdown table."""
    sources = sorted({source for run in runs for source in run.test.get("per_source", {})})
    header = ["run", "optimizer", "loss", "best_epoch", "threshold", *SUMMARY_METRICS,
              "tol_f1_source_thresholds", *(f"tol_f1 {source}" for source in sources)]
    rows = []
    for run in runs:
        own = run.test.get("with_source_thresholds", {})
        rows.append([run.path.name, OPTIMIZER_LABELS.get(run.optimizer, run.optimizer), run.loss, run.best_epoch,
                     f"{run.test['threshold']:.3f}", *(f"{run.test[name]:.4f}" for name in SUMMARY_METRICS),
                     f"{own['tol_f1']:.4f}" if "tol_f1" in own else "",
                     *(f"{run.test['per_source'][source]['tol_f1']:.4f}" if source in run.test.get("per_source", {})
                       else "" for source in sources)])
    with (folder / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([header, *rows])
    best = max(runs, key=lambda run: run.test["tol_f1"])
    lines = ["| Optimiser | Loss | Precision | Recall | F1 | IoU | Tolerant F1 | Best epoch |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in sorted(runs, key=lambda run: -run.test["tol_f1"]):
        mark = "**" if run is best else ""
        lines.append(f"| {OPTIMIZER_LABELS.get(run.optimizer, run.optimizer)} | {run.loss} | "
                     + " | ".join(f"{run.test[name]:.3f}" for name in ("precision", "recall", "f1", "iou"))
                     + f" | {mark}{run.test['tol_f1']:.3f}{mark} | {run.best_epoch} |")
    (folder / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ["comparison_summary.csv", "comparison_summary.md"]


def save_comparison_figures(runs: list[Run], folder: Path, mode: str) -> list[str]:
    """The figures of the study, in the style of plots.save_plots(); returns the file names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    ink, written = PLOT_INK, []
    losses = sorted({run.terms for run in runs}, key=loss_order)
    optimizers = [name for name in OPTIMIZER_LABELS if any(run.optimizer == name for run in runs)]
    optimizers += sorted({run.optimizer for run in runs} - set(optimizers))
    colour = {name: OPTIMIZER_COLOURS.get(name, ink["secondary"]) for name in optimizers}
    by_cell = {(run.optimizer, run.terms): run for run in runs}
    tta_note = "flip TTA" if mode == "tta" else "no TTA"

    def finish(figure, name):
        figure.savefig(folder / name)
        plt.close(figure)
        written.append(name)

    def legend_handles():
        return [Line2D([], [], marker="o", linestyle="", markersize=8, color=colour[name],
                       markeredgecolor="white", markeredgewidth=1.5, label=OPTIMIZER_LABELS.get(name, name))
                for name in optimizers]

    with plt.rc_context(PLOT_STYLE):
        # 1. Test metrics: one row per loss, one dot per optimiser. Dots, not bars, so the axis can
        #    start near the values: the runs differ by hundredths.
        height = max(3.2, 0.55 * len(losses) + 1.6)
        figure, axes = plt.subplots(1, len(FIGURE_METRICS), figsize=(4.3 * len(FIGURE_METRICS), height), sharey=True)
        positions = {terms: index for index, terms in enumerate(reversed(losses))}
        offsets = {name: (index - (len(optimizers) - 1) / 2) * 0.16 for index, name in enumerate(optimizers)}
        for axis, (key, title) in zip(axes, FIGURE_METRICS):
            values = [run.test[key] for run in runs]
            for terms in losses:  # a hairline joins the optimisers of one loss
                row = [by_cell[(name, terms)].test[key] for name in optimizers if (name, terms) in by_cell]
                if len(row) > 1:
                    axis.plot([min(row), max(row)], [positions[terms]] * 2, color=ink["grid"], linewidth=2, zorder=1)
            for run in runs:
                axis.plot(run.test[key], positions[run.terms] + offsets[run.optimizer], "o", markersize=9,
                          color=colour[run.optimizer], markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            best = max(runs, key=lambda run: run.test[key])
            axis.annotate(f"{best.test[key]:.3f}", (best.test[key], positions[best.terms] + offsets[best.optimizer]),
                          xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5,
                          color=ink["secondary"])
            span = max(values) - min(values)
            pad = max(0.01, 0.15 * span)
            axis.set_xlim(min(values) - pad, max(values) + 2.2 * pad)
            axis.set_title(title)
            axis.grid(axis="y", visible=False)
            axis.xaxis.set_major_locator(MaxNLocator(5))
        axes[0].set_yticks(list(positions.values()), [loss_label(terms) for terms in positions])
        axes[0].set_ylim(-0.6, len(losses) - 0.4)
        axes[-1].legend(handles=legend_handles(), loc="lower right", bbox_to_anchor=(1.0, 1.08), ncol=len(optimizers),
                        borderaxespad=0, handletextpad=0.3, columnspacing=1.2)
        figure.suptitle(f"Test metrics by loss and optimiser ({tta_note}, at each run's calibrated threshold)",
                        x=0.02, ha="left", fontsize=12, fontweight="semibold")
        figure.subplots_adjust(top=1 - 0.95 / height)  # the same room for the heading at any number of losses
        figure.text(0.02, -0.02, "The best run of each metric is labelled. Strict F1 and IoU count pixels exactly; "
                                 "tolerant F1 accepts a prediction within 2 px of a label.",
                    fontsize=8, color=ink["muted"])
        finish(figure, "comparison_test_metrics.png")

        # 2. Heatmap: optimiser x loss, the tolerant F1 and the F1 of every run in one grid each.
        colour_map = LinearSegmentedColormap.from_list("study_blue", BLUE_RAMP)
        panels = (("tol_f1", "Tolerant F1 (2 px)"), ("f1", "F1"))
        figure, axes = plt.subplots(len(panels), 1, figsize=(max(6.5, 1.55 * len(losses) + 2.2),
                                                             1.15 * len(optimizers) * len(panels) + 1.8))
        for axis, (key, title) in zip(np.atleast_1d(axes), panels):
            grid = np.full((len(optimizers), len(losses)), np.nan)
            for row, name in enumerate(optimizers):
                for column, terms in enumerate(losses):
                    if (name, terms) in by_cell:
                        grid[row, column] = by_cell[(name, terms)].test[key]
            low, high = np.nanmin(grid), np.nanmax(grid)
            image = axis.imshow(grid, cmap=colour_map, vmin=low, vmax=high if high > low else low + 1e-3,
                                aspect="auto")
            for row in range(grid.shape[0]):
                for column in range(grid.shape[1]):
                    value = grid[row, column]
                    if math.isnan(value):
                        axis.text(column, row, "not run", ha="center", va="center", fontsize=9, color=ink["muted"])
                        continue
                    share = (value - low) / (high - low) if high > low else 1.0
                    axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=11,
                              fontweight="semibold" if value == high else "normal",
                              color="white" if share > 0.55 else ink["primary"])
            axis.set_xticks(range(len(losses)), [loss_label(terms) for terms in losses])
            axis.set_yticks(range(len(optimizers)), [OPTIMIZER_LABELS.get(name, name) for name in optimizers])
            axis.set_title(title)
            axis.grid(False)
            axis.set_xticks(np.arange(len(losses) - 1) + 0.5, minor=True)
            axis.set_yticks(np.arange(len(optimizers) - 1) + 0.5, minor=True)
            axis.grid(which="minor", color="white", linewidth=2)  # surface gap between cells
            axis.tick_params(which="both", length=0, labelcolor=ink["secondary"])
            for spine in axis.spines.values():
                spine.set_visible(False)
            bar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
            bar.outline.set_visible(False)
            bar.ax.tick_params(labelsize=8, colors=ink["muted"])
        figure.suptitle(f"Every run of the study on the test set ({tta_note})", x=0.02, ha="left", fontsize=12,
                        fontweight="semibold")
        figure.tight_layout()
        finish(figure, "comparison_heatmap.png")

        # 3. Validation curves: one panel per loss, one line per optimiser.
        with_history = [run for run in runs if "epoch" in run.history and "tol_f1" in run.history]
        if with_history:
            columns = min(3, len(losses))
            rows = math.ceil(len(losses) / columns)
            figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.0 * rows + 0.9), sharey=True,
                                        squeeze=False)
            for axis, terms in zip(axes.flat, losses):
                for name in optimizers:
                    run = by_cell.get((name, terms))
                    if run is None or run not in with_history:
                        continue
                    axis.plot(run.history["epoch"], run.history["tol_f1"], color=colour[name])
                    if run.best_epoch in run.history["epoch"]:
                        index = int(np.where(run.history["epoch"] == run.best_epoch)[0][0])
                        axis.plot(run.best_epoch, run.history["tol_f1"][index], "o", markersize=7, color=colour[name],
                                  markeredgecolor="white", markeredgewidth=1.5, zorder=4)
                axis.set_title(loss_label(terms))
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            for axis in axes.flat[len(losses):]:
                axis.axis("off")
            for axis in axes[:, 0]:
                axis.set_ylabel("Validation tolerant F1")
            for axis in axes[-1]:
                axis.set_xlabel("Epoch")
            axes.flat[min(columns, len(losses)) - 1].legend(
                handles=[Line2D([], [], color=colour[name], label=OPTIMIZER_LABELS.get(name, name))
                         for name in optimizers], loc="lower right", bbox_to_anchor=(1.0, 1.12),
                ncol=len(optimizers), borderaxespad=0)
            figure.suptitle("Validation tolerant F1 per epoch (dots: the epoch kept as best.pt)", x=0.02, ha="left",
                            fontsize=12, fontweight="semibold")
            figure.tight_layout()
            finish(figure, "comparison_validation_curves.png")

        # 4. Precision against recall: what each loss trades. Faint curves join equal tolerant F1.
        figure, axis = plt.subplots(figsize=(7.6, 5.6))
        precision = np.array([run.test["tol_precision"] for run in runs])
        recall = np.array([run.test["tol_recall"] for run in runs])
        span = max(np.ptp(precision), np.ptp(recall), 0.02)
        pad = 0.3 * span
        x_low, x_high = recall.min() - pad, min(1.0, recall.max() + 2.5 * pad)  # room for the labels
        y_low, y_high = precision.min() - pad, min(1.0, precision.max() + pad)
        f1 = lambda p, r: 2 * p * r / (p + r)
        f_low, f_high = f1(y_low, x_low), f1(y_high, x_high)
        step = next(value for value in (0.005, 0.01, 0.02, 0.025, 0.05, 0.1) if (f_high - f_low) / value <= 6)
        grid = np.linspace(x_low, x_high, 400)
        for level in np.arange(math.ceil(f_low / step) * step, f_high, step):
            with np.errstate(divide="ignore", invalid="ignore"):
                curve = level * grid / (2 * grid - level)
            visible = (2 * grid > level) & (curve >= y_low) & (curve <= y_high)
            if visible.sum() < 2:
                continue
            axis.plot(grid[visible], curve[visible], color=ink["grid"], linewidth=1.0, zorder=1)
            last = np.where(visible)[0][-1]
            axis.annotate(f"{level:.3g}", (grid[last], curve[last]), xytext=(2, 2), textcoords="offset points",
                          fontsize=7.5, color=ink["muted"])
        for run in runs:
            axis.plot(run.test["tol_recall"], run.test["tol_precision"], "o", markersize=9, color=colour[run.optimizer],
                      markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        # One label per loss, right of its runs; labels that would touch move down, with a leader line.
        groups = sorted(losses, key=lambda terms: -np.mean([run.test["tol_precision"] for run in runs
                                                            if run.terms == terms]))
        gap, previous = 0.05 * (y_high - y_low), None
        for terms in groups:
            members = [run for run in runs if run.terms == terms]
            x = max(run.test["tol_recall"] for run in members)
            y = float(np.mean([run.test["tol_precision"] for run in members]))
            label_y = y if previous is None else min(y, previous - gap)
            previous = label_y
            moved = abs(label_y - y) > gap / 2
            axis.annotate(loss_label(terms), (x, y), xytext=(x + 0.03 * (x_high - x_low), label_y),
                          textcoords="data", fontsize=8.5, color=ink["secondary"], va="center",
                          arrowprops=dict(arrowstyle="-", color=ink["muted"], linewidth=0.8, shrinkA=0, shrinkB=6)
                          if moved else None)
        axis.set(xlabel="Tolerant recall", ylabel="Tolerant precision", xlim=(x_low, x_high), ylim=(y_low, y_high),
                 title="Tolerant precision and recall")
        axis.legend(handles=legend_handles(), loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(optimizers),
                    borderaxespad=0.2)
        axis.text(0, -0.12, f"Test set, {tta_note}. Each dot is one run; grey curves join points of equal "
                            "tolerant F1.", transform=axis.transAxes, fontsize=8, color=ink["muted"], va="top")
        finish(figure, "comparison_precision_recall.png")
    return written


def compare_runs(paths, output: Path, mode: str = "tta") -> list[str]:
    """Read the runs in `paths` (run directories, or folders of them), write the summary tables and
    figures to `output` and return the file names. Unfinished runs are reported and left out."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs, seen = [], {}
    for path in run_directories(paths):
        run = load_run(path, mode)
        if run is None:
            print(f"Left out, no test metrics yet: {path}")
            continue
        key = (run.optimizer, run.terms)
        if key in seen:
            print(f"Left out, the same optimiser and loss as {seen[key].name}: {path}")
            continue
        seen[key] = path
        runs.append(run)
    if not runs:
        raise SystemExit("No finished run (with test_metrics.json) found in " + ", ".join(map(str, paths)))
    runs.sort(key=lambda run: (loss_order(run.terms), list(OPTIMIZER_LABELS).index(run.optimizer)
                               if run.optimizer in OPTIMIZER_LABELS else 99))
    written = write_summary(runs, output)
    try:
        written += save_comparison_figures(runs, output, mode)
    except ImportError:
        print("matplotlib is not installed, so only the tables were written (python -m pip install matplotlib).")
    for run in runs:
        print(f"{run.label:42s} tolerant F1 {run.test['tol_f1']:.4f}  F1 {run.test['f1']:.4f}  "
              f"IoU {run.test['iou']:.4f}  (best epoch {run.best_epoch}, {run.path.name})")
    return written


__all__ = [
    'OPTIMIZER_LABELS', 'OPTIMIZER_COLOURS', 'FIGURE_METRICS', 'SUMMARY_METRICS', 'Run', 'run_terms',
    'run_directories', 'load_run', 'loss_order', 'write_summary', 'save_comparison_figures', 'compare_runs',
]
