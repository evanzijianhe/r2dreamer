#!/usr/bin/env python3
"""Plot training curves from TensorBoard event files.

Example:
    python evals/plot_training_progress.py \
        --task acrobot_swingup \
        --series ppo=logdir/ppo \
        --series r2dreamer=logdir \
        --out logdir/plots/acrobot_compare.png 
    # PPO vs Dreamer on same task
    python evals/plot_training_progress.py \
        --task acrobot_swingup \
        --series ppo=logdir/ppo \
        --series-tag ppo=charts/episodic_return \
        --series dreamer=logdir \
        --series-tag dreamer=episode/score \
        --ema-alpha 0.1 \
        --out logdir/plots/acrobot_ppo_vs_dreamer.png \
        --csv-out logdir/plots/acrobot_ppo_vs_dreamer.csv
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError as exc:
    raise SystemExit(
        "tensorboard is required to read event files. "
        "Install it with: pip install tensorboard"
    ) from exc


DEFAULT_TAG_CANDIDATES = (
    "charts/episodic_return",  # PPO script
    "episode/score",           # Dreamer train score
    "eval/episodic_return",    # PPO eval score
    "episode/eval_score",      # Dreamer eval score
)


@dataclass
class Curve:
    run_id: str
    steps: np.ndarray
    values: np.ndarray
    tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reward-vs-steps from TensorBoard logs.")
    parser.add_argument("--task", type=str, required=True, help="Task substring, e.g. acrobot_swingup")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="Series definition in NAME=PATH form. Repeat for multiple methods.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Scalar tag to read. Repeat for fallbacks. Defaults are PPO/Dreamer reward tags.",
    )
    parser.add_argument(
        "--series-tag",
        action="append",
        default=[],
        help="Per-series tags in NAME=tag1,tag2 form. Overrides --tag for that series.",
    )
    parser.add_argument("--num-grid", type=int, default=400, help="Resampling grid size for averaging.")
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.0,
        help="EMA smoothing alpha in [0,1]. 0 disables smoothing.",
    )
    parser.add_argument(
        "--hide-runs",
        action="store_true",
        help="Hide individual run curves and only show mean/std.",
    )
    parser.add_argument("--title", type=str, default="", help="Custom plot title.")
    parser.add_argument("--out", type=str, default="training_progress.png", help="Output image path.")
    parser.add_argument(
        "--csv-out",
        type=str,
        default="",
        help="Optional CSV output path for aggregated curves.",
    )
    return parser.parse_args()


def parse_series(items: Sequence[str]) -> List[Tuple[str, Path]]:
    series: List[Tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --series '{item}'. Expected NAME=PATH.")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser()
        if not name:
            raise ValueError(f"Invalid --series '{item}'. NAME cannot be empty.")
        if not path.exists():
            raise ValueError(f"Series path does not exist: {path}")
        series.append((name, path))
    return series


def parse_series_tags(items: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    mapping: Dict[str, Tuple[str, ...]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --series-tag '{item}'. Expected NAME=tag1,tag2.")
        name, raw_tags = item.split("=", 1)
        name = name.strip()
        tags = tuple(tag.strip() for tag in raw_tags.split(",") if tag.strip())
        if not name:
            raise ValueError(f"Invalid --series-tag '{item}'. NAME cannot be empty.")
        if not tags:
            raise ValueError(f"Invalid --series-tag '{item}'. At least one tag is required.")
        mapping[name] = tags
    return mapping


def find_event_files(root: Path, task: str) -> List[Path]:
    task = task.strip()
    matches: List[Path] = []
    for event_path in root.rglob("events.out.tfevents*"):
        if task in str(event_path):
            matches.append(event_path)
    return sorted(matches)


def dedup_by_step(steps: Iterable[int], values: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    latest: Dict[int, float] = {}
    for step, value in zip(steps, values):
        latest[int(step)] = float(value)
    dedup_steps = np.array(sorted(latest.keys()), dtype=np.float64)
    dedup_values = np.array([latest[int(s)] for s in dedup_steps], dtype=np.float64)
    return dedup_steps, dedup_values


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return values
    out = np.empty_like(values)
    out[0] = values[0]
    for idx in range(1, len(values)):
        out[idx] = alpha * values[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def infer_run_id(event_path: Path) -> str:
    run_dir = event_path.parent.name
    # PPO run name: ppo_dmc__task__seed0__timestamp
    m = re.search(r"seed(\d+)", run_dir)
    if m:
        return f"{run_dir.split('__')[0]}_seed{m.group(1)}"
    # Generic fallback: try one level up.
    parent = event_path.parent.parent.name if len(event_path.parents) > 1 else run_dir
    m = re.search(r"_([0-9]+)$", parent)
    if m:
        return f"{parent.rsplit('_', 1)[0]}_seed{m.group(1)}"
    return run_dir


def load_curve(event_path: Path, tag_candidates: Sequence[str], ema_alpha: float) -> Optional[Curve]:
    accumulator = event_accumulator.EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    chosen_tag = next((tag for tag in tag_candidates if tag in available), None)
    if chosen_tag is None:
        return None

    records = accumulator.Scalars(chosen_tag)
    if not records:
        return None

    steps, values = dedup_by_step((rec.step for rec in records), (rec.value for rec in records))
    if len(steps) < 2:
        return None
    values = ema(values, ema_alpha)
    return Curve(run_id=infer_run_id(event_path), steps=steps, values=values, tag=chosen_tag)


def aggregate_curves(curves: Sequence[Curve], num_grid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_step = min(curve.steps[0] for curve in curves)
    max_step = max(curve.steps[-1] for curve in curves)
    grid = np.linspace(min_step, max_step, num_grid, dtype=np.float64)

    stacked = []
    for curve in curves:
        y = np.interp(grid, curve.steps, curve.values)
        y[grid < curve.steps[0]] = np.nan
        y[grid > curve.steps[-1]] = np.nan
        stacked.append(y)
    matrix = np.stack(stacked, axis=0)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    return grid, mean, std


def write_csv(csv_path: Path, rows: List[Tuple[str, float, float, float]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("series,step,mean,std\n")
        for series, step, mean, std in rows:
            f.write(f"{series},{step:.6f},{mean:.6f},{std:.6f}\n")


def main() -> None:
    args = parse_args()
    series_defs = parse_series(args.series)
    series_tag_map = parse_series_tags(args.series_tag)
    tag_candidates = tuple(args.tag) if args.tag else DEFAULT_TAG_CANDIDATES
    if not (0.0 <= args.ema_alpha <= 1.0):
        raise ValueError("--ema-alpha must be in [0, 1].")
    if args.num_grid < 10:
        raise ValueError("--num-grid must be >= 10.")

    fig, ax = plt.subplots(figsize=(10, 6))
    csv_rows: List[Tuple[str, float, float, float]] = []

    for name, root in series_defs:
        event_files = find_event_files(root, args.task)
        per_series_tags = series_tag_map.get(name, tag_candidates)
        curves = []
        for event_path in event_files:
            curve = load_curve(event_path, tag_candidates=per_series_tags, ema_alpha=args.ema_alpha)
            if curve is not None:
                curves.append(curve)
        if not curves:
            print(f"[warn] {name}: no matching scalar data for task='{args.task}' under {root}")
            continue

        if not args.hide_runs:
            for curve in curves:
                ax.plot(curve.steps, curve.values, alpha=0.2, linewidth=1.0)

        grid, mean, std = aggregate_curves(curves, num_grid=args.num_grid)
        ax.plot(grid, mean, linewidth=2.5, label=f"{name} (n={len(curves)})")
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2)

        for step, m, s in zip(grid, mean, std):
            csv_rows.append((name, float(step), float(m), float(s)))

        tags_used = sorted({curve.tag for curve in curves})
        print(f"[info] {name}: {len(curves)} runs, tags={tags_used}")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episodic Return")
    ax.set_title(args.title if args.title else f"{args.task}: training progress")
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1100])
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    if ax.has_data():
        ax.legend()
    fig.tight_layout()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"[info] saved plot to {out_path}")

    if args.csv_out:
        csv_path = Path(args.csv_out).expanduser()
        write_csv(csv_path, csv_rows)
        print(f"[info] saved csv to {csv_path}")


if __name__ == "__main__":
    main()
