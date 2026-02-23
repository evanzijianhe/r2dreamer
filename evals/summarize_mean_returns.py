#!/usr/bin/env python3
"""Summarize mean episodic returns across seeds for all DMC tasks.

Reads TensorBoard event files from logdir/ppo and prints a table showing
the mean ± std of the final episodic return for each task across seeds.

Example:
    python evals/summarize_mean_returns.py
    python evals/summarize_mean_returns.py --series ppo=logdir/ppo --series-tag ppo=charts/episodic_return
    python evals/summarize_mean_returns.py --series ppo=logdir/ppo --series dreamer=logdir \
        --series-tag ppo=charts/episodic_return --series-tag dreamer=episode/score
    python evals/summarize_mean_returns.py --csv-out logdir/plots/summary.csv
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

TASKS = (
    "acrobot_swingup",
    "ball_in_cup_catch",
    "cartpole_balance",
    "cartpole_balance_sparse",
    "cartpole_swingup",
    "cartpole_swingup_sparse",
    "cheetah_run",
    "finger_spin",
    "finger_turn_easy",
    "finger_turn_hard",
    "hopper_hop",
    "hopper_stand",
    "pendulum_swingup",
    "quadruped_run",
    "quadruped_walk",
    "reacher_easy",
    "reacher_hard",
    "walker_run",
    "walker_stand",
    "walker_walk",
)


@dataclass
class Curve:
    run_id: str
    steps: np.ndarray
    values: np.ndarray
    tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize mean episodic returns across seeds for all DMC tasks."
    )
    parser.add_argument(
        "--series",
        action="append",
        default=None,
        help="Series definition in NAME=PATH form. Repeat for multiple methods. "
             "Default: ppo=logdir/ppo",
    )
    parser.add_argument(
        "--series-tag",
        action="append",
        default=None,
        help="Per-series tags in NAME=tag1,tag2 form. "
             "Default: ppo=charts/episodic_return",
    )
    parser.add_argument(
        "--tail-frac",
        type=float,
        default=0.1,
        help="Fraction of the tail of training to average over for the final score "
             "(default 0.1 = last 10%%).",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="",
        help="Optional CSV output path for the summary table.",
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


def infer_run_id(event_path: Path) -> str:
    run_dir = event_path.parent.name
    m = re.search(r"seed(\d+)", run_dir)
    if m:
        return f"{run_dir.split('__')[0]}_seed{m.group(1)}"
    parent = event_path.parent.parent.name if len(event_path.parents) > 1 else run_dir
    m = re.search(r"_([0-9]+)$", parent)
    if m:
        return f"{parent.rsplit('_', 1)[0]}_seed{m.group(1)}"
    return run_dir


def load_curve(event_path: Path, tag_candidates: Sequence[str]) -> Optional[Curve]:
    accumulator = event_accumulator.EventAccumulator(
        str(event_path), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    chosen_tag = next((tag for tag in tag_candidates if tag in available), None)
    if chosen_tag is None:
        return None

    records = accumulator.Scalars(chosen_tag)
    if not records:
        return None

    steps, values = dedup_by_step(
        (rec.step for rec in records), (rec.value for rec in records)
    )
    if len(steps) < 2:
        return None
    return Curve(run_id=infer_run_id(event_path), steps=steps, values=values, tag=chosen_tag)


def tail_mean(curve: Curve, tail_frac: float) -> float:
    """Return the mean value over the last `tail_frac` fraction of steps."""
    cutoff = curve.steps[-1] - tail_frac * (curve.steps[-1] - curve.steps[0])
    mask = curve.steps >= cutoff
    return float(np.mean(curve.values[mask]))


def main() -> None:
    args = parse_args()

    # Defaults
    series_raw = args.series if args.series else ["ppo=logdir/ppo"]
    series_tag_raw = args.series_tag if args.series_tag else ["ppo=charts/episodic_return"]

    series_defs = parse_series(series_raw)
    series_tag_map = parse_series_tags(series_tag_raw)
    series_names = [name for name, _ in series_defs]

    if not (0.0 < args.tail_frac <= 1.0):
        raise ValueError("--tail-frac must be in (0, 1].")

    # ── Collect results ──────────────────────────────────────────────
    # results[task][series_name] = (mean, std, n_seeds)
    results: Dict[str, Dict[str, Tuple[float, float, int]]] = {}

    for task in TASKS:
        results[task] = {}
        for name, root in series_defs:
            event_files = find_event_files(root, task)
            per_series_tags = series_tag_map.get(name, DEFAULT_TAG_CANDIDATES)
            curves: List[Curve] = []
            for event_path in event_files:
                curve = load_curve(event_path, tag_candidates=per_series_tags)
                if curve is not None:
                    curves.append(curve)
            if not curves:
                results[task][name] = (float("nan"), float("nan"), 0)
                continue

            seed_means = np.array([tail_mean(c, args.tail_frac) for c in curves])
            results[task][name] = (
                float(np.mean(seed_means)),
                float(np.std(seed_means)),
                len(curves),
            )

    # ── Print table ──────────────────────────────────────────────────
    task_col_w = max(len(t) for t in TASKS) + 2
    series_col_w = 28  # "mean ± std (n=X)"

    # Header
    header = f"{'Task':<{task_col_w}}"
    for name in series_names:
        header += f"  {name:^{series_col_w}}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    overall_means: Dict[str, List[float]] = {name: [] for name in series_names}

    for task in TASKS:
        row = f"{task:<{task_col_w}}"
        for name in series_names:
            mean, std, n = results[task].get(name, (float("nan"), float("nan"), 0))
            if n == 0:
                cell = "no data"
            else:
                cell = f"{mean:8.1f} ± {std:6.1f}  (n={n})"
                overall_means[name].append(mean)
            row += f"  {cell:^{series_col_w}}"
        print(row)

    print(sep)

    # Overall mean row
    row = f"{'MEAN':<{task_col_w}}"
    for name in series_names:
        vals = overall_means[name]
        if vals:
            cell = f"{np.mean(vals):8.1f}"
        else:
            cell = "—"
        row += f"  {cell:^{series_col_w}}"
    print(row)
    print(sep)

    # ── Optional CSV ─────────────────────────────────────────────────
    if args.csv_out:
        csv_path = Path(args.csv_out).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("task," + ",".join(
                f"{n}_mean,{n}_std,{n}_n" for n in series_names
            ) + "\n")
            for task in TASKS:
                parts = [task]
                for name in series_names:
                    mean, std, n = results[task].get(name, (float("nan"), float("nan"), 0))
                    parts.extend([f"{mean:.2f}", f"{std:.2f}", str(n)])
                f.write(",".join(parts) + "\n")
        print(f"\n[info] saved csv to {csv_path}")


if __name__ == "__main__":
    main()
