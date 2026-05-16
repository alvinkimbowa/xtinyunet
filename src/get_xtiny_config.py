#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_dataset_name(dataset_id, nnunet_raw):
    dataset_id = str(dataset_id).zfill(3)
    matches = [
        path.name for path in nnunet_raw.iterdir()
        if path.is_dir() and path.name.startswith(f"Dataset{dataset_id}")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Found {len(matches)} datasets with id {dataset_id}, expected 1")
    return matches[0]


def get_scores_path(dataset_id, batch_size, seed, nnunet_raw, nas_dir):
    dataset_name = get_dataset_name(dataset_id, nnunet_raw)
    return nas_dir / f"{dataset_name}_metrics_b{batch_size}_seed{seed}.csv"


def config_cap(config_name):
    match = re.fullmatch(r"2d_xtiny(\d+)", config_name)
    return int(match.group(1)) if match else None


def normalize(values):
    valid = [v for v in values if math.isfinite(v)]
    if not valid:
        return [math.nan] * len(values)

    min_value = min(valid)
    max_value = max(valid)
    if max_value == min_value:
        return [0.0 if math.isfinite(v) else math.nan for v in values]

    return [
        (v - min_value) / (max_value - min_value) if math.isfinite(v) else math.nan
        for v in values
    ]


def get_metric_curve(rows, metric):
    scores = {}
    params = {}
    for row in rows:
        cfg = row["cfg"]
        cap = config_cap(cfg)
        if cap is None:
            continue
        value = float(row[metric])
        if math.isfinite(value):
            scores[cfg] = value
            params[cfg] = float(row["params"])

    configs = sorted(scores, key=lambda c: config_cap(c))
    values = normalize([scores[cfg] for cfg in configs])
    param_values = [params[cfg] for cfg in configs]
    return configs, param_values, values


def relative_change(values):
    return [math.nan] + [abs(values[i] - values[i - 1]) for i in range(1, len(values))]


def knee_index_tv_slopes(x, y, drop=0, include_hinge_in_both_lines=False):
    x0 = np.asarray(x, dtype=float)
    y0 = np.asarray(y, dtype=float)

    if drop > 0:
        x0 = x0[drop:]
        y0 = y0[drop:]
        base_idx = np.arange(len(x), dtype=int)[drop:]
    else:
        base_idx = np.arange(len(x), dtype=int)

    valid = np.isfinite(x0) & (x0 > 0) & np.isfinite(y0)
    if valid.sum() < 5:
        return None

    xlog = np.log10(x0[valid])
    yv = y0[valid]
    idxv = base_idx[valid]

    s = np.diff(yv) / np.diff(xlog)
    if len(s) < 2:
        return None

    r = np.abs(np.diff(s))
    n = len(xlog)
    best_k_local = None
    best_score = -np.inf

    for k_local in range(2, n - 2):
        if include_hinge_in_both_lines:
            left = r[:k_local]
            right = r[k_local - 1:]
        else:
            left = r[: k_local - 1]
            right = r[k_local - 1:]

        if len(left) == 0 or len(right) == 0:
            continue

        L = np.mean(left)
        R = np.mean(right)
        score = L - R
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_k_local = k_local

    if best_k_local is None:
        return None

    return int(idxv[best_k_local])


def select_xtiny_config(configs, params, values):
    selected_idx = knee_index_tv_slopes(params, relative_change(values))
    if selected_idx is None:
        raise RuntimeError("Could not select XTiny config from sensitivity curve")
    return configs[selected_idx]


def plot_difference_curve(configs, params, values, selected_config, out_path):
    diffs = relative_change(values)
    selected_idx = configs.index(selected_config)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(params, diffs, marker="o", linewidth=1.2, color="black")
    ax.scatter(
        params[selected_idx],
        diffs[selected_idx],
        marker="*",
        s=130,
        color="red",
        zorder=5,
        label=selected_config,
    )
    ax.set_xscale("log")
    ax.set_xlabel("# Parameters")
    ax.set_ylabel("Difference in normalized sensitivity")
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--batch_size", required=True)
    parser.add_argument("--seed", type=int, default=369)
    parser.add_argument("--metric", default="jacobian")
    args = parser.parse_args()

    scores_path = get_scores_path(
        args.dataset_id,
        args.batch_size,
        args.seed,
        Path(os.environ["nnUNet_raw"]),
        Path("results/nas_metrics"),
    )

    with scores_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    configs, params, values = get_metric_curve(rows, args.metric)
    selected_config = select_xtiny_config(configs, params, values)
    plot_difference_curve(
        configs,
        params,
        values,
        selected_config,
        scores_path.with_suffix(".png"),
    )
    print(selected_config)


if __name__ == "__main__":
    main()
