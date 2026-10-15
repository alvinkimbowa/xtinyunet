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


def get_scores_path(dataset_id, num_samples, seed, nnunet_raw, nas_dir):
    dataset_name = get_dataset_name(dataset_id, nnunet_raw)
    return nas_dir / f"{dataset_name}_metrics_K{num_samples}_seed{seed}.csv"


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


def knee_index_tv_slopes(x, y):
    x0 = np.asarray(x, dtype=float)
    y0 = np.asarray(y, dtype=float)
    base_idx = np.arange(len(x), dtype=int)

    x0 = x0[::-1]
    y0 = y0[::-1]
    base_idx = base_idx[::-1]
    
    # Compute local variation in sensitivity between consecutive configurations
    d = np.abs(np.diff(y0))

    n = len(x)
    best_k_local = None
    best_score = -np.inf

    for k_local in range(1, n - 3):
        pre = d[:k_local + 1]
        post = d[k_local + 1:]

        if len(post) == 0 or len(pre) == 0:
            continue

        pre_tv = np.mean(pre)
        post_tv = np.mean(post)
        score = post_tv - pre_tv
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_k_local = k_local

    if best_k_local is None:
        return None

    return int(base_idx[best_k_local])


def select_xtiny_config(configs, params, values):
    print("configs: ", configs)
    selected_idx = knee_index_tv_slopes(params, values)
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
        label=f"XTinyU-Net: {selected_config}",
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
    parser.add_argument("--num_samples", required=True)
    parser.add_argument("--seed", type=int, default=369)
    parser.add_argument("--metric", default="jacobian")
    args = parser.parse_args()

    scores_path = get_scores_path(
        args.dataset_id,
        args.num_samples,
        args.seed,
        Path(os.environ["nnUNet_raw"]),
        Path("results/nas_metrics"),
    )

    with scores_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    configs, params, values = get_metric_curve(rows, args.metric)
    selected_config = select_xtiny_config(configs, params, values)
    curve_path = scores_path.with_suffix(".png")
    plot_difference_curve(
        configs,
        params,
        values,
        selected_config,
        curve_path,
    )
    print(f"Analyzed scores: {scores_path}")
    print(f"Saved curve: {curve_path}")
    print(f"XTinyU-Net: {selected_config}")


if __name__ == "__main__":
    main()
