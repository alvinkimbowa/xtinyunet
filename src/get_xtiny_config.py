#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from pathlib import Path


def get_dataset_name(dataset_id, nnunet_raw):
    dataset_id = str(dataset_id).zfill(3)
    matches = [
        path.name for path in nnunet_raw.iterdir()
        if path.is_dir() and path.name.startswith(f"Dataset{dataset_id}")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Found {len(matches)} datasets with id {dataset_id}, expected 1")
    return matches[0]


def get_scores_path(dataset_id, batch_size, nnunet_raw, nas_dir):
    dataset_name = get_dataset_name(dataset_id, nnunet_raw)
    return nas_dir / f"{dataset_name}_metrics_b{batch_size}.csv"


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


def select_xtiny_config(rows, metric):
    scores = {}
    for row in rows:
        cfg = row["cfg"]
        cap = config_cap(cfg)
        if cap is None:
            continue
        value = float(row[metric])
        if math.isfinite(value):
            scores[cfg] = value

    configs = sorted(scores, key=lambda c: config_cap(c), reverse=True)
    values = normalize([scores[cfg] for cfg in configs])
    if len(values) < 4:
        raise RuntimeError("Need at least four finite XTiny configs to select a collapse boundary")

    diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    best_k = None
    best_score = -math.inf
    for k in range(2, len(values) - 1):
        tv_left = sum(diffs[:k])
        tv_right = sum(diffs[k:])
        score = tv_right - tv_left
        if score > best_score:
            best_score = score
            best_k = k

    return configs[best_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--batch_size", required=True)
    parser.add_argument("--metric", default="jacobian")
    args = parser.parse_args()

    scores_path = get_scores_path(
        args.dataset_id,
        args.batch_size,
        Path(os.environ["nnUNet_raw"]),
        Path("results/nas_metrics"),
    )

    with scores_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    print(select_xtiny_config(rows, args.metric))


if __name__ == "__main__":
    main()
