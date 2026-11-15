#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from generate_candidate_configs import add_half_encoder_config, add_xtiny_configs


def get_dataset_name(dataset_id, nnunet_raw):
    dataset_id = str(dataset_id).zfill(3)
    matches = [
        path.name for path in nnunet_raw.iterdir()
        if path.is_dir() and path.name.startswith(f"Dataset{dataset_id}")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Found {len(matches)} datasets with id {dataset_id}, expected 1")
    return matches[0]


def get_required_env_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return Path(value)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate, score, and select an XTinyU-Net configuration."
    )
    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--num_samples", default="all")
    parser.add_argument("--fold", default="0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=369)
    parser.add_argument("--metric", default="jacobian")
    parser.add_argument("--out_dir", default="results/nas_metrics")
    return parser


def generate_configs(plans_path):
    with plans_path.open("r") as f:
        plans = json.load(f)

    generated = add_xtiny_configs(plans)

    with plans_path.open("w") as f:
        json.dump(plans, f, indent=4)
        f.write("\n")

    return generated


def score_configs(args):
    from score_net import main as score_main

    score_args = SimpleNamespace(
        train_dataset_id=args.dataset_id,
        plans=args.plans,
        trainer=args.trainer,
        cfg=None,
        fold=args.fold,
        gpu=args.gpu,
        split="Tr",
        split_type="train",
        num_samples=args.num_samples,
        seed=args.seed,
        out_dir=args.out_dir,
        save_batch_jacobian=False,
        ncd_alpha=0.95,
        save_naswot_codes=False,
        save_swap_codes=False,
        save_ncd_swap_codes=False,
        save_ncd_naswot_codes=False,
        metrics=[args.metric],
    )
    score_main(score_args)


def select_config(args):
    from get_xtiny_config import (
        get_metric_curve,
        get_scores_path,
        plot_difference_curve,
        select_xtiny_config,
    )
    import csv

    nnunet_raw = get_required_env_path("nnUNet_raw")
    scores_path = get_scores_path(
        args.dataset_id,
        args.num_samples,
        args.seed,
        nnunet_raw,
        Path(args.out_dir),
    )

    with scores_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    configs, params, values = get_metric_curve(rows, args.metric)
    selected_config = select_xtiny_config(configs, params, values)
    curve_path = scores_path.with_suffix(".png")
    plot_difference_curve(configs, params, values, selected_config, curve_path)
    return selected_config, scores_path, curve_path


def main():
    args = build_arg_parser().parse_args()
    nnunet_raw = get_required_env_path("nnUNet_raw")
    nnunet_preprocessed = get_required_env_path("nnUNet_preprocessed")
    dataset_name = get_dataset_name(args.dataset_id, nnunet_raw)
    plans_path = nnunet_preprocessed / dataset_name / f"{args.plans}.json"

    if not plans_path.exists():
        raise FileNotFoundError(f"Plans file not found: {plans_path}")

    generated = generate_configs(plans_path)
    print(f"Generated configs in {plans_path}: {', '.join(generated)}")

    score_configs(args)

    selected_config, scores_path, curve_path = select_config(args)
    print(f"Analyzed scores: {scores_path}")
    print(f"Saved curve: {curve_path}")
    print(f"XTinyU-Net: {selected_config}")

    with plans_path.open("r") as f:
        plans = json.load(f)
    half_encoder_config = add_half_encoder_config(plans, selected_config)
    with plans_path.open("w") as f:
        json.dump(plans, f, indent=4)
    print(f"XTinyU-Net with half-width encoder: {half_encoder_config}")


if __name__ == "__main__":
    main()
