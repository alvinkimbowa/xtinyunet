#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def add_xtiny_configs(plans):
    base_config = plans["configurations"]["2d"]
    base_features_per_stage = base_config["architecture"]["arch_kwargs"]["features_per_stage"]
    max_features = base_features_per_stage[-1]

    generated = []
    while max_features >= 1:
        name = f"2d_xtiny{max_features}"
        features_per_stage = [
            min(max_features, value)
            for value in base_features_per_stage
        ]
        plans["configurations"][name] = {
            "inherits_from": "2d",
            "architecture": {
                "arch_kwargs": {
                    "features_per_stage": features_per_stage,
                }
            },
        }
        generated.append(name)
        max_features //= 2

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", required=True, type=Path)
    args = parser.parse_args()

    with args.plans.open("r") as f:
        plans = json.load(f)

    generated = add_xtiny_configs(plans)

    with args.plans.open("w") as f:
        json.dump(plans, f, indent=4)
        f.write("\n")

    for name in generated:
        print(name)


if __name__ == "__main__":
    main()
