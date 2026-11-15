#!/usr/bin/env python3
import argparse
from copy import deepcopy
import json
from pathlib import Path


def add_xtiny_configs(plans):
    base_config = plans["configurations"]["2d"]
    base_features_per_stage = base_config["architecture"]["arch_kwargs"]["features_per_stage"]
    max_features = base_features_per_stage[-1]

    generated = []
    while max_features >= 1:
        name = f"2d_xtiny{max_features}"
        architecture = deepcopy(base_config["architecture"])
        features_per_stage = [
            min(max_features, value)
            for value in base_features_per_stage
        ]
        architecture["arch_kwargs"]["features_per_stage"] = features_per_stage
        plans["configurations"][name] = {
            "inherits_from": "2d",
            "architecture": architecture,
        }
        generated.append(name)
        max_features //= 2

    return generated


def add_half_encoder_config(plans, cfg):
    """Add `<cfg>_halfenc`: `cfg` with every encoder stage at half width and the decoder unchanged."""
    architecture = deepcopy(plans["configurations"][cfg]["architecture"])
    arch_kwargs = architecture["arch_kwargs"]
    arch_kwargs["features_per_stage_decoder"] = list(arch_kwargs["features_per_stage"])
    arch_kwargs["features_per_stage"] = [max(1, value // 2) for value in arch_kwargs["features_per_stage"]]
    architecture["network_class_name"] = "asym_unet.HalfEncoderUNet"
    name = f"{cfg}_halfenc"
    plans["configurations"][name] = {
        "inherits_from": "2d",
        "architecture": architecture,
    }
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", required=True, type=Path)
    parser.add_argument("--half_encoder", metavar="CFG",
                        help="only add the half-encoder variant of this existing config")
    args = parser.parse_args()

    with args.plans.open("r") as f:
        plans = json.load(f)

    if args.half_encoder:
        generated = [add_half_encoder_config(plans, args.half_encoder)]
    else:
        generated = add_xtiny_configs(plans)

    with args.plans.open("w") as f:
        json.dump(plans, f, indent=4)
        f.write("\n")

    for name in generated:
        print(name)


if __name__ == "__main__":
    main()
