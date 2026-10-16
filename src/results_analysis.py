import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from get_xtiny_config import knee_index_tv_slopes

CFGS = [
    "2d_tiny1", "2d_tiny2", "2d_tiny4", "2d_tiny8", "2d_tiny16",
    "2d_tiny32", "2d_tiny64", "2d_tiny128", "2d_tiny256", "2d",
]
DATASETS = [
    ("Dataset301_busbra", "BUS-BRA"),
    ("Dataset302_EchoNet-Dynamic", "EchoNet-Dynamic"),
    ("Dataset300_isic2018", "ISIC2018"),
    ("Dataset309_FIVES", "FIVES"),
    ("Dataset137_BraTS2021", "BraTS2021"),
    ("Dataset306_ACDC", "ACDC"),
    ("Dataset304_BUSI", "BUSI"),
    ("Dataset308_PROMISE12", "PROMISE12"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="data/nnUNet_results/nnunet")
    p.add_argument("--trainer", type=str, default="nnUNetTrainer_100epochs")
    p.add_argument("--plans", type=str, default="nnUNetPlans")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--nas_dir", type=str, default="results/nas_metrics")
    p.add_argument("--seed", type=int, default=369)
    p.add_argument("--out", type=str, default="results/analysis/reduced_unet_normalized_dice_all_datasets.pdf")
    return p.parse_args()


def read_params(model_dir):
    path = model_dir / "model_analysis.json"
    if not path.exists():
        return np.nan
    return float(json.load(open(path))["parameters"]["total"])


def read_test_dice(model_dir, fold):
    path = model_dir / f"fold_{fold}" / "test" / "results.csv"
    if not path.exists():
        return np.nan
    df = pd.read_csv(path)
    return float(df.iloc[-1]["dice"]) if not df.empty else np.nan


def read_val_dice(model_dir, fold):
    path = model_dir / f"fold_{fold}" / "validation" / "summary.json"
    if not path.exists():
        return np.nan
    dice = json.load(open(path)).get("foreground_mean", {}).get("Dice", np.nan)
    return 100.0 * float(dice) if pd.notna(dice) else np.nan


def dataset_curve(results_dir, dataset, trainer, plans, fold):
    dirs = [results_dir / dataset / f"{trainer}__{plans}__{cfg}" for cfg in CFGS]
    params = np.array([read_params(d) for d in dirs])
    test = np.array([read_test_dice(d, fold) for d in dirs])
    # Test and validation Dice are not comparable, so one source is used for the whole curve.
    if np.isfinite(test).all():
        return params, test, "test"
    return params, np.array([read_val_dice(d, fold) for d in dirs]), "validation"


def normalized_jacobian(nas_dir, dataset, seed):
    df = pd.read_csv(nas_dir / f"{dataset}_batch_jacobian_Kall.csv")
    df = df[df["seed"] == seed]
    jac = np.array([df.loc[df["cfg"] == cfg, "jacobian"].astype(float).mean() for cfg in CFGS])
    return (jac - np.nanmin(jac)) / (np.nanmax(jac) - np.nanmin(jac))


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    fig, ax = plt.subplots(figsize=(8, 5.8))
    ax_jac = ax.twinx()
    for dataset, label in DATASETS:
        params, dice, source = dataset_curve(results_dir, dataset, args.trainer, args.plans, args.fold)
        dice_2d = dice[CFGS.index("2d")]
        if not np.isfinite(dice_2d) or dice_2d <= 0:
            print(f"warning: no 2d Dice for {dataset}, skipped")
            continue
        valid = np.isfinite(params) & (params > 0) & np.isfinite(dice)
        line, = ax.plot(params[valid], dice[valid] / dice_2d, marker="o", markersize=4, linewidth=1.2, label=label)
        print(f"{label}: {source} Dice")

        jac_norm = normalized_jacobian(Path(args.nas_dir), dataset, args.seed)
        jac = np.concatenate([[np.nan], np.abs(np.diff(jac_norm))])
        valid_jac = np.isfinite(params) & (params > 0) & np.isfinite(jac)
        ax_jac.plot(params[valid_jac], jac[valid_jac], marker="s", markersize=3, linewidth=1.0, linestyle="dashed",
                    alpha=0.7, color=line.get_color())

        i = knee_index_tv_slopes(params, jac_norm)
        if i is not None and valid[i]:
            print(f"{label}: XTiny pick {CFGS[i]}")
            ax.scatter(params[i], dice[i] / dice_2d, marker="*", s=180, color=line.get_color(),
                       edgecolors="black", linewidths=0.8, zorder=5)

    ax.axhline(1.0, color="gray", linestyle="dashed", linewidth=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("# Parameters")
    ax.set_ylabel("Dice / Dice of nnU-Net (solid)")
    ax_jac.set_ylabel("Change in normalized Jacobian (dashed)")
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_jac.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(which="major", alpha=0.25)
    ax.grid(which="minor", axis="x", alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="*", linestyle="none", markersize=12,
                              markerfacecolor="white", markeredgecolor="black"))
    labels.append("XTiny-UNet config")
    ax.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
