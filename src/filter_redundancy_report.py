"""Figures and a text summary from results/analysis/filter_redundancy/*/*.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results/analysis/filter_redundancy")
TRAINER = "nnUNetTrainer_100epochs"
DATASETS = [
    ("Dataset301_busbra", "BUS-BRA"),
    ("Dataset302_EchoNet-Dynamic", "EchoNet"),
    ("Dataset300_isic2018", "ISIC2018"),
    ("Dataset309_FIVES", "FIVES"),
    ("Dataset137_BraTS2021", "BraTS2021"),
    ("Dataset306_ACDC", "ACDC"),
]
WIDTHS = [4, 8, 16, 32, 64]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
DS_COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
DICE_TOL = 0.01

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def load(ds, cfg, init=False):
    f = ROOT / ds / TRAINER / f"{cfg}{'_init' if init else ''}.json"
    return json.load(open(f)) if f.exists() else None


def block_label(L):
    return f"{'E' if L['part'] == 'enc' else 'D'}{L['level']}.{L['name'][-1]}"


def min_rank(L, base):
    """Smallest tested k whose rank-k projection keeps Dice within DICE_TOL of the unablated model."""
    for k, v in sorted(((int(k), v) for k, v in L["ablation"]["rank_k"].items())):
        if v >= base - DICE_TOL:
            return k
    return L["channels"]


def u_axis(ax, layers):
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([block_label(L) for L in layers], rotation=90, fontsize=6)
    n_enc = sum(L["part"] == "enc" for L in layers)
    ax.axvline(n_enc - 0.5, color=INK2, lw=0.8, ls=":")


def fig_per_layer(cfg_of, out):
    fig, axes = plt.subplots(3, len(DATASETS), figsize=(3.1 * len(DATASETS), 7.4), squeeze=False)
    for j, (ds, label) in enumerate(DATASETS):
        cfg = cfg_of(ds)
        r, r0 = load(ds, cfg), load(ds, cfg, init=True)
        if r is None or "ablation" not in r["layers"][0]:
            continue
        layers, base = r["layers"], r["base_dice"]
        x = np.arange(len(layers))
        drop_zero = np.array([base - np.array(L["ablation"]["zero"]) for L in layers], dtype=object)
        mean_drop = [np.mean(d) for d in drop_zero]
        max_drop = [np.max(d) for d in drop_zero]
        ax = axes[0, j]
        ax.bar(x, mean_drop, width=0.7, color=BLUE, label="mean over filters")
        ax.scatter(x, max_drop, s=14, color=ORANGE, zorder=3, label="worst filter")
        ax.set_title(f"{label} ({cfg.replace('2d_', '')}, Dice {base:.3f})", fontsize=9, color=INK)
        u_axis(ax, layers)
        ax = axes[1, j]
        C = layers[0]["channels"]
        ax.bar(x, [min_rank(L, base) for L in layers], width=0.7, color=AQUA)
        ax.axhline(C, color=INK2, lw=0.8, ls="--")
        ax.set_ylim(0, C + 0.5)
        u_axis(ax, layers)
        ax = axes[2, j]
        ax.plot(x, [L["post"]["k95"] for L in layers], color=BLUE, lw=2, marker="o", ms=4, label="trained")
        if r0:
            ax.plot(x, [L["post"]["k95"] for L in r0["layers"]], color=INK2, lw=1.2, ls="--", label="random init")
        ax.set_ylim(0, C + 0.5)
        u_axis(ax, layers)
    axes[0, 0].set_ylabel("Dice drop when one\nfilter is zeroed")
    axes[1, 0].set_ylabel(f"PCs needed to keep\nDice within {DICE_TOL}")
    axes[2, 0].set_ylabel("PCs for 95% of\nactivation variance")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    axes[2, 0].legend(fontsize=7, loc="lower left")
    fig.text(0.5, 0.005, "conv block along the U (E = encoder, D = decoder, number = resolution level; dotted line = bottleneck→decoder)",
             ha="center", color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_width(out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for (ds, label), col in zip(DATASETS, DS_COLORS):
        k95, k95_0, frac, dice = [], [], [], []
        for w in WIDTHS:
            r, r0 = load(ds, f"2d_tiny{w}"), load(ds, f"2d_tiny{w}", init=True)
            k95.append(np.mean([L["post"]["k95"] for L in r["layers"]]) if r else np.nan)
            k95_0.append(np.mean([L["post"]["k95"] for L in r0["layers"]]) if r0 else np.nan)
            frac.append(np.mean([L["post"]["k95"] / L["channels"] for L in r["layers"]]) if r else np.nan)
            dice.append(r["base_dice"] if r else np.nan)
        axes[0].plot(WIDTHS, k95, color=col, lw=2, marker="o", ms=4, label=label)
        axes[0].plot(WIDTHS, k95_0, color=col, lw=1, ls="--", alpha=0.7)
        axes[1].plot(WIDTHS, frac, color=col, lw=2, marker="o", ms=4, label=label)
        axes[2].plot(WIDTHS, dice, color=col, lw=2, marker="o", ms=4, label=label)
    axes[0].plot(WIDTHS, WIDTHS, color=INK2, lw=0.8, ls=":")
    axes[0].set_ylabel("mean PCs for 95% variance\n(solid trained, dashed init, dotted = width)")
    axes[1].set_ylabel("fraction of filters needed (k95 / width)")
    axes[2].set_ylabel("pooled validation Dice")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(WIDTHS)
        ax.set_xticklabels(WIDTHS)
        ax.set_xlabel("filters per block (tinyN)")
    axes[0].set_yscale("log", base=2)
    axes[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def summary():
    lines = []
    for ds, label in DATASETS:
        for w in WIDTHS:
            r, r0 = load(ds, f"2d_tiny{w}"), load(ds, f"2d_tiny{w}", init=True)
            if r is None:
                continue
            Ls = r["layers"]
            enc = [L for L in Ls if L["part"] == "enc"]
            dec = [L for L in Ls if L["part"] == "dec"]
            row = dict(
                ds=label, w=w, dice=r["base_dice"],
                k95_frac=np.mean([L["post"]["k95"] / w for L in Ls]),
                k95_frac_init=np.mean([L["post"]["k95"] / w for L in r0["layers"]]) if r0 else np.nan,
                absr=np.mean([L["post"]["mean_abs_r"] for L in Ls]),
                absr_init=np.mean([L["post"]["mean_abs_r"] for L in r0["layers"]]) if r0 else np.nan,
                maxr=np.mean([np.mean(L["post"]["max_abs_r_per_channel"]) for L in Ls]),
                flip_pre=sum(L["pre"]["n_pairs_r_lt_m0p8"] for L in Ls),
                same_pre=sum(L["pre"]["n_pairs_r_gt_0p8"] for L in Ls),
                same_post=sum(L["post"]["n_pairs_r_gt_0p8"] for L in Ls),
                wflip=sum(L["weights"]["n_pairs_cos_lt_m0p8"] for L in Ls),
                dead=sum(L["post"]["n_dead"] for L in Ls),
            )
            if "ablation" in Ls[0]:
                base = r["base_dice"]
                def drop(group):
                    return np.mean([np.mean(base - np.array(L["ablation"]["zero"])) for L in group])
                deep = [L for L in enc if L["level"] >= len({l["level"] for l in enc}) - 2]
                row.update(
                    drop_enc=drop(enc), drop_dec=drop(dec), drop_deep_enc=drop(deep),
                    n_harmless=sum(int(np.sum(base - np.array(L["ablation"]["zero"]) < 0.005)) for L in Ls),
                    n_filters=sum(L["channels"] for L in Ls),
                    rank_frac=np.mean([min_rank(L, base) / w for L in Ls]),
                )
            lines.append(row)
    return lines


def export_compact(out):
    """Per-layer numbers for the web report, small enough to inline."""
    data = {}
    for ds, label in DATASETS:
        data[label] = {}
        for w in WIDTHS:
            r, r0 = load(ds, f"2d_tiny{w}"), load(ds, f"2d_tiny{w}", init=True)
            if r is None:
                continue
            base = r["base_dice"]
            layers = []
            for i, L in enumerate(r["layers"]):
                row = dict(
                    label=block_label(L), part=L["part"], level=L["level"], C=L["channels"],
                    k95=L["post"]["k95"], pr=round(L["post"]["participation_ratio"], 2),
                    absr=round(L["post"]["mean_abs_r"], 3),
                    flip=L["pre"]["n_pairs_r_lt_m0p8"], same=L["pre"]["n_pairs_r_gt_0p8"],
                    dead=L["post"]["n_dead"],
                    wcos=round(L["weights"]["mean_abs_cos"], 3),
                )
                if r0:
                    I = r0["layers"][i]
                    row.update(k95_0=I["post"]["k95"], pr_0=round(I["post"]["participation_ratio"], 2),
                               absr_0=round(I["post"]["mean_abs_r"], 3), wcos_0=round(I["weights"]["mean_abs_cos"], 3))
                if "ablation" in L:
                    row["drop"] = [round(base - v, 4) for v in L["ablation"]["zero"]]
                    row["rank"] = {k: round(base - v, 4) for k, v in L["ablation"]["rank_k"].items()}
                    row["kmin"] = min_rank(L, base)
                layers.append(row)
            data[label][w] = dict(dice=round(base, 4), n=r["n_slices"], layers=layers)
    json.dump(data, open(out, "w"), separators=(",", ":"))


def main():
    export_compact(ROOT / "compact.json")
    fig_per_layer(lambda ds: "2d_tiny8", ROOT / "per_layer_tiny8.png")
    fig_per_layer(lambda ds: "2d_tiny32", ROOT / "per_layer_tiny32.png")
    fig_width(ROOT / "width_scaling.png")
    rows = summary()
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(ROOT / "summary.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys) + "\n")
    print(open(ROOT / "summary.csv").read())


if __name__ == "__main__":
    main()
