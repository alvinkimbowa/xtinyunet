"""Filter redundancy analysis of trained (X)TinyU-Net models.

For every conv block (conv -> InstanceNorm -> LeakyReLU) of a trained nnU-Net
PlainConvUNet this measures, on fold-0 validation slices:

- weight space: pairwise cosine similarity of flattened filters and the
  effective rank of the filter matrix,
- activation space: channel correlation before and after the nonlinearity and
  the PCA spectrum of the block output (components needed for 90/95/99% of the
  variance, participation ratio),
- ablation (optional): pooled Dice when one channel of the block output is
  zeroed, and when the whole block output is projected
  onto its top-k principal components.

Results go to results/analysis/filter_redundancy/<dataset>/<trainer>/<cfg>[_init].json.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

RESULTS_ROOT = Path("data/nnUNet_results/nnunet")
PREPROCESSED_ROOT = Path("data/nnUNet_preprocessed")
OUT_ROOT = Path("results/analysis/filter_redundancy")


def deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def build_model(dataset, cfg, trainer, plans, fold, device, trained=True, seed=0):
    """Build the network; tiny configs override only nested arch kwargs, so inheritance is resolved deeply."""
    model_dir = RESULTS_ROOT / dataset / f"{trainer}__{plans}__{cfg}"
    plans_dict = json.load(open(model_dir / "plans.json"))
    confs = plans_dict["configurations"]
    while "inherits_from" in confs[cfg]:
        parent = confs[cfg].pop("inherits_from")
        confs[cfg] = deep_merge(confs[parent], confs[cfg])
    plans_manager = PlansManager(plans_dict)
    dataset_json = json.load(open(model_dir / "dataset.json"))
    cm = plans_manager.get_configuration(cfg)
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_in = len(dataset_json.get("channel_names", dataset_json.get("modality")))
    torch.manual_seed(seed)
    net = get_network_from_plans(
        cm.network_arch_class_name,
        cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import,
        num_in,
        label_manager.num_segmentation_heads,
        allow_init=True,
        deep_supervision=False,
    )
    if trained:
        ckpt = torch.load(model_dir / f"fold_{fold}" / "checkpoint_final.pth", map_location="cpu", weights_only=False)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["network_weights"].items()}
        net.load_state_dict(sd)
    return net.to(device).eval(), label_manager, cm


def total_stride(cm):
    strides = np.array(cm.network_arch_init_kwargs["strides"])
    return np.prod(strides, axis=0).astype(int)


def load_val_slices(dataset, plans, fold, max_slices, seed):
    """Return a list of (image[C,H,W], seg[H,W]) from the fold's validation cases.

    3D cases contribute a few slices each, preferring slices with foreground.
    """
    root = PREPROCESSED_ROOT / dataset
    val_ids = json.load(open(root / "splits_final.json"))[fold]["val"]
    ds = nnUNetDatasetBlosc2(str(root / f"{plans}_2d"), val_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(val_ids)
    probe = ds.load_case(val_ids[0])[0]
    per_case = 1 if probe.shape[1] == 1 else max(1, int(np.ceil(max_slices / len(val_ids))) + 1)
    slices = []
    for cid in val_ids:
        data, seg, _, _ = ds.load_case(cid)
        data, seg = np.asarray(data[:]), np.asarray(seg[:])[0]
        z_all = np.arange(data.shape[1])
        fg = z_all[(seg > 0).reshape(seg.shape[0], -1).any(1)]
        pool = fg if len(fg) >= per_case else z_all
        for z in rng.choice(pool, size=min(per_case, len(pool)), replace=False):
            s = seg[z].astype(np.int64)
            s[s < 0] = 0
            slices.append((data[:, z].astype(np.float32), s))
        if len(slices) >= max_slices:
            break
    return slices[:max_slices]


def make_batches(slices, stride, batch_size, device):
    """Pad slices to a common size divisible by the network stride, keep a validity mask."""
    H = max(s[0].shape[1] for s in slices)
    W = max(s[0].shape[2] for s in slices)
    H, W = int(np.ceil(H / stride[0]) * stride[0]), int(np.ceil(W / stride[1]) * stride[1])
    batches = []
    for i in range(0, len(slices), batch_size):
        chunk = slices[i:i + batch_size]
        x = np.zeros((len(chunk), chunk[0][0].shape[0], H, W), np.float32)
        y = np.zeros((len(chunk), H, W), np.int64)
        m = np.zeros((len(chunk), H, W), bool)
        for j, (img, seg) in enumerate(chunk):
            h, w = seg.shape
            x[j, :, :h, :w], y[j, :h, :w], m[j, :h, :w] = img, seg, True
        batches.append(tuple(torch.from_numpy(a).to(device) for a in (x, y, m)))
    return batches


def conv_blocks(net):
    blocks = [(n, m) for n, m in net.named_modules() if isinstance(m, ConvDropoutNormReLU)]
    enc = [b for b in blocks if b[0].startswith("encoder")]
    dec = [b for b in blocks if b[0].startswith("decoder")]
    n_stages = len(net.encoder.stages)
    meta = []
    for k, (name, _) in enumerate(enc):
        stage = int(name.split(".")[2])
        meta.append(dict(name=name, part="enc", stage=stage, level=stage, order=k))
    for k, (name, _) in enumerate(dec):
        stage = int(name.split(".")[2])
        meta.append(dict(name=name, part="dec", stage=stage, level=n_stages - 2 - stage, order=len(enc) + k))
    return enc + dec, meta


class Dice:
    def __init__(self, label_manager):
        self.lm = label_manager
        self.regions = label_manager.has_regions
        self.classes = label_manager.foreground_regions if self.regions else label_manager.foreground_labels
        self.tp = np.zeros(len(self.classes))
        self.fp = np.zeros(len(self.classes))
        self.fn = np.zeros(len(self.classes))

    def update(self, logits, y, m):
        if self.regions:
            pred = torch.sigmoid(logits) > 0.5
        else:
            pred = logits.argmax(1)
        for i, c in enumerate(self.classes):
            if self.regions:
                p = pred[:, i]
                t = torch.isin(y, torch.tensor(c if isinstance(c, (list, tuple)) else [c], device=y.device))
            else:
                p, t = pred == c, y == c
            p, t = p & m, t & m
            self.tp[i] += (p & t).sum().item()
            self.fp[i] += (p & ~t).sum().item()
            self.fn[i] += (~p & t).sum().item()

    def value(self):
        d = 2 * self.tp / np.maximum(2 * self.tp + self.fp + self.fn, 1)
        return float(d.mean()), d.tolist()


@torch.no_grad()
def evaluate(net, batches, label_manager, interventions=None):
    """Pooled Dice; `interventions` maps block name -> fn(block_output) -> new output."""
    handles = []
    blocks = dict(conv_blocks(net)[0])
    for name, fn in (interventions or {}).items():
        handles.append(blocks[name].register_forward_hook(lambda mod, inp, out, fn=fn: fn(out)))
    dice = Dice(label_manager)
    for x, y, m in batches:
        dice.update(net(x), y, m)
    for h in handles:
        h.remove()
    return dice.value()


@torch.no_grad()
def activation_stats(net, batches):
    """Accumulate first/second moments of pre- and post-activation channels over valid pixels."""
    blocks, _ = conv_blocks(net)
    acc = {}
    state = {}

    def accumulate(key, out):
        mask = state["mask"]
        if mask.shape[-2:] != out.shape[-2:]:
            mask = F.interpolate(mask[:, None].float(), size=out.shape[-2:], mode="nearest")[:, 0].bool()
        v = out.permute(0, 2, 3, 1)[mask].double()
        a = acc.setdefault(key, dict(n=0, s=0, ss=0, pos=0))
        a["n"] += v.shape[0]
        a["s"] = a["s"] + v.sum(0)
        a["ss"] = a["ss"] + v.T @ v
        a["pos"] = a["pos"] + (v > 0).sum(0).double()

    handles = []
    for name, blk in blocks:
        handles.append(blk.norm.register_forward_hook(lambda mod, inp, out, name=name: accumulate((name, "pre"), out)))
        handles.append(blk.register_forward_hook(lambda mod, inp, out, name=name: accumulate((name, "post"), out)))
    for x, _, m in batches:
        state["mask"] = m
        net(x)
    for h in handles:
        h.remove()

    stats = {}
    for (name, kind), a in acc.items():
        mean = a["s"] / a["n"]
        cov = a["ss"] / a["n"] - torch.outer(mean, mean)
        stats[(name, kind)] = dict(mean=mean.float(), cov=cov.float(), frac_pos=(a["pos"] / a["n"]).float())
    return stats


def spectrum_summary(cov):
    ev = torch.linalg.eigvalsh(cov.double()).flip(0).clamp(min=0).cpu().numpy()
    tot = ev.sum()
    if tot <= 0:
        return dict(eigvals=ev.tolist(), k90=0, k95=0, k99=0, participation_ratio=0.0)
    cum = np.cumsum(ev) / tot
    k = lambda q: int(np.searchsorted(cum, q) + 1)
    return dict(
        eigvals=(ev / tot).tolist(),
        k90=k(0.90), k95=k(0.95), k99=k(0.99),
        participation_ratio=float(tot ** 2 / (ev ** 2).sum()),
    )


def corr_summary(cov):
    std = cov.diag().clamp(min=0).sqrt()
    alive = std > 1e-6 * max(std.max().item(), 1e-12)
    c = cov / torch.outer(std, std).clamp(min=1e-12)
    C = c.shape[0]
    off = ~torch.eye(C, dtype=bool, device=c.device) & alive[:, None] & alive[None, :]
    absr = c.abs().masked_fill(~off, 0)
    vals = c[off]
    return dict(
        corr=c.cpu().numpy().round(4).tolist(),
        n_dead=int((~alive).sum()),
        mean_abs_r=float(vals.abs().mean()) if vals.numel() else 0.0,
        max_abs_r_per_channel=absr.max(1).values.cpu().numpy().tolist(),
        n_pairs_r_gt_0p8=int((vals > 0.8).sum() // 2),
        n_pairs_r_lt_m0p8=int((vals < -0.8).sum() // 2),
    )


def weight_summary(conv):
    W = conv.weight.detach().flatten(1).float()
    Wn = W / W.norm(dim=1, keepdim=True).clamp(min=1e-12)
    cos = Wn @ Wn.T
    C = cos.shape[0]
    off = ~torch.eye(C, dtype=bool, device=cos.device)
    sv = torch.linalg.svdvals(W).cpu().numpy()
    e = sv ** 2
    return dict(
        mean_abs_cos=float(cos[off].abs().mean()),
        max_abs_cos_per_filter=cos.abs().masked_fill(~off, 0).max(1).values.cpu().numpy().tolist(),
        n_pairs_cos_gt_0p8=int((cos[off] > 0.8).sum() // 2),
        n_pairs_cos_lt_m0p8=int((cos[off] < -0.8).sum() // 2),
        filter_norms=W.norm(dim=1).cpu().numpy().tolist(),
        sv_energy=(e / e.sum()).tolist(),
        participation_ratio=float(e.sum() ** 2 / (e ** 2).sum()),
        fan_in=int(W.shape[1]),
    )


def run(args, dataset, cfg, device):
    trained = not args.init
    net, label_manager, cm = build_model(dataset, cfg, args.trainer, args.plans, args.fold, device, trained=trained)
    slices = load_val_slices(dataset, args.plans, args.fold, args.max_slices, args.seed)
    batches = make_batches(slices, total_stride(cm), args.batch_size, device)
    blocks, meta = conv_blocks(net)
    base_dice, base_per_class = evaluate(net, batches, label_manager)
    print(f"{dataset} {cfg}{' (init)' if not trained else ''}: {len(slices)} slices, pooled Dice {base_dice:.4f}")

    stats = activation_stats(net, batches)
    layers = []
    for (name, blk), mt in zip(blocks, meta):
        post, pre = stats[(name, "post")], stats[(name, "pre")]
        layers.append(dict(
            **mt,
            channels=blk.conv.out_channels,
            weights=weight_summary(blk.conv),
            post=dict(**spectrum_summary(post["cov"]), **corr_summary(post["cov"]), frac_pos=post["frac_pos"].tolist(),
                      std=post["cov"].diag().clamp(min=0).sqrt().tolist()),
            pre=dict(**spectrum_summary(pre["cov"]), **corr_summary(pre["cov"])),
        ))

    if args.ablate and trained:
        for L, (name, blk) in zip(layers, blocks):
            mean = stats[(name, "post")]["mean"].to(device)
            cov = stats[(name, "post")]["cov"].to(device)
            C = L["channels"]
            zero = []
            for c in range(C):
                def zero_fn(out, c=c):
                    out = out.clone()
                    out[:, c] = 0
                    return out

                zero.append(evaluate(net, batches, label_manager, {name: zero_fn})[0])
            ev, V = torch.linalg.eigh(cov)
            V = V.flip(1)
            ks = sorted({k for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48] if k < C})
            rank = {}
            for k in ks:
                P = V[:, :k] @ V[:, :k].T

                def proj_fn(out, P=P):
                    o = out.permute(0, 2, 3, 1) - mean
                    return (o @ P + mean).permute(0, 3, 1, 2).contiguous()

                rank[k] = evaluate(net, batches, label_manager, {name: proj_fn})[0]
            L["ablation"] = dict(zero=zero, rank_k=rank)
            print(f"  {name:40s} zero min {min(zero):.3f} "
                  f"rank-k {', '.join(f'{k}:{v:.3f}' for k, v in rank.items())}")

    out = OUT_ROOT / dataset / args.trainer / f"{cfg}{'_init' if not trained else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(dataset=dataset, cfg=cfg, trained=trained, n_slices=len(slices),
                   base_dice=base_dice, base_dice_per_class=base_per_class, layers=layers), open(out, "w"))
    print(f"  -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--cfgs", nargs="+", required=True)
    p.add_argument("--trainer", default="nnUNetTrainer_100epochs")
    p.add_argument("--plans", default="nnUNetPlans")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--max_slices", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablate", action="store_true")
    p.add_argument("--init", action="store_true", help="analyse the randomly initialised network instead")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in args.datasets:
        for c in args.cfgs:
            run(args, d, c, device)


if __name__ == "__main__":
    main()
