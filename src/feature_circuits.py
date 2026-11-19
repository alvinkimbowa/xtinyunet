"""Features and circuits of a trained (X)TinyU-Net.

A feature is a non-negative mix of a block's filters, found by NMF on the block's
activations over fold-0 validation pixels; each block gets the fewest features
that reconstruct its activations with R^2 >= 0.9. Each feature is then profiled
like a filter (concept IoU, maps on example images, effect of removing it), and
circuits are traced between features of consecutive blocks.

Circuit drive: the conv before InstanceNorm is linear, so the contribution of a
source feature (its code map times its filter mix) to every target conv channel
is computed exactly. It is carried through the per-image InstanceNorm scale and
projected onto the target feature with the pseudo-inverse of the target's mix
(LeakyReLU is treated as identity). The drive is the mean of that contribution
where the target feature is in its top 20% minus the mean elsewhere, in units
of the target feature's std. The part of the source block not captured by its
features is traced as "rest".

Output: results/analysis/feature_circuits/<dataset>.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import NMF

from filter_redundancy import build_model, conv_blocks, evaluate, load_val_slices, make_batches, total_stride
from filter_roles import (SHORT, XTINY_WIDTH, block_label, block_sources, class_masks, effect, effects, examples,
                          in_out_ratio, profile, receptive_fields, role, role_phrase)

OUT_ROOT = Path("results/analysis/feature_circuits")
LETTERS = "ABCDEFGHIJKLMNOP"


@torch.no_grad()
def sample_activations(net, batches, n_per_block=60000, seed=0):
    blocks, _ = conv_blocks(net)
    g = torch.Generator(device="cpu").manual_seed(seed)
    acc = {n: [] for n, _ in blocks}
    state = {}

    def hook(name):
        def fn(mod, inp, out):
            valid = F.interpolate(state["m"][:, None].float(), size=out.shape[-2:], mode="area")[:, 0] > 0.5
            acc[name].append(out.permute(0, 2, 3, 1)[valid].clamp(min=0).float().cpu())
        return fn

    hs = [blk.register_forward_hook(hook(n)) for n, blk in blocks]
    for x, y, m in batches:
        state["m"] = m
        net(x)
    for h in hs:
        h.remove()
    out = {}
    for n, v in acc.items():
        v = torch.cat(v)
        idx = torch.randperm(v.shape[0], generator=g)[:n_per_block]
        out[n] = v[idx].numpy()
    return out


def fit_features(A, r2_target=0.9, seed=0):
    """Fewest NMF components with R^2 >= r2_target; rows of H scaled to unit norm."""
    tss = ((A - A.mean(0)) ** 2).sum()
    for k in range(1, A.shape[1] + 1):
        m = NMF(n_components=k, init="nndsvda", max_iter=600, random_state=seed)
        W = m.fit_transform(A)
        H = m.components_
        r2 = 1 - ((A - W @ H) ** 2).sum() / tss
        if r2 >= r2_target or k == A.shape[1]:
            nrm = np.linalg.norm(H, axis=1, keepdims=True) + 1e-12
            H, W = H / nrm, W * nrm.T
            energy = (W ** 2).mean(0)
            order = np.argsort(-energy)
            return H[order], float(r2), (energy[order] / energy.sum()).tolist()


def codes(out, H, iters=30):
    """Non-negative codes X with relu(out) ~ X H, per pixel; out [B,C,h,w] -> [B,K,h,w]."""
    A = out.clamp(min=0).permute(0, 2, 3, 1)
    sh = A.shape
    A = A.reshape(-1, sh[-1])
    X = (A @ torch.linalg.pinv(H)).clamp(min=1e-6)
    HHt = H @ H.T
    AHt = A @ H.T
    for _ in range(iters):
        X = X * AHt / (X @ HHt + 1e-9)
    return X.reshape(*sh[:-1], -1).permute(0, 3, 1, 2)


@torch.no_grad()
def feature_circuits(net, batches, Hs, q=0.8):
    blocks, _ = conv_blocks(net)
    srcs = block_sources(net)
    tconv = {}
    for st in range(len(net.decoder.stages)):
        first = [n for n, _ in blocks if n.startswith(f"decoder.stages.{st}.")][0]
        tconv[first] = net.decoder.transpconvs[st]
    cap = {}
    hs = []
    for name, blk in blocks:
        hs.append(blk.conv.register_forward_hook(lambda m, i, o, name=name: cap.update({(name, "in"): i[0], (name, "z"): o})))
        hs.append(blk.register_forward_hook(lambda m, i, o, name=name: cap.__setitem__((name, "out"), o)))
    acc = {}
    for x, y, m in batches:
        net(x)
        for name, blk in blocks:
            conv, W = blk.conv, blk.conv.weight
            Ht = Hs[name]
            P = torch.linalg.pinv(Ht)
            out, z = cap[(name, "out")], cap[(name, "z")]
            size = out.shape[-2:]
            valid = F.interpolate(m[:, None].float(), size=size, mode="area")[:, 0] > 0.5
            Xt = codes(out, Ht)
            xv = Xt.permute(0, 2, 3, 1)[valid]
            thr = torch.quantile(xv.float(), q, dim=0)
            on = ((Xt >= thr[None, :, None, None]) & valid[:, None]).double()
            off = ((Xt < thr[None, :, None, None]) & valid[:, None]).double()
            scale = blk.norm.weight[None, :, None, None] / (z.std(dim=(2, 3), unbiased=False, keepdim=True) + blk.norm.eps)

            def to_feat(contrib):
                return torch.einsum("bthw,tk->bkhw", contrib * scale, P)

            terms, labels = [], []
            kind, a, b = srcs[name]
            if kind == "image":
                xin = cap[(name, "in")]
                for c in range(xin.shape[1]):
                    terms.append(to_feat(F.conv2d(xin[:, c:c + 1], W[:, c:c + 1], stride=conv.stride, padding=conv.padding)))
                    labels.append(("image", c))
            else:
                parts = []
                if kind == "up+skip":
                    tc = tconv[name]
                    parts.append(("below", a, W[:, :tc.out_channels], tc))
                    parts.append(("skip", b, W[:, tc.out_channels:], None))
                else:
                    parts.append(("prev", a, W, None))
                for via, sname, Wp, tc in parts:
                    src_out = cap[(sname, "out")]
                    Hsrc = Hs[sname]
                    Xs = codes(src_out, Hsrc)
                    recon = torch.einsum("bkhw,kc->bchw", Xs, Hsrc)
                    rest = src_out - recon
                    pieces = [(Xs[:, i:i + 1], Hsrc[i]) for i in range(Hsrc.shape[0])]
                    for i, (Xi, Hi) in enumerate(pieces + [(rest, None)]):
                        inp = Xi * Hi[None, :, None, None] if Hi is not None else Xi
                        if tc is not None:
                            inp = F.conv_transpose2d(inp, tc.weight, stride=tc.stride)
                        terms.append(to_feat(F.conv2d(inp, Wp, stride=conv.stride, padding=conv.padding)))
                        labels.append((via, i if Hi is not None else "rest"))
            r = acc.setdefault(name, dict(on=0, off=0, non=0, noff=0, s=0, ss=0, n=0, labels=labels))
            S = torch.stack(terms, 1).double()
            r["on"] = r["on"] + (S * on[:, None]).sum((0, 3, 4))
            r["off"] = r["off"] + (S * off[:, None]).sum((0, 3, 4))
            r["non"] = r["non"] + on.sum((0, 2, 3))
            r["noff"] = r["noff"] + off.sum((0, 2, 3))
            xd = xv.double()
            r["s"] = r["s"] + xd.sum(0)
            r["ss"] = r["ss"] + (xd * xd).sum(0)
            r["n"] += xd.shape[0]
    for h in hs:
        h.remove()
    res = {}
    for name, _ in blocks:
        r = acc[name]
        std = (r["ss"] / r["n"] - (r["s"] / r["n"]) ** 2).clamp(min=1e-12).sqrt()
        drive = (r["on"] / r["non"].clamp(min=1) - r["off"] / r["noff"].clamp(min=1)) / std
        res[name] = dict(drive=drive.T.cpu().numpy(), labels=r["labels"], kind=srcs[name][0], a=srcs[name][1], b=srcs[name][2])
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--cfg", default=None)
    p.add_argument("--trainer", default="nnUNetTrainer_100epochs")
    p.add_argument("--plans", default="nnUNetPlans")
    p.add_argument("--max_slices", type=int, default=96)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    w = XTINY_WIDTH.get(a.dataset, 8)
    cfg = a.cfg or f"2d_tiny{w}"
    device = torch.device(a.device)
    net, lm, cm = build_model(a.dataset, cfg, a.trainer, a.plans, 0, device)
    slices = load_val_slices(a.dataset, a.plans, 0, a.max_slices, 0)
    stride = total_stride(cm)
    batches = make_batches(slices, stride, 8, device)
    blocks, meta = conv_blocks(net)
    label_of = {n: block_label(mt) for (n, _), mt in zip(blocks, meta)}

    samples = sample_activations(net, batches)
    Hs, fit = {}, {}
    for name, _ in blocks:
        H, r2, energy = fit_features(samples[name])
        Hs[name] = torch.tensor(H, dtype=torch.float32, device=device)
        fit[name] = dict(r2=r2, energy=energy)
        print(f"  {label_of[name]}: {H.shape[0]} features, R2 {r2:.3f}")

    view = {n: (lambda o, n=n: codes(o, Hs[n])) for n in Hs}

    def remove(n):
        def fn(o, i):
            X = codes(o, Hs[n])
            return o - X[:, i:i + 1] * Hs[n][i][None, :, None, None]
        return fn

    ablate = {n: (Hs[n].shape[0], remove(n)) for n in Hs}
    layers = profile(net, batches, lm, view=view)
    eff = effects(net, batches, lm, ablate=ablate)
    base_dice = evaluate(net, batches, lm)[0]
    drops = {}
    for n in Hs:
        drops[n] = [base_dice - evaluate(net, batches, lm, {n: (lambda o, n=n, i=i: remove(n)(o, i))})[0]
                    for i in range(Hs[n].shape[0])]
    ex, maps = examples(net, slices, stride, lm, device, view=view)
    circ = feature_circuits(net, batches, Hs)

    ch_names = list(json.load(open(Path("data/nnUNet_preprocessed") / a.dataset / "dataset.json"))["channel_names"].values())
    fname = lambda n, i: f"{label_of[n]}-{LETTERS[i]}"
    class_list = [n for n, _ in class_masks(torch.zeros(1, dtype=torch.long), lm)]
    feats, roles = [], {}
    for L, (name, _) in zip(layers, blocks):
        for i in range(L["C"]):
            f = dict(active=L["active"][i], iou={k: v[i] for k, v in L["iou"].items()},
                     mean={k: v[i] for k, v in L["mean"].items()}, mean_fg=L["mean_fg"][i])
            kind, gain = role(f, L["base"], class_list)
            roles[fname(name, i)] = kind
            H = Hs[name][i].cpu().numpy()
            mix = [dict(filter=f"{label_of[name]}·{c}", w=round(float(v / H.max()), 2)) for c, v in enumerate(H) if v / H.max() >= 0.2]
            mix.sort(key=lambda m: -m["w"])
            feats.append(dict(id=fname(name, i), block=label_of[name], part=L["part"], level=L["level"], k=i, role=kind, gain=gain,
                              share=round(fit[name]["energy"][i], 3), mix=mix, ratio=round(in_out_ratio(f), 3),
                              iou={k: round(v, 3) for k, v in f["iou"].items()},
                              drop=round(float(drops[name][i]), 4), d_in=round(eff[name][i][0], 4), d_out=round(eff[name][i][1], 4),
                              effect=effect(*eff[name][i]), maps=maps[name][i], name=name))
    idx = {f["id"]: j for j, f in enumerate(feats)}
    for f in feats:
        r = circ[f["name"]]
        d = r["drive"][f["k"]]
        tot = np.abs(d).sum() + 1e-12
        items = []
        for (via, i), v in zip(r["labels"], d):
            if via == "image":
                src = f"input {ch_names[i]}"
            else:
                sname = r["a"] if via in ("prev", "below") else r["b"]
                src = f"{label_of[sname]}-rest" if i == "rest" else fname(sname, i)
            items.append(dict(src=src, via={"prev": "previous block", "below": "from below", "skip": "skip connection", "image": "image"}[via],
                              role=roles.get(src, "image" if via == "image" else "rest"), drive=round(float(v), 3),
                              share=round(float(abs(v) / tot), 3), i=idx.get(src)))
        items.sort(key=lambda it: -abs(it["drive"]))
        f["inputs"] = items
        f.pop("name")

    def ph(it):
        what = {"image": "the image", "rest": "what the features miss"}.get(it["role"]) or role_phrase(it["role"])
        return f"{it['src']} ({what})"
    for f in feats:
        pos = [it for it in f["inputs"] if it["drive"] > 0 and it["share"] >= 0.12][:3]
        neg = [it for it in f["inputs"] if it["drive"] < 0 and it["share"] >= 0.12][:2]
        t = ""
        if pos:
            t = "Built up from " + ", ".join(ph(it) for it in pos)
        if neg:
            t += ("; " if t else "") + "pushed down by " + ", ".join(ph(it) for it in neg)
        f["built"] = (t + ".") if t else "No single input stands out."

    first = blocks[0][1].conv.weight.detach().cpu()
    short = SHORT.get(a.dataset, a.dataset)
    out = dict(dataset=short, cfg=cfg, n_slices=len(slices), base_dice=base_dice, concepts=list(layers[0]["iou"].keys()),
               base={L["label"]: {k: round(v, 3) for k, v in L["base"].items()} for L in layers},
               rf={L["label"]: rf for L, rf in zip(layers, receptive_fields(net))},
               blocks=[dict(label=L["label"], part=L["part"], level=L["level"], K=L["C"], C=blk.conv.out_channels,
                            r2=round(fit[n]["r2"], 3)) for L, (n, blk) in zip(layers, blocks)],
               first_kernels=first.numpy().round(3).tolist(), examples=ex, features=feats)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / f"{a.dataset}.json"
    json.dump(out, open(path, "w"))
    print(short, cfg, len(feats), "features over", len(blocks), "blocks", f"-> {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
