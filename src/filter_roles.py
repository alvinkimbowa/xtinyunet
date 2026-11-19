"""Per-filter profile of a trained (X)TinyU-Net: what each conv filter responds to.

For every output channel of every conv block (conv -> InstanceNorm -> LeakyReLU),
on fold-0 validation slices:

- dissection: best IoU, over thresholds, between the channel's map and each
  concept (each class, its boundary, background, bright, dark, image edges),
- inside-vs-outside contrast: mean activation inside the object against outside,
- a role label from the contrast first and the IoU gain second (see role()),
- effect: mean change in object probability inside and outside the true object
  when the channel is zeroed,
- circuit: how strongly each input filter drives the channel where it fires,
  how its 3x3 weight slice reads that input, and a sentence composing them,
- fraction of pixels where the channel is active (> 0) and receptive field size,
- feature-map thumbnails (PNG, base64) on a few example slices, together with
  the slices' input, ground truth and prediction.

Zeroing-ablation Dice drops come from results/analysis/filter_redundancy/compact.json.
Output: results/analysis/filter_roles/<dataset>.json
"""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from filter_redundancy import build_model, conv_blocks, load_val_slices, make_batches, total_stride

OUT_ROOT = Path("results/analysis/filter_roles")
COMPACT = Path("results/analysis/filter_redundancy/compact.json")
SHORT = {"Dataset301_busbra": "BUS-BRA", "Dataset302_EchoNet-Dynamic": "EchoNet", "Dataset300_isic2018": "ISIC2018",
         "Dataset309_FIVES": "FIVES", "Dataset137_BraTS2021": "BraTS2021", "Dataset306_ACDC": "ACDC"}
XTINY_WIDTH = {"Dataset137_BraTS2021": 16}
THUMB = 72
INPUT_THUMB = 160


def block_label(meta):
    j = int(meta["name"].split(".")[-1]) if meta["name"].split(".")[-1].isdigit() else 0
    return f"{'E' if meta['part'] == 'enc' else 'D'}{meta['level']}.{j}"


def class_masks(y, label_manager):
    """List of (name, bool mask[B,H,W]) for each foreground class or region."""
    if label_manager.has_regions:
        groups = label_manager.foreground_regions
    else:
        groups = label_manager.foreground_labels
    names = {}
    for name, v in label_manager.label_dict.items():
        names[tuple(v) if isinstance(v, (list, tuple)) else (v,)] = name
    out = []
    for g in groups:
        key = tuple(g) if isinstance(g, (list, tuple)) else (g,)
        out.append((names.get(key, str(key)), torch.isin(y, torch.tensor(key, device=y.device))))
    return out


def sobel_mag(x):
    k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
    w = torch.stack([k, k.T])[:, None]
    g = F.conv2d(x.mean(1, keepdim=True), w, padding=1)
    return g.pow(2).sum(1).sqrt()


def concepts(x, y, m, size, label_manager):
    """Bool masks [B,h,w] at the block resolution for each concept, plus the valid mask."""
    valid = F.interpolate(m[:, None].float(), size=size, mode="area")[:, 0] > 0.5
    img = F.interpolate(x.mean(1, keepdim=True), size=size, mode="area")[:, 0]
    grad = F.interpolate(sobel_mag(x)[:, None], size=size, mode="area")[:, 0]
    cls = [(n, F.interpolate(c[:, None].float(), size=size, mode="area")[:, 0] > 0.5) for n, c in class_masks(y, label_manager)]
    fg = torch.stack([c for _, c in cls]).any(0)
    out = {}
    for n, c in cls:
        out[n] = c
        f = c.float()[:, None]
        out[f"{n} boundary"] = (F.max_pool2d(f, 3, 1, 1)[:, 0] > 0) & ~(-F.max_pool2d(-f, 3, 1, 1)[:, 0] > 0)
    out["background"] = ~fg
    iv, gv = img[valid], grad[valid]
    out["bright"] = img > torch.quantile(iv.float(), 0.7)
    out["dark"] = img < torch.quantile(iv.float(), 0.3)
    out["edges"] = grad > torch.quantile(gv.float(), 0.8)
    return {k: v & valid for k, v in out.items()}, valid


NBINS = 64


@torch.no_grad()
def profile(net, batches, label_manager, view=None):
    """Dissection: for every channel and concept, the best IoU between the thresholded map and the concept.

    `view` maps a block name to fn(block_output) -> unit maps [B, K, H, W]; units are channels by default.
    """
    view = view or {}
    blocks, meta = conv_blocks(net)
    state, acc = {}, {}

    edges = {}
    def sample(name):
        def fn(mod, inp, out):
            out = view[name](out) if name in view else out
            v = out.permute(0, 2, 3, 1).reshape(-1, out.shape[1])
            idx = torch.randint(0, v.shape[0], (min(4096, v.shape[0]),), device=v.device)
            edges.setdefault(name, []).append(v[idx].float())
        return fn
    hs = [blk.register_forward_hook(sample(n)) for n, blk in blocks]
    for x, y, m in batches:
        net(x)
    for h in hs:
        h.remove()
    for n in list(edges):
        v = torch.cat(edges[n])
        q = torch.linspace(0, 1, NBINS + 1, device=v.device)[1:-1]
        edges[n] = torch.quantile(v, q, dim=0).T.contiguous()

    def hist(name):
        def fn(mod, inp, out):
            out = view[name](out) if name in view else out
            cmap, valid = concepts(state["x"], state["y"], state["m"], out.shape[-2:], label_manager)
            a = out.permute(0, 2, 3, 1)[valid]
            C = a.shape[1]
            b = torch.stack([torch.bucketize(a[:, c].contiguous(), edges[name][c]) for c in range(C)], 1)
            r = acc.setdefault(name, dict(all=torch.zeros(C, NBINS, device=a.device, dtype=torch.double), k={}, n=0, pos=0, s=0))
            r["s"] = r["s"] + a.clamp(min=0).double().sum(0)
            r["all"].scatter_add_(1, b.T, torch.ones_like(b.T, dtype=torch.double))
            r["n"] += a.shape[0]
            r["pos"] = r["pos"] + (a > 0).sum(0).double()
            for k, msk in cmap.items():
                mv = msk[valid]
                t = r["k"].setdefault(k, dict(h=torch.zeros(C, NBINS, device=a.device, dtype=torch.double), n=0, s=0))
                t["s"] = t["s"] + a[mv].clamp(min=0).double().sum(0)
                bb = b[mv]
                t["h"].scatter_add_(1, bb.T, torch.ones_like(bb.T, dtype=torch.double))
                t["n"] += int(mv.sum())
        return fn
    hs = [blk.register_forward_hook(hist(n)) for n, blk in blocks]
    for x, y, m in batches:
        state.update(x=x, y=y, m=m)
        net(x)
    for h in hs:
        h.remove()

    out = []
    for (name, blk), mt in zip(blocks, meta):
        r = acc[name]
        A = r["all"].flip(1).cumsum(1)[:, :-1]
        iou, base, mean = {}, {}, {}
        for k, t in r["k"].items():
            mean[k] = (t["s"] / max(t["n"], 1)).tolist()
            I = t["h"].flip(1).cumsum(1)[:, :-1]
            u = A + t["n"] - I
            iou[k] = (I / u.clamp(min=1)).max(1).values.tolist()
            base[k] = t["n"] / max(r["n"], 1)
        out.append(dict(name=name, label=block_label(mt), part=mt["part"], level=mt["level"], C=r["all"].shape[0],
                        active=(r["pos"] / r["n"]).tolist(), iou=iou, base=base, mean=mean,
                        mean_fg=((r["s"] - r["k"]["background"]["s"]) / max(r["n"] - r["k"]["background"]["n"], 1)).tolist()))
    return out


def object_prob(logits, label_manager):
    if label_manager.has_regions:
        return torch.sigmoid(logits[:, 0])
    return 1 - torch.softmax(logits, 1)[:, 0]


@torch.no_grad()
def effects(net, batches, label_manager, ablate=None):
    """Mean change in object probability inside and outside the true object when one unit is removed.

    `ablate` maps a block name to (number of units, fn(block_output, unit) -> new output);
    by default a unit is one channel and removing it sets it to zero.
    """
    blocks, _ = conv_blocks(net)
    base = [object_prob(net(x), label_manager) for x, _, _ in batches]
    out = {}
    for name, blk in blocks:
        res = []
        n_units, fn = (ablate or {}).get(name, (blk.conv.out_channels, None))
        for c in range(n_units):
            def zero(mod, inp, o, c=c):
                if fn is not None:
                    return fn(o, c)
                o = o.clone()
                o[:, c] = 0
                return o
            h = blk.register_forward_hook(zero)
            din = dout = nin = nout = 0
            for (x, y, m), p0 in zip(batches, base):
                d = object_prob(net(x), label_manager) - p0
                fg = (y > 0) & m
                bg = (y == 0) & m
                din += d[fg].sum().item(); nin += fg.sum().item()
                dout += d[bg].sum().item(); nout += bg.sum().item()
            h.remove()
            res.append((din / max(nin, 1), dout / max(nout, 1)))
        out[name] = res
    return out


MIN_COVER = 0.002


def contrast(a, b):
    return (a - b) / (a + b + 1e-9)


def role(f, base, classes):
    """Label a unit by what it responds to.

    Inside-vs-outside contrast decides object and background, since a concept
    covering most of the image (background) gets a high IoU even from a map that
    fires everywhere. Contrast c = (in - out) / (in + out) on the mean
    non-negative activation; |c| >= 0.6 means one side is at least 4x the other.
    Otherwise the concept with the largest IoU gain (background excluded) wins
    if the gain exceeds 0.1, then weaker contrast (|c| >= 0.33, 2x), else mixed.
    Contrast is skipped when the object covers under MIN_COVER of the pixels at
    the block's resolution (thin objects vanish in deep blocks).
    """
    if f["active"] < 0.01:
        return "silent", 0.0
    bg = f["mean"]["background"]
    cs = {k: contrast(f["mean"][k], bg) if base[k] >= MIN_COVER else -1.0 for k in classes}
    kc = max(cs, key=cs.get)
    c_bg = -contrast(f["mean_fg"], bg) if 1 - base["background"] >= MIN_COVER else -1.0
    if cs[kc] >= 0.6:
        return kc, round(cs[kc], 3)
    if c_bg >= 0.6:
        return "background", round(c_bg, 3)
    gains = {k: (v - base[k]) / max(1 - base[k], 1e-6) for k, v in f["iou"].items() if k != "background"}
    k = max(gains, key=gains.get)
    if gains[k] > 0.1:
        return k, round(gains[k], 3)
    if cs[kc] >= 0.33:
        return kc, round(cs[kc], 3)
    if c_bg >= 0.33:
        return "background", round(c_bg, 3)
    return "mixed", round(gains[k], 3)


def in_out_ratio(f):
    """Mean activation inside the object divided by outside it."""
    return f["mean_fg"] / max(f["mean"]["background"], 1e-9)


def effect(d_in, d_out, tol=0.02):
    """What zeroing the channel does to the object probability."""
    parts = []
    if d_in < -tol:
        parts.append("supports the object")
    if d_out > tol:
        parts.append("suppresses false positives")
    if d_in > tol:
        parts.append("holds the object back")
    if d_out < -tol:
        parts.append("adds false positives")
    return ", ".join(parts) or "little effect on its own"


def receptive_fields(net):
    """Theoretical receptive field (pixels) of each conv block's output, in network order."""
    blocks, _ = conv_blocks(net)
    rf, jump, out, enc_rf = 1, 1, [], {}
    for name, blk in blocks:
        if name.startswith("encoder"):
            st = blk.conv.stride[0] if isinstance(blk.conv.stride, tuple) else blk.conv.stride
            rf += (blk.conv.kernel_size[0] - 1) * jump
            jump *= st
            out.append(rf)
        else:
            if name.endswith(".0"):
                jump //= 2
            rf += (blk.conv.kernel_size[0] - 1) * jump
            out.append(rf)
    return out


def describe_kernel(k):
    """Plain description of a first-layer 3x3 kernel (summed over input channels)."""
    w = np.asarray(k).sum(0)
    gx = (w[:, 2] - w[:, 0]).sum()
    gy = (w[2, :] - w[0, :]).sum()
    centre = w[1, 1] - (w.sum() - w[1, 1]) / 8
    mean = w.mean()
    parts = {"edge": np.hypot(gx, gy) / 3, "spot": abs(centre), "level": abs(mean) * 3}
    kind = max(parts, key=parts.get)
    if kind == "edge":
        ang = np.degrees(np.arctan2(gy, gx)) % 360
        dirs = ["left to right", "top-left to bottom-right", "top to bottom", "top-right to bottom-left",
                "right to left", "bottom-right to top-left", "bottom to top", "bottom-left to top-right"]
        return f"its kernel is an edge detector: it responds where the image gets brighter from {dirs[int((ang + 22.5) // 45) % 8]}"
    if kind == "spot":
        return f"its kernel is a {'bright' if centre > 0 else 'dark'}-spot detector: the centre weight stands out from its 8 neighbours"
    return f"its kernel is a smoothing filter that passes {'brightness' if mean > 0 else 'darkness'} through"


def describe(f, base, kernel=None):
    """One plain-English paragraph on what a filter does, from its measurements."""
    ratio = f.get("ratio")
    if f["role"] in f["classes"] and ratio is not None:
        where = (f"Fires inside the {f['role']}: {ratio:.1f}× stronger inside than outside." if ratio >= 1.5 else
                 f"Its strongest responses line up with the {f['role']}, though on average it is about as strong outside it.")
        where += (" It is sparse, active on only " if f["active"] < 0.2 else " It is broad, active on " if f["active"] > 0.6 else " Active on ") + f"{f['active'] * 100:.0f}% of pixels."
    elif f["role"] == "background" and ratio is not None:
        where = f"Fires outside the {' / '.join(f['classes'])}: {1 / max(ratio, 1e-9):.1f}× stronger outside than inside, so it marks where the {' / '.join(f['classes'])} is not."
        where += (" It is sparse, active on only " if f["active"] < 0.2 else " It is broad, active on " if f["active"] > 0.6 else " Active on ") + f"{f['active'] * 100:.0f}% of pixels."
    elif f["role"] == "silent":
        where = f"Almost never switches on (active on {f['active'] * 100:.1f}% of pixels), so it carries no information on these images."
    else:
        gains = sorted(((v - base[k]) / max(1 - base[k], 1e-6), k) for k, v in f["iou"].items() if k != "background")[::-1]
        (g1, k1), (g2, k2) = gains[0], gains[1]
        phrase = lambda k: ("inside the " + k if k in f["classes"] else "along the " + k if k.endswith("boundary")
                            else "on the background" if k == "background" else "on image edges" if k == "edges"
                            else f"on {k} regions")
        if g1 < 0.1:
            where = f"Does not line up with any tested concept (best: {phrase(k1)}, weak); it encodes texture or context together with other filters."
        else:
            strength = "strongly" if g1 > 0.45 else "clearly" if g1 > 0.25 else "loosely"
            where = f"Fires {strength} {phrase(k1)}"
            if g2 > 0.1 and g2 > 0.6 * g1:
                where += f", and also {phrase(k2)}"
            where += "."
        spread = f["active"]
        where += (" It is sparse, active on only " if spread < 0.2 else " It is broad, active on " if spread > 0.6 else " Active on ") + f"{spread * 100:.0f}% of pixels."
    if kernel is not None:
        where += " At the input, " + describe_kernel(kernel) + "."
    d = f.get("drop")
    if d is None:
        imp = ""
    elif d > 0.2:
        imp = f" Critical: switching it off costs {d * 100:.0f} Dice points."
    elif d > 0.05:
        imp = f" Important: switching it off costs {d * 100:.1f} Dice points."
    elif d > 0.01:
        imp = f" Minor: switching it off costs {d * 100:.1f} Dice points."
    else:
        imp = " Spare: switching it off changes Dice by under 1 point."
    din, dout = f["d_in"], f["d_out"]
    acts = []
    if din < -0.02:
        acts.append(f"it adds to the object probability inside the {f['classes'][0] if len(f['classes']) == 1 else 'object'} ({-din * 100:.0f} pts)")
    if dout > 0.02:
        acts.append(f"it suppresses false positives outside it ({dout * 100:.0f} pts)")
    if din > 0.02:
        acts.append(f"it holds the object probability down inside ({din * 100:.0f} pts)")
    if dout < -0.02:
        acts.append(f"it raises the object probability outside ({-dout * 100:.0f} pts)")
    if acts:
        imp += " Its job in the output: " + "; ".join(acts) + "."
    return where + imp


def block_sources(net):
    """For each conv block name: (kind, source block name or None).

    kind is "image" for the first conv, "prev" for a conv fed by one earlier block,
    and "up+skip" for the first conv of a decoder stage, fed by the transposed-conv
    upsampling of the block below plus the encoder skip at the same level.
    """
    blocks, meta = conv_blocks(net)
    names = [n for n, _ in blocks]
    enc = [n for n in names if n.startswith("encoder")]
    dec = [n for n in names if n.startswith("decoder")]
    src = {enc[0]: ("image", None, None)}
    for a, b in zip(enc, enc[1:]):
        src[b] = ("prev", a, None)
    n_stages = len(net.encoder.stages)
    last_enc_of_stage = {}
    for n in enc:
        last_enc_of_stage[int(n.split(".")[2])] = n
    below = enc[-1]
    for st in range(len(net.decoder.stages)):
        stage_convs = [n for n in dec if int(n.split(".")[2]) == st]
        skip = last_enc_of_stage[n_stages - 2 - st]
        src[stage_convs[0]] = ("up+skip", below, skip)
        for a, b in zip(stage_convs, stage_convs[1:]):
            src[b] = ("prev", a, None)
        below = stage_convs[-1]
    return src


@torch.no_grad()
def circuits(net, batches, q=0.8):
    """Per filter, how much each input filter drives it.

    The conv before InstanceNorm is linear, so its output splits exactly into one
    term per input channel. For each target channel the drive of a source is the
    mean of that term on the pixels where the target fires (its top 20% after the
    activation) minus the mean elsewhere, in units of the target's pre-norm std,
    with the sign of the norm's scale applied. For the first conv of a decoder
    stage the upsampled channels are traced through the transposed conv back to
    the channels of the block below.
    """
    blocks, _ = conv_blocks(net)
    bmod = dict(blocks)
    srcs = block_sources(net)
    tconv = {}
    for st, stage in enumerate(net.decoder.stages):
        first = [n for n, _ in blocks if n.startswith(f"decoder.stages.{st}.")][0]
        tconv[first] = net.decoder.transpconvs[st]
    cap = {}
    hs = []
    for name, blk in blocks:
        hs.append(blk.conv.register_forward_hook(lambda m, i, o, name=name: cap.__setitem__((name, "in"), i[0])))
        hs.append(blk.conv.register_forward_hook(lambda m, i, o, name=name: cap.__setitem__((name, "z"), o)))
        hs.append(blk.register_forward_hook(lambda m, i, o, name=name: cap.__setitem__((name, "out"), o)))
    acc = {}
    for x, y, m in batches:
        net(x)
        for name, blk in blocks:
            conv = blk.conv
            W = conv.weight
            T, Cin = W.shape[:2]
            xin, z, out = cap[(name, "in")], cap[(name, "z")], cap[(name, "out")]
            size = out.shape[-2:]
            valid = F.interpolate(m[:, None].float(), size=size, mode="area")[:, 0] > 0.5
            ov = out.permute(0, 2, 3, 1)[valid]
            thr = torch.quantile(ov.float(), q, dim=0) if ov.shape[0] < 2 ** 24 else torch.quantile(ov[torch.randperm(ov.shape[0], device=ov.device)[:2 ** 22]].float(), q, dim=0)
            on = (out >= thr[None, :, None, None]) & valid[:, None]
            off = (~on) & valid[:, None]
            kind, a, b = srcs[name]
            terms = []
            if kind == "up+skip":
                tc = tconv[name]
                below = cap[(a, "out")]
                Cup = tc.out_channels
                for k in range(below.shape[1]):
                    up = F.conv_transpose2d(below[:, k:k + 1], tc.weight[k:k + 1], stride=tc.stride)
                    terms.append(F.conv2d(up, W[:, :Cup], stride=conv.stride, padding=conv.padding))
                xs, ws = xin[:, Cup:], W[:, Cup:]
            else:
                xs, ws = xin, W
            C = xs.shape[1]
            g = F.conv2d(xs, ws.permute(1, 0, 2, 3).reshape(C * T, 1, *ws.shape[-2:]), stride=conv.stride,
                         padding=conv.padding, groups=C)
            g = g.reshape(g.shape[0], C, T, *g.shape[-2:])
            terms += [g[:, c] for c in range(C)]
            r = acc.setdefault(name, dict(on=0, off=0, non=0, noff=0, zs=0, zss=0, n=0))
            on_f, off_f = on.double(), off.double()
            S = torch.stack(terms, 1).double()
            r["on"] = r["on"] + (S * on_f[:, None]).sum((0, 3, 4))
            r["off"] = r["off"] + (S * off_f[:, None]).sum((0, 3, 4))
            r["non"] = r["non"] + on_f.sum((0, 2, 3))
            r["noff"] = r["noff"] + off_f.sum((0, 2, 3))
            zv = z.permute(0, 2, 3, 1)[valid].double()
            r["zs"] = r["zs"] + zv.sum(0)
            r["zss"] = r["zss"] + (zv * zv).sum(0)
            r["n"] += zv.shape[0]
    for h in hs:
        h.remove()
    res = {}
    for name, blk in blocks:
        r = acc[name]
        zstd = (r["zss"] / r["n"] - (r["zs"] / r["n"]) ** 2).clamp(min=1e-12).sqrt()
        sign = torch.sign(blk.norm.weight.detach().double())
        drive = (r["on"] / r["non"].clamp(min=1) - r["off"] / r["noff"].clamp(min=1)) / zstd * sign
        kind, a, b = srcs[name]
        W = blk.conv.weight.detach().cpu()
        if kind == "up+skip":
            Cup = tconv[name].out_channels
            slices = [None] * cap[(a, "out")].shape[1] + [W[:, Cup + c] for c in range(W.shape[1] - Cup)]
        else:
            slices = [W[:, c] for c in range(W.shape[1])]
        res[name] = dict(kind=kind, a=a, b=b, drive=drive.T.cpu().numpy(), slices=slices)
    return res


DIRS = ["the right", "the lower right", "below", "the lower left", "the left", "the upper left", "above", "the upper right"]


def slice_pattern(w, sign):
    """How a 3x3 weight slice reads its input, given the sign the source acts with."""
    w = np.asarray(w) * sign
    gx = (w[:, 2] - w[:, 0]).sum()
    gy = (w[2, :] - w[0, :]).sum()
    centre = w[1, 1] - (w.sum() - w[1, 1]) / 8
    parts = {"edge": np.hypot(gx, gy) / 3, "spot": abs(centre), "level": abs(w.mean()) * 3}
    kind = max(parts, key=parts.get)
    if kind == "edge":
        ang = np.degrees(np.arctan2(gy, gx)) % 360
        return f"as an edge, more of it toward {DIRS[int((ang + 22.5) // 45) % 8]}"
    if kind == "spot":
        return "as a spot, more of it here than around" if centre > 0 else "as a ring, more of it around than here"
    return "over the 3×3 neighbourhood"


ROLE_PHRASE = {"dark": "dark regions", "bright": "bright regions", "edges": "image edges", "background": "background",
               "mixed": "no clear concept", "silent": "silent"}


def role_phrase(role):
    if role in ROLE_PHRASE:
        return ROLE_PHRASE[role]
    return role if role.endswith("boundary") else f"the {role}"


def built_from(name, c, circ, label_of, roles, channel_names, k_on=3, k_off=2, min_share=0.12):
    """Inputs that drive filter c of block `name`, and a sentence composing them."""
    r = circ[name]
    d = r["drive"][c]
    tot = np.abs(d).sum() + 1e-12
    items = []
    for s, v in enumerate(d):
        if r["kind"] == "image":
            src, role, via = f"input {channel_names[s]}", "image", "image"
        elif r["kind"] == "up+skip":
            nb = len([sl for sl in r["slices"] if sl is None])
            if s < nb:
                src, via = f"{label_of[r['a']]}·{s}", "from below (upsampled)"
            else:
                src, via = f"{label_of[r['b']]}·{s - nb}", "skip connection"
            role = roles.get(src)
        else:
            src, via = f"{label_of[r['a']]}·{s}", "previous block"
            role = roles.get(src)
        sl = r["slices"][s]
        pat = None if sl is None else slice_pattern(sl[c].numpy(), np.sign(v))
        items.append(dict(src=src, role=role, via=via, drive=round(float(v), 3), share=round(float(abs(v) / tot), 3), pattern=pat))
    items.sort(key=lambda it: -abs(it["drive"]))
    pos = [it for it in items if it["drive"] > 0 and it["share"] >= min_share][:k_on]
    neg = [it for it in items if it["drive"] < 0 and it["share"] >= min_share][:k_off]
    if not pos and items[0]["drive"] > 0:
        pos = items[:1]
    def ph(it):
        what = "the image" if it["role"] == "image" else role_phrase(it["role"])
        pat = f", read {it['pattern']}" if it["pattern"] else ""
        return f"{it['src']} ({what}){pat}"
    text = ""
    if pos:
        text = "Turned on where " + "; and ".join(ph(it) + " is high" for it in pos)
    if neg:
        text += ("; " if text else "Mostly ") + "held off where " + "; and ".join(ph(it) + " is high" for it in neg)
    text = (text + ".") if text else "No input stands out; many small contributions add up."
    return items, text


def png(a, size_cap, mode="L"):
    h, w = a.shape[-2:]
    s = min(1.0, size_cap / max(h, w))
    if s < 1:
        a = F.interpolate(a[None, None].float(), size=(max(1, round(h * s)), max(1, round(w * s))), mode="area")[0, 0]
    im = Image.fromarray(a.cpu().numpy().astype(np.uint8), mode)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def examples(net, slices, stride, label_manager, device, k=3, view=None):
    fg_area = np.array([(s[1] > 0).sum() for s in slices])
    order = [i for i in np.argsort(fg_area) if fg_area[i] > 0]
    pick = [order[int(q * (len(order) - 1))] for q in (0.3, 0.6, 0.9)][:k]
    blocks, _ = conv_blocks(net)
    feats = {name: [] for name, _ in blocks}
    ex = []
    for i in pick:
        img, seg = slices[i]
        (x, y, m), = make_batches([(img, seg)], stride, 1, device)
        cap = {}
        view = view or {}
        hs = [blk.register_forward_hook(lambda mod, inp, out, name=name: cap.__setitem__(name, (view[name](out) if name in view else out)[0])) for name, blk in blocks]
        logits = net(x)
        for h in hs:
            h.remove()
        h0, w0 = seg.shape
        for name, _ in blocks:
            f = cap[name]
            sh, sw = f.shape[-2] / x.shape[-2], f.shape[-1] / x.shape[-1]
            feats[name].append(f[:, :max(1, round(h0 * sh)), :max(1, round(w0 * sw))])
        xi = x[0].mean(0)[:h0, :w0]
        lo, hi = torch.quantile(xi.flatten(), 0.01), torch.quantile(xi.flatten(), 0.99)
        if label_manager.has_regions:
            pred = (torch.sigmoid(logits[0]) > 0.5).long().sum(0)
        else:
            pred = logits[0].argmax(0)
        ex.append(dict(img=png(((xi - lo) / (hi - lo + 1e-8)).clamp(0, 1) * 255, INPUT_THUMB),
                       gt=png(y[0, :h0, :w0].clamp(min=0), INPUT_THUMB), pred=png(pred[:h0, :w0], INPUT_THUMB),
                       h=h0, w=w0))
    maps = {}
    for name, fl in feats.items():
        C = fl[0].shape[0]
        per = []
        for c in range(C):
            vals = torch.cat([f[c].flatten() for f in fl])
            top = torch.quantile(vals.float(), 0.995).clamp(min=1e-6)
            per.append([png((f[c] / top).clamp(0, 1) * 255, THUMB) for f in fl])
        maps[name] = per
    return ex, maps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--cfg", default=None, help="defaults to the dataset's XTinyU-Net width (2d_tiny8, 2d_tiny16 for BraTS)")
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
    layers = profile(net, batches, lm)
    eff = effects(net, batches, lm)
    circ = circuits(net, batches)
    ex, maps = examples(net, slices, stride, lm, device)

    short = SHORT.get(a.dataset, a.dataset)
    drops = {}
    if COMPACT.exists():
        comp = json.load(open(COMPACT)).get(short, {}).get(str(w), {})
        drops = {L["label"]: L.get("drop") for L in comp.get("layers", [])}
    first = conv_blocks(net)[0][0][1].conv.weight.detach().cpu()
    class_list = [n for n, _ in class_masks(torch.zeros(1, dtype=torch.long), lm)]
    filters = []
    for L in layers:
        dl = drops.get(L["label"])
        for c in range(L["C"]):
            f = dict(active=L["active"][c], iou={k: v[c] for k, v in L["iou"].items()},
                     mean={k: v[c] for k, v in L["mean"].items()}, mean_fg=L["mean_fg"][c])
            kind, gain = role(f, L["base"], class_list)
            filters.append(dict(block=L["label"], part=L["part"], level=L["level"], c=c, role=kind, gain=gain,
                                drop=None if dl is None else dl[c], active=round(f["active"], 3),
                                ratio=round(in_out_ratio(f), 3),
                                iou={k: round(v, 3) for k, v in f["iou"].items()},
                                d_in=round(eff[L["name"]][c][0], 4), d_out=round(eff[L["name"]][c][1], 4),
                                effect=effect(*eff[L["name"]][c]),
                                maps=maps[L["name"]][c]))
            filters[-1]["text"] = describe(dict(filters[-1], classes=class_list), L["base"],
                                           first[c].numpy() if L is layers[0] else None)
    blocks_ = conv_blocks(net)
    label_of = {n: block_label(mt) for (n, _), mt in zip(*blocks_)}
    roles = {f"{f['block']}·{f['c']}": f["role"] for f in filters}
    idx = {f"{f['block']}·{f['c']}": i for i, f in enumerate(filters)}
    ch_names = list(json.load(open(Path("data/nnUNet_preprocessed") / a.dataset / "dataset.json"))["channel_names"].values())
    for f in filters:
        name = next(n for n, l in label_of.items() if l == f["block"])
        items, text = built_from(name, f["c"], circ, label_of, roles, ch_names)
        for it in items:
            it["i"] = idx.get(it["src"])
        f["inputs"] = items
        f["built"] = text
    out = dict(dataset=short, cfg=cfg, n_slices=len(slices),
               concepts=list(layers[0]["iou"].keys()),
               base={L["label"]: {k: round(v, 3) for k, v in L["base"].items()} for L in layers},
               rf={L["label"]: rf for L, rf in zip(layers, receptive_fields(net))},
               blocks=[dict(label=L["label"], part=L["part"], level=L["level"], C=L["C"]) for L in layers],
               first_kernels=first.numpy().round(3).tolist(), examples=ex, filters=filters)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / f"{a.dataset}.json"
    json.dump(out, open(path, "w"))
    counts = {}
    for f in filters:
        counts[f["role"]] = counts.get(f["role"], 0) + 1
    print(short, cfg, len(slices), "slices,", len(filters), "filters:", counts, f"-> {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
