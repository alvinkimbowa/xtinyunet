"""Feature visualization of the filters of a trained (X)TinyU-Net.

For each filter (output channel of a conv block, read at the raw conv output, before InstanceNorm):

- optimized images: a batch of input images optimized so the neuron at the centre of the filter's
  map fires as strongly as possible. The image lives in a 1/f-scaled Fourier basis, is z-scored every
  step and squashed by a sigmoid into the 0.5th-99.5th percentile range of real preprocessed pixels,
  so neither overall contrast nor single pixels can exceed what real images contain. It is randomly shifted
  (and, for receptive fields of 31 px and more, scaled and rotated) every step, with shifts of 1/8 of
  the neuron's receptive field (none below 8 px, at most 16 px). Several input channels are decorrelated
  with the Cholesky factor of the real channel covariance. A Gram-matrix diversity term over 3x3 pixel patches inside the
  receptive field pushes the batch apart.
- dataset examples: every fold-0 validation image is passed through the model whole; the images are
  ranked by the filter's highest value anywhere in its map, and for the top ones the effective
  receptive field of the neuron at that position (the box holding 90% of the input gradient of that
  neuron) is cropped for display.

Results accumulate in results/analysis/feature_vis/<dataset>.json; later runs optimize filters not
yet present.
"""

import argparse
import base64
import io
import json
import math
import random
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from filter_redundancy import build_model, conv_blocks, load_val_slices, make_batches, total_stride
from filter_roles import SHORT, XTINY_WIDTH, block_label, receptive_fields

OUT_ROOT = Path("results/analysis/feature_vis")
CROP_PX = 96
THUMB_PX = 160


class Stop(Exception):
    pass


def run_to(net, conv, x, stop=True):
    """Forward x until `conv`; return its output and input."""
    got = {}

    def hook(m, i, o):
        got["out"], got["inp"] = o, i[0]
        if stop:
            raise Stop

    h = conv.register_forward_hook(hook)
    try:
        net(x)
    except Stop:
        pass
    finally:
        h.remove()
    return got["out"], got["inp"]


def encode(im):
    """Base64 image: PNG for crops up to 24 px, where JPEG blocks would show, JPEG otherwise."""
    buf = io.BytesIO()
    if max(im.size) <= 24:
        im.save(buf, "PNG", optimize=True)
        kind = "png"
    else:
        im.save(buf, "JPEG", quality=85)
        kind = "jpeg"
    return f"data:image/{kind};base64," + base64.b64encode(buf.getvalue()).decode()


def gray(a, lo=None, hi=None):
    """Map a 2D (or CxHxW, averaged) tensor to uint8 using its 1st-99th percentiles."""
    a = a.float()
    if a.dim() == 3:
        a = a.mean(0)
    if lo is None:
        lo, hi = torch.quantile(a.flatten(), 0.01), torch.quantile(a.flatten(), 0.99)
    return ((a - lo) / (hi - lo + 1e-8)).clamp(0, 1).mul(255).byte()


DISPLAY = {"channel": None}


def disp(img):
    """uint8 display of img[C,H,W]: RGB for 3 channels, else the channel DISPLAY["channel"] (or the mean) in gray.

    Percentiles are shared across colour channels so their balance is kept.
    """
    img = img.float()
    if img.shape[0] == 3:
        lo, hi = torch.quantile(img.flatten(), 0.01), torch.quantile(img.flatten(), 0.99)
        return ((img - lo) / (hi - lo + 1e-8)).clamp(0, 1).mul(255).byte().permute(1, 2, 0)
    if DISPLAY["channel"] is not None and img.shape[0] > 1:
        return gray(img[DISPLAY["channel"]])
    return gray(img)


def crop_png(img, box):
    """Crop img[C,H,W] to box (y0, x0, y1, x1); crops larger than CROP_PX are downsampled, smaller ones kept at native size."""
    y0, x0, y1, x1 = box
    c = disp(img[:, y0:y1, x0:x1]).cpu()
    h, w = c.shape[:2]
    s = CROP_PX / max(h, w)
    im = Image.fromarray(np.ascontiguousarray(c.numpy()), "RGB" if c.dim() == 3 else "L")
    if s < 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BOX)
    return encode(im)


def erf_box(g, frac=0.9):
    """Box holding `frac` of the absolute gradient mass, cut per axis from both ends."""
    g = g.abs().sum(0)
    tail = (1 - frac) / 4
    out = []
    for marg in (g.sum(1), g.sum(0)):
        cs = torch.cumsum(marg, 0) / marg.sum().clamp(min=1e-12)
        lo = int((cs < tail).sum())
        hi = int((cs < 1 - tail).sum()) + 1
        out += [lo, max(hi, lo + 1)]
    return out[0], out[2], out[1], out[3]


class FrozenNormStats:
    """Inside this context InstanceNorm layers treat their per-image mean and variance as constants.

    Each InstanceNorm output depends on its channel's statistics over the whole image, so without this
    every neuron after the first norm has a small gradient on every input pixel; with it the gradient
    is confined to the local receptive field.
    """

    def __init__(self, net):
        self.mods = [m for m in net.modules() if isinstance(m, torch.nn.modules.instancenorm._InstanceNorm)]

    @staticmethod
    def forward(m, x):
        mu = x.mean((2, 3), keepdim=True).detach()
        var = x.var((2, 3), keepdim=True, unbiased=False).detach()
        y = (x - mu) / torch.sqrt(var + m.eps)
        if m.affine:
            y = y * m.weight.view(1, -1, 1, 1) + m.bias.view(1, -1, 1, 1)
        return y

    def __enter__(self):
        for m in self.mods:
            m.forward = (lambda x, m=m: self.forward(m, x))
        return self

    def __exit__(self, *exc):
        for m in self.mods:
            del m.forward


def neuron_grad(net, conv, x, c, i, j, local=True):
    """Gradient of the raw conv output at channel c, position (i, j) with respect to the input x.

    local=True holds InstanceNorm statistics fixed (see FrozenNormStats).
    """
    x = x.detach().clone().requires_grad_(True)
    if local:
        with FrozenNormStats(net):
            out, _ = run_to(net, conv, x)
    else:
        out, _ = run_to(net, conv, x)
    act = out[0, c, i, j]
    (g,) = torch.autograd.grad(act, x)
    return act.item(), g[0]


def outline(seg):
    """Boolean edge of the foreground of seg[H,W]."""
    fg = (seg > 0).float()[None, None]
    er = -F.max_pool2d(-fg, 3, 1, 1)
    return (fg - er)[0, 0] > 0


def thumb(img, seg):
    """Image with the object edge drawn, as an RGB image of at most THUMB_PX."""
    g = disp(img).cpu()
    rgb = g.clone() if g.dim() == 3 else torch.stack([g, g, g], -1)
    edge = outline(seg).cpu()
    rgb[edge] = torch.tensor([240, 140, 40], dtype=torch.uint8)
    h, w = g.shape[:2]
    s = min(1.0, THUMB_PX / max(h, w))
    im = Image.fromarray(rgb.numpy(), "RGB")
    if s < 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return encode(im)


@torch.no_grad()
def scan(net, blocks, slices, stride, device):
    """Per block: each image's per-channel max of the raw conv output and its position, and per-channel std."""
    stats = {name: dict(mx=[], pos=[], s=0.0, s2=0.0, n=0) for name, _ in blocks}
    for img, seg in slices:
        (x, _, _), = make_batches([(img, seg)], stride, 1, device)
        h0, w0 = seg.shape
        cap = {}
        hs = [blk.conv.register_forward_hook(lambda m, i, o, name=name: cap.__setitem__(name, o[0]))
              for name, blk in blocks]
        net(x)
        for h in hs:
            h.remove()
        for name, _ in blocks:
            f = cap[name]
            sh, sw = f.shape[-2] / x.shape[-2], f.shape[-1] / x.shape[-1]
            f = f[:, :max(1, math.ceil(h0 * sh)), :max(1, math.ceil(w0 * sw))]
            C, H, W = f.shape
            flat = f.reshape(C, -1)
            v, k = flat.max(1)
            st = stats[name]
            st["mx"].append(v.cpu())
            st["pos"].append(torch.stack([k // W, k % W], 1).cpu())
            st["s"] += flat.sum(1).double().cpu()
            st["s2"] += (flat.double() ** 2).sum(1).cpu()
            st["n"] += flat.shape[1]
    for st in stats.values():
        st["mx"] = torch.stack(st["mx"])
        st["pos"] = torch.stack(st["pos"])
        mean = st["s"] / st["n"]
        st["std"] = (st["s2"] / st["n"] - mean ** 2).clamp(min=1e-12).sqrt().float()
    return stats


def dataset_examples(net, blocks, labels, slices, stride, device, stats, keep=None, k=4):
    out = {}
    for name, blk in blocks:
        st = stats[name]
        for c in range(st["mx"].shape[1]):
            if keep is not None and c not in keep[name]:
                continue
            order = torch.argsort(st["mx"][:, c], descending=True)[:k].tolist()
            top = []
            for n in order:
                img, seg = slices[n]
                (x, _, _), = make_batches([(img, seg)], stride, 1, device)
                i, j = st["pos"][n, c].tolist()
                act, g = neuron_grad(net, blk.conv, x, c, i, j)
                h0, w0 = seg.shape
                g = g[:, :h0, :w0]
                box = erf_box(g)
                top.append(dict(img=n, act=round(act, 4), box=list(box),
                                crop=crop_png(torch.from_numpy(img).to(device), box)))
            out[f"{labels[name]}·{c}"] = dict(top=top, max=round(float(st["mx"][:, c].max()), 4),
                                               std=round(float(st["std"][c]), 4))
    return out


class FourierImage(torch.nn.Module):
    """Batch of images parameterized by rfft2 coefficients scaled by 1/frequency.

    For several channels the optimized channels are decorrelated: they are mixed by `mix`, the Cholesky
    factor of the real pixels' channel covariance, so the image's channel correlations match real data.
    """

    def __init__(self, b, ch, h, w, lo, hi, device, mix=None, sd=0.01):
        super().__init__()
        fy = torch.fft.fftfreq(h)[:, None]
        fx = torch.fft.rfftfreq(w)[None]
        freq = torch.sqrt(fx ** 2 + fy ** 2)
        self.register_buffer("scale", 1.0 / freq.clamp(min=1.0 / max(h, w)))
        self.spec = torch.nn.Parameter(torch.randn(b, ch, h, w // 2 + 1, 2) * sd)
        self.h, self.w = h, w
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("mix", torch.tensor(mix if mix is not None else np.eye(ch), dtype=torch.float32))
        self.to(device)

    def forward(self):
        x = torch.fft.irfft2(torch.view_as_complex(self.spec) * self.scale, s=(self.h, self.w))
        x = torch.einsum("ij,bjhw->bihw", self.mix, x)
        m, s = x.mean((-2, -1), keepdim=True), x.std((-2, -1), keepdim=True)
        return self.lo + (self.hi - self.lo) * torch.sigmoid((x - m) / (s + 1e-8))


def transform(x, jit, affine):
    """Pad, shift by up to jit, optionally scale/rotate, shift by up to jit//2, crop back (per image)."""
    B, C, H, W = x.shape
    P = 2 * jit + 2
    xp = F.pad(x, (P, P, P, P))
    if affine:
        ang = torch.tensor([math.radians(random.randint(-5, 5)) for _ in range(B)], device=x.device)
        sc = torch.tensor([random.choice([1.0, 0.975, 1.025, 0.95, 1.05]) for _ in range(B)], device=x.device)
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(cos)], 1),
                             torch.stack([sin, cos, torch.zeros_like(cos)], 1)], 1)
        grid = F.affine_grid(theta, xp.shape, align_corners=False)
        xp = F.grid_sample(xp, grid, align_corners=False, padding_mode="zeros")
    j2 = jit // 2
    outs = []
    for b in range(B):
        dy = random.randint(-jit, jit) + random.randint(-j2, j2)
        dx = random.randint(-jit, jit) + random.randint(-j2, j2)
        outs.append(xp[b, :, P + dy:P + dy + H, P + dx:P + dx + W])
    return torch.stack(outs)


def patch_gram(x, box):
    """Gram matrix of 3x3 pixel patches inside box, one per image, flattened and unit-normalized."""
    y0, x0, y1, x1 = box
    r = x[:, :, max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1]
    p = F.unfold(r, 3)
    G = p @ p.transpose(1, 2) / p.shape[-1]
    G = G.flatten(1)
    return F.normalize(G, dim=1)


def optimize(net, conv, c, shape, rf, std, rng, mix, device, b, steps, lam, lr, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    ch, H, W = shape
    probe = torch.randn(1, ch, H, W, device=device)
    with torch.no_grad():
        out, _ = run_to(net, conv, probe)
    ci, cj = out.shape[-2] // 2, out.shape[-1] // 2
    _, g = neuron_grad(net, conv, probe, c, ci, cj)
    nz = (g.abs().sum(0) > 0).nonzero()
    rf_box = (int(nz[:, 0].min()), int(nz[:, 1].min()), int(nz[:, 0].max()) + 1, int(nz[:, 1].max()) + 1)
    jit = int(min(16, rf // 8))
    affine = rf >= 31
    param = FourierImage(b, ch, H, W, rng[0], rng[1], device, mix)
    opt = torch.optim.Adam(param.parameters(), lr=lr)
    use_div = lam > 0 and b > 1 and rf > 3
    for _ in range(steps):
        opt.zero_grad()
        x = param()
        out, _ = run_to(net, conv, transform(x, jit, affine))
        act = out[:, c, ci, cj] / std
        loss = -act.mean()
        if use_div:
            G = patch_gram(x, rf_box)
            sim = G @ G.T
            loss = loss + lam * (sim.sum() - sim.diagonal().sum()) / (b * (b - 1))
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = param()
        out, _ = run_to(net, conv, x)
        acts = out[:, c, ci, cj].tolist()
    imgs, boxes = [], []
    for k in range(b):
        _, g = neuron_grad(net, conv, x[k:k + 1], c, ci, cj)
        box = erf_box(g)
        boxes.append(box)
        imgs.append(crop_png(x[k].detach(), box))
    y0 = min(bx[0] for bx in boxes); x0 = min(bx[1] for bx in boxes)
    y1 = max(bx[2] for bx in boxes); x1 = max(bx[3] for bx in boxes)
    v = x[:, :, y0:y1, x0:x1].flatten(1)
    v = F.normalize(v - v.mean(1, keepdim=True), dim=1)
    cor = v @ v.T
    agree = float((cor.sum() - cor.diagonal().sum()) / (b * (b - 1))) if b > 1 else 1.0
    return dict(opt=[dict(png=p, act=round(a, 4), box=list(bx)) for p, a, bx in zip(imgs, acts, boxes)],
                agree=round(agree, 3), jitter=jit, affine=affine, rf_box=list(rf_box), steps=steps, lam=lam)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--cfg", default=None, help="defaults to the dataset's XTinyU-Net width (2d_tiny8, 2d_tiny16 for BraTS)")
    p.add_argument("--trainer", default="nnUNetTrainer_100epochs")
    p.add_argument("--plans", default="nnUNetPlans")
    p.add_argument("--max_slices", type=int, default=1000)
    p.add_argument("--n", type=int, default=24, help="number of filters to optimize in this run (0: dataset examples only)")
    p.add_argument("--filters", default="", help="comma-separated filter ids such as E0.0·3, optimized instead of a random sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per_block", type=int, default=0, help="if > 0, keep only this many random filters per block")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=2560)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lam", type=float, default=1.0, help="weight of the diversity term")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    w = XTINY_WIDTH.get(a.dataset, 8)
    cfg = a.cfg or f"2d_tiny{w}"
    device = torch.device(a.device)
    net, lm, cm = build_model(a.dataset, cfg, a.trainer, a.plans, 0, device)
    net.requires_grad_(False)
    blocks, meta = conv_blocks(net)
    labels = {m["name"]: block_label(m) for m in meta}
    rfs = dict(zip([n for n, _ in blocks], receptive_fields(net)))
    stride = total_stride(cm)
    patch = list(cm.patch_size)
    out_path = Path(a.out) if a.out else OUT_ROOT / f"{a.dataset}.json"
    res = json.load(open(out_path)) if out_path.exists() else None

    ch_names = list(json.load(open(Path("data/nnUNet_preprocessed") / a.dataset / "dataset.json"))["channel_names"].values())
    if len(ch_names) not in (1, 3):
        DISPLAY["channel"] = next((i for i, n in enumerate(ch_names) if n.lower() == "flair"), 0)
    t0 = time.time()
    if res is None:
        slices = load_val_slices(a.dataset, a.plans, 0, a.max_slices, 0)
        stats = scan(net, blocks, slices, stride, device)
        keep = None
        if a.per_block > 0:
            r = random.Random(a.seed)
            keep = {n: set(r.sample(range(blk.conv.out_channels), min(a.per_block, blk.conv.out_channels))) for n, blk in blocks}
        ex = dataset_examples(net, blocks, labels, slices, stride, device, stats, keep)
        px = np.concatenate([s[0].reshape(s[0].shape[0], -1) for s in slices], 1)
        px = px[:, np.random.default_rng(0).choice(px.shape[1], min(px.shape[1], 2_000_000), replace=False)]
        rng = [np.quantile(px, 0.005, axis=1).tolist(), np.quantile(px, 0.995, axis=1).tolist()]
        mix = np.linalg.cholesky(np.atleast_2d(np.cov(px)) + 1e-6 * np.eye(px.shape[0])).tolist()
        used = sorted({t["img"] for v in ex.values() for t in v["top"]})
        remap = {n: i for i, n in enumerate(used)}
        for v in ex.values():
            for t in v["top"]:
                t["img"] = remap[t["img"]]
        res = dict(dataset=SHORT.get(a.dataset, a.dataset), cfg=cfg, patch=patch, range=rng, mix=mix,
                   channels=ch_names, display_channel=DISPLAY["channel"],
                   rf={labels[n]: r for n, r in rfs.items()},
                   images=[dict(h=int(slices[n][1].shape[0]), w=int(slices[n][1].shape[1]),
                                png=thumb(torch.from_numpy(slices[n][0]), torch.from_numpy(slices[n][1]))) for n in used],
                   filters={k: dict(v, block=k.split("·")[0], c=int(k.split("·")[1])) for k, v in ex.items()})
        print(f"dataset examples for {len(ex)} filters from {len(slices)} images in {time.time() - t0:.0f}s", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out_path, "w"))

    by_label = {labels[n]: (n, blk) for n, blk in blocks}
    if a.filters:
        todo = [f.strip() for f in a.filters.split(",") if f.strip()]
    else:
        pending = [k for k, v in res["filters"].items() if "opt" not in v]
        random.Random(a.seed).shuffle(pending)
        todo = pending[:a.n]
    ch = len(ch_names)
    for i, key in enumerate(todo):
        t1 = time.time()
        lab, c = key.split("·")
        name, blk = by_label[lab]
        f = res["filters"][key]
        r = optimize(net, blk.conv, int(c), (ch, *patch), rfs[name], max(f["std"], 1e-6), res["range"], res.get("mix"), device,
                     a.batch, a.steps, a.lam, a.lr, a.seed * 1000 + zlib.crc32(key.encode()) % 1000)
        f.update(r)
        json.dump(res, open(out_path, "w"))
        print(f"[{i + 1}/{len(todo)}] {key}: act/real-max {[round(o['act'] / max(f['max'], 1e-6), 2) for o in r['opt']]}"
              f" agree {r['agree']} ({time.time() - t1:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
