import os
import numpy as np
import torch
import re


def _get_output_tensor(y):
    if isinstance(y, (tuple, list)):
        y = y[0]
    return y


def _scalar_loss_from_output(y):
    y = _get_output_tensor(y)
    return y.float().mean()


def _get_stage_prefixes(model):
    enc_idxs = []
    for name, _ in model.named_modules():
        m = re.search(r"encoder\.stages\.(\d+)", name)
        if m:
            enc_idxs.append(int(m.group(1)))
    if not enc_idxs:
        return None, None
    return min(enc_idxs), max(enc_idxs)


def _sanitize_key(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def collect_naswot_packed_codes(model, x, stage_only=False):
    batch_size = x.size(0)
    codes = {}
    nbits = {}

    enc_first = enc_last = None
    if stage_only:
        enc_first, enc_last = _get_stage_prefixes(model)

    def forward_hook(module, inp, out):
        try:
            name = getattr(module, "_naswot_key", None)
            if name is None:
                return
            xh = _get_output_tensor(out)
            xh = xh.view(batch_size, -1)
            xh = (xh > 0).to(torch.uint8)
            arr = xh.cpu().numpy()
            packed = np.packbits(arr, axis=1)
            key = _sanitize_key(name)
            codes[key] = packed
            nbits[key] = arr.shape[1]
        except Exception:
            pass

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            if stage_only:
                if enc_first is None:
                    continue
                keep = (
                    name.startswith(f"encoder.stages.{enc_first}") or
                    name.startswith(f"encoder.stages.{enc_last}")
                )
                if not keep:
                    continue
            if module.inplace:
                module.inplace = False
            module._naswot_key = name
            handles.append(module.register_forward_hook(forward_hook))

    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()

    return codes, nbits


def collect_swap_packed_codes(model, x):
    batch_size = x.size(0)
    codes = {}
    nbits = {}

    def forward_hook(module, inp, out):
        try:
            name = getattr(module, "_swap_key", None)
            if name is None:
                return
            xh = _get_output_tensor(out)
            xh = xh.view(batch_size, -1)
            xh = (xh > 0).to(torch.uint8)
            arr = xh.cpu().numpy()
            packed = np.packbits(arr, axis=1)
            key = _sanitize_key(name)
            codes[key] = packed
            nbits[key] = arr.shape[1]
        except Exception:
            pass

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            if module.inplace:
                module.inplace = False
            module._swap_key = name
            handles.append(module.register_forward_hook(forward_hook))

    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()

    return codes, nbits


def swap_from_packed(codes_by_layer, nbits_by_layer, chunk_bits=8192):
    seen = set()
    for key, packed in codes_by_layer.items():
        nbits = nbits_by_layer[key]
        for start in range(0, nbits, chunk_bits):
            end = min(nbits, start + chunk_bits)
            byte_start = start // 8
            byte_end = (end + 7) // 8
            sub = packed[:, byte_start:byte_end]
            bits = np.unpackbits(sub, axis=1)
            left = start - byte_start * 8
            right = left + (end - start)
            bits = bits[:, left:right]
            xt = bits.T
            packed_cols = np.packbits(xt, axis=1)
            for row in packed_cols:
                seen.add(row.tobytes())
    return float(len(seen))


def naswot_from_packed(codes_by_layer, nbits_by_layer):
    K_accum = None
    for key, packed in codes_by_layer.items():
        nbits = nbits_by_layer[key]
        x = np.unpackbits(packed, axis=1)[:, :nbits].astype(np.float32)
        K1 = x @ x.T
        K2 = (1.0 - x) @ (1.0 - x.T)
        if K_accum is None:
            K_accum = K1 + K2
        else:
            K_accum += K1 + K2
    if K_accum is None:
        return float("nan")
    _, logdet = np.linalg.slogdet(K_accum)
    return float(logdet)

def _install_swap_hooks(model, batch_size):
    handles = []
    # store unique neuron-wise activation patterns across the batch
    seen = set()

    def forward_hook(module, inp, _out):
        try:
            x = inp[0]
            if x.size(0) != batch_size:
                return  # SWAP upper bound logic assumes fixed batch size

            x = x.view(x.size(0), -1)          # [B, N]
            x = (x > 0).to(torch.uint8)        # uint8 for packing
            xt = x.t().contiguous()            # [N, B] neuron-wise patterns

            # pack each length-B bitvector into bytes so we can hash it
            # packbits works on CPU numpy, so move once per hook
            arr = xt.cpu().numpy()             # uint8 {0,1}, shape [N,B]
            packed = np.packbits(arr, axis=1)  # shape [N, ceil(B/8)]

            for row in packed:
                seen.add(row.tobytes())
        except Exception:
            pass

    # attach to ReLU modules (common) or whatever you used for NASWOT
    for m in model.modules():
        if isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            handles.append(m.register_forward_hook(forward_hook))

    def score():
        return len(seen)

    return handles, score

def apply_sam(x, alpha):
    if alpha <= 0:
        return x
    mask = torch.bernoulli(
        torch.full_like(x, 1.0 - alpha)
    )
    return x * mask

def install_ncd_swap_hooks(model, batch_size, alpha=0.0):
    seen = set()
    handles = []

    def hook(module, inp, _out):
        x = inp[0]
        if x.size(0) != batch_size:
            return

        # flatten + SAM
        x = x.view(batch_size, -1)
        x = apply_sam(x, alpha)

        # binarize
        x = (x > 0).to(torch.uint8)

        # neuron-wise patterns
        xt = x.t().contiguous()   # [N, B]

        # pack bits for hashing
        packed = np.packbits(
            xt.cpu().numpy(), axis=1
        )

        for row in packed:
            seen.add(row.tobytes())

    for m in model.modules():
        if isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            handles.append(m.register_forward_hook(hook))

    def score():
        return len(seen)

    return handles, score


def install_ncd_naswot_hooks(model, batch_size, alpha=0.0):
    K_accum = torch.zeros(batch_size, batch_size)
    handles = []

    def hook(module, inp, _out):
        x = inp[0]
        if x.size(0) != batch_size:
            return

        # flatten + SAM
        x = x.view(batch_size, -1)
        x = apply_sam(x, alpha)

        # binarize
        x = (x > 0).float()

        # NASWOT kernel
        K1 = x @ x.t()
        K2 = (1 - x) @ (1 - x.t())
        K_accum.add_((K1 + K2).cpu())

    for m in model.modules():
        if isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            handles.append(m.register_forward_hook(hook))

    def score():
        # log |K|
        return torch.logdet(K_accum + 1e-6 * torch.eye(batch_size))

    return handles, score


def collect_ncd_swap_packed_codes(model, x, alpha=0.0):
    batch_size = x.size(0)
    codes = {}
    nbits = {}

    def forward_hook(module, inp, _out):
        try:
            name = getattr(module, "_ncd_swap_key", None)
            if name is None:
                return
            xh = inp[0]
            xh = xh.view(batch_size, -1)
            xh = apply_sam(xh, alpha)
            xh = (xh > 0).to(torch.uint8)
            arr = xh.cpu().numpy()
            packed = np.packbits(arr, axis=1)
            key = _sanitize_key(name)
            codes[key] = packed
            nbits[key] = arr.shape[1]
        except Exception:
            pass

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            if module.inplace:
                module.inplace = False
            module._ncd_swap_key = name
            handles.append(module.register_forward_hook(forward_hook))

    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()

    return codes, nbits


def collect_ncd_naswot_packed_codes(model, x, alpha=0.0):
    batch_size = x.size(0)
    codes = {}
    nbits = {}

    def forward_hook(module, inp, _out):
        try:
            name = getattr(module, "_ncd_naswot_key", None)
            if name is None:
                return
            xh = inp[0]
            xh = xh.view(batch_size, -1)
            xh = apply_sam(xh, alpha)
            xh = (xh > 0).to(torch.uint8)
            arr = xh.cpu().numpy()
            packed = np.packbits(arr, axis=1)
            key = _sanitize_key(name)
            codes[key] = packed
            nbits[key] = arr.shape[1]
        except Exception:
            pass

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            if module.inplace:
                module.inplace = False
            module._ncd_naswot_key = name
            handles.append(module.register_forward_hook(forward_hook))

    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()

    return codes, nbits


def ncd_swap_score(model, x, alpha=0.95):
    import copy
    model = copy.deepcopy(model)
    swap_bn_to_ln(model)
    handles, score_fn = install_ncd_swap_hooks(model, x.size(0), alpha=alpha)
    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()
    return float(score_fn())


def ncd_naswot_score(model, x, alpha=0.95):
    import copy
    model = copy.deepcopy(model)
    swap_bn_to_ln(model)
    handles, score_fn = install_ncd_naswot_hooks(model, x.size(0), alpha=alpha)
    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()
    score = score_fn()
    if torch.is_tensor(score):
        score = score.item()
    return float(score)

def swap_bn_to_ln(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.BatchNorm2d):
            ln = torch.nn.LayerNorm(
                module.num_features,
                elementwise_affine=True
            )
            setattr(model, name, ln)
        else:
            swap_bn_to_ln(module)



def _install_naswot_hooks(model, batch_size, stage_only=False, save_codes=False):
    handles = []
    K_accum = np.zeros((batch_size, batch_size), dtype=np.float32)

    def forward_hook(module, inp, out):
        try:
            x = _get_output_tensor(out)
            x = x.view(x.size(0), -1)
            x = (x > 0).float()
            K = x @ x.t()
            K2 = (1.0 - x) @ (1.0 - x.t())
            K_accum[:] = K_accum + K.cpu().numpy() + K2.cpu().numpy()
        except Exception:
            pass

    enc_first = enc_last = None
    if stage_only:
        enc_first, enc_last = _get_stage_prefixes(model)

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            if stage_only:
                if enc_first is None:
                    continue
                keep = (
                    name.startswith(f"encoder.stages.{enc_first}") or
                    name.startswith(f"encoder.stages.{enc_last}")
                )
                if not keep:
                    continue
            if module.inplace:
                module.inplace = False
            handles.append(module.register_forward_hook(forward_hook))

    return handles, K_accum


def swap_score(model, x):
    handles, score_fn = _install_swap_hooks(model, x.size(0))
    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()
    return float(score_fn())


def naswot_score(model, x, stage_only=False):
    handles, K = _install_naswot_hooks(model, x.size(0), stage_only=stage_only)
    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()
    _, logdet = np.linalg.slogdet(K)
    return float(logdet)


def az_nas_score(model, x, offload_to_cpu=True):
    model.zero_grad(set_to_none=True)
    orig_device = next(model.parameters()).device
    if offload_to_cpu and orig_device.type == "cuda":
        model = model.to("cpu")
        x = x.detach().cpu()
    conv_modules = []
    for _name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
            conv_modules.append(module)

    if len(conv_modules) < 2:
        return float("nan")

    # Collect conv features for expressivity/progressivity on CPU to save GPU memory.
    features_cpu = []
    handles = []

    def forward_hook(_module, _inp, out):
        out_t = _get_output_tensor(out)
        if out_t is None:
            return
        if out_t.dim() >= 4:
            features_cpu.append(out_t.detach().cpu())

    for module in conv_modules:
        handles.append(module.register_forward_hook(forward_hook))

    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()

    if len(features_cpu) < 2:
        return float("nan")

    expressivity_scores = []
    for feat in features_cpu:
        c = feat.size(1)
        feat = feat.permute(0, *range(2, feat.dim()), 1).contiguous().view(-1, c)
        m = feat.mean(dim=0, keepdim=True)
        feat = feat - m
        sigma = torch.mm(feat.t(), feat) / (feat.size(0))
        s = torch.linalg.eigvalsh(sigma)
        s_sum = s.sum()
        if not torch.isfinite(s_sum) or s_sum.item() == 0:
            continue
        prob_s = s / (s_sum + 1e-8)
        score = (-prob_s) * torch.log(prob_s + 1e-8)
        score_val = score.sum().item()
        if np.isfinite(score_val):
            expressivity_scores.append(score_val)
    expressivity_scores = np.array(expressivity_scores)
    if expressivity_scores.size < 2:
        return float("nan")
    progressivity = np.min(expressivity_scores[1:] - expressivity_scores[:-1])
    expressivity = np.sum(expressivity_scores)

    # Compute trainability per pair with a fresh forward to avoid keeping all features on GPU.
    scores = []
    for i in reversed(range(1, len(conv_modules))):
        f_in = None
        f_out = None
        handles = []

        def hook_in(_module, _inp, out):
            nonlocal f_in
            out_t = _get_output_tensor(out)
            if out_t is not None and out_t.dim() >= 4:
                f_in = out_t

        def hook_out(_module, _inp, out):
            nonlocal f_out
            out_t = _get_output_tensor(out)
            if out_t is not None and out_t.dim() >= 4:
                f_out = out_t

        handles.append(conv_modules[i - 1].register_forward_hook(hook_in))
        handles.append(conv_modules[i].register_forward_hook(hook_out))

        _ = model(x)

        for h in handles:
            h.remove()

        if f_in is None or f_out is None:
            continue

        if f_out.grad is not None:
            f_out.grad.zero_()
        if f_in.grad is not None:
            f_in.grad.zero_()
        g_out = torch.ones_like(f_out) * 0.5
        g_out = (torch.bernoulli(g_out) - 0.5) * 2
        g_in = torch.autograd.grad(outputs=f_out, inputs=f_in, grad_outputs=g_out, retain_graph=False)[0]
        if g_out.size() == g_in.size() and torch.all(g_in == g_out):
            continue
        else:
            if g_out.dim() >= 4:
                if g_out.size(2) != g_in.size(2) or g_out.size(3) != g_in.size(3):
                    if g_out.dim() == 4:
                        g_in = torch.nn.functional.adaptive_avg_pool2d(
                            g_in, (g_out.size(2), g_out.size(3))
                        )
                    elif g_out.dim() == 5:
                        g_in = torch.nn.functional.adaptive_avg_pool3d(
                            g_in, (g_out.size(2), g_out.size(3), g_out.size(4))
                        )
            g_out = g_out.permute(0, 2, 3, 1).contiguous().view(-1, g_out.size(1))
            g_in = g_in.permute(0, 2, 3, 1).contiguous().view(-1, g_in.size(1))
            mat = torch.mm(g_in.t(), g_out) / (g_out.size(0))
            if mat.size(0) < mat.size(1):
                mat = mat.t()
            s = torch.linalg.svdvals(mat)
            s_max = s.max().item()
            if not np.isfinite(s_max) or s_max <= 0:
                continue
            scores.append(-s_max - 1 / (s_max + 1e-6) + 2)
    if len(scores) == 0:
        return float("nan")
    trainability = np.mean(scores)

    az_score = float(expressivity + progressivity + trainability)
    if np.isnan(az_score):
        az_score = float("nan")

    if offload_to_cpu and orig_device.type == "cuda":
        model = model.to(orig_device)

    return az_score


def jacobian_score(model, x, targets=None, loss_fn=None):
    model.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(True)
    y = model(x)
    y = _get_output_tensor(y)
    loss = y.float().sum()
    grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
    return float(grad.norm().item())
