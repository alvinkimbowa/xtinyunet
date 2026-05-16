import argparse
from contextlib import contextmanager
import importlib
import json
import os
import random
import re
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import nibabel as nib
from tqdm import tqdm
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder

from at_init_metrics import (
    swap_score,
    ncd_swap_score,
    ncd_naswot_score,
    collect_ncd_swap_packed_codes,
    collect_ncd_naswot_packed_codes,
    naswot_score,
    collect_naswot_packed_codes,
    naswot_from_packed,
    collect_swap_packed_codes,
    swap_from_packed,
    az_nas_score,
    jacobian_score,
)

nnUNet_raw = os.environ['nnUNet_raw']
nnUNet_preprocessed = os.environ['nnUNet_preprocessed']


@contextmanager
def temporary_trainer_results_dir():
    trainer_module = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
    old_results = trainer_module.nnUNet_results
    with tempfile.TemporaryDirectory(prefix="xtiny_score_nnunet_results_") as tmp_dir:
        trainer_module.nnUNet_results = tmp_dir
        try:
            yield
        finally:
            trainer_module.nnUNet_results = old_results


def make_predictor(device):
    return nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Compute NASWOT score for an nnUNet model")
    parser.add_argument("--train_dataset_id", type=int, required=True)
    parser.add_argument("--plans", type=str, required=True)
    parser.add_argument("--trainer", type=str, required=True)
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", type=str, default="Tr", choices=["Tr", "Ts"])
    parser.add_argument("--split_type", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=str, default="all", help="number of samples to score")
    parser.add_argument("--seed", type=int, default=369)
    parser.add_argument("--out_dir", type=str, default="results/nas_metrics")
    parser.add_argument("--save_batch_jacobian", action="store_true",
                        help="save per-batch jacobian with image ids")
    parser.add_argument("--ncd_alpha", type=float, default=0.95,
                        help="SAM masking probability alpha for NCD metrics")
    parser.add_argument("--save_naswot_codes", action="store_true",
                        help="save packed naswot codes across batches and compute a single global naswot")
    parser.add_argument("--save_swap_codes", action="store_true",
                        help="save packed swap codes across batches and compute a single global swap")
    parser.add_argument("--save_ncd_swap_codes", action="store_true",
                        help="save packed NCD SWAP codes across batches and compute a single global NCD SWAP")
    parser.add_argument("--save_ncd_naswot_codes", action="store_true",
                        help="save packed NCD NASWOT codes across batches and compute a single global NCD NASWOT")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["jacobian"],
        help=(
            "list of metrics to compute"
        ),
        choices=[
            "jacobian",
            "naswot",
            "swap",
            "ncd_naswot",
            "ncd_swap",
            "az_nas",
        ],
    )
    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_num_samples(num_samples):
    try:
        num_samples = int(num_samples)
    except (TypeError, ValueError):
        return None, "all"
    if num_samples < 0:
        return None, "all"
    return num_samples, str(num_samples)


def get_dataset_name(dataset_id):
    dataset_id = str(dataset_id).zfill(3)
    dataset_name = [
        d for d in os.listdir(nnUNet_raw)
        if d.startswith(f"Dataset{dataset_id}") and os.path.isdir(join(nnUNet_raw, d))
    ]
    if len(dataset_name) != 1:
        raise RuntimeError(f"Found {len(dataset_name)} datasets with id {dataset_id}, expected 1")
    return dataset_name[0]


def get_dataset_json(dataset_id):
    dataset_name = get_dataset_name(dataset_id)
    with open(join(nnUNet_raw, dataset_name, "dataset.json"), "r") as f:
        dataset_json = json.load(f)
    return dataset_json


def get_xtiny_configs(dataset_name, plans):
    with open(join(nnUNet_preprocessed, dataset_name, f"{plans}.json"), "r") as f:
        plans_json = json.load(f)
    configs = [
        cfg for cfg in plans_json["configurations"]
        if re.fullmatch(r"2d_xtiny\d+", cfg)
    ]
    return sorted(configs, key=lambda cfg: int(cfg.replace("2d_xtiny", "")), reverse=True)

def get_num_input_channels(dataset_name):
    with open(join(nnUNet_raw, dataset_name, "dataset.json"), "r") as f:
        dataset_json = json.load(f)
    return len(dataset_json["channel_names"])

def get_target(meta, dataset_name, dataset_json, slice_idx=None):
    file_ending = dataset_json["file_ending"]
    target_path = join(nnUNet_raw, dataset_name, "labelsTr", f"{os.path.basename(meta)}{file_ending}")

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target label not found: {target_path}")

    if file_ending in (".nii.gz", ".nii"):
        arr = np.asarray(nib.load(target_path).get_fdata())
        if arr.ndim == 3:
            z = arr.shape[-1] // 2 if slice_idx is None else int(slice_idx)
            z = max(0, min(z, arr.shape[-1] - 1))
            arr = arr[..., z]
        elif arr.ndim > 3:
            arr = np.squeeze(arr)
            if arr.ndim == 3:
                z = arr.shape[-1] // 2 if slice_idx is None else int(slice_idx)
                z = max(0, min(z, arr.shape[-1] - 1))
                arr = arr[..., z]
    else:
        arr = np.array(Image.open(target_path))

    target = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    return target

def load_nnunet_model(train_dataset_id, plans, trainer, cfg, fold, device, num_cases=None):
    fold = fold if fold == "all" else int(fold)
    dataset_name = get_dataset_name(train_dataset_id)
    dataset_json = get_dataset_json(train_dataset_id)
    os.environ["nnUNet_n_proc_DA"] = "0"    # Use a single process for data augmentation to avoid wierd errors
    num_train = get_dataset_json(train_dataset_id)["numTraining"]
    predictor = make_predictor(device)
    with temporary_trainer_results_dir():
        nnunet_trainer = get_trainer_from_args(dataset_name, cfg, fold, trainer, plans, device=device)
        nnunet_trainer.enable_deep_supervision = False
        nnunet_trainer.initialize()
        model = nnunet_trainer.network.to(device)
        loss_fn = nnunet_trainer.loss
        mini_batch_size = nnunet_trainer.configuration_manager.batch_size
        patch_size = nnunet_trainer.configuration_manager.patch_size
        predictor.plans_manager = nnunet_trainer.plans_manager
        predictor.configuration_manager = nnunet_trainer.configuration_manager
        predictor.dataset_json = dataset_json
        predictor.trainer_name = trainer
        predictor.allowed_mirroring_axes = nnunet_trainer.inference_allowed_mirroring_axes
        predictor.label_manager = nnunet_trainer.label_manager
    
    all_cases = create_lists_from_splitted_dataset_folder(
        join(nnUNet_raw, dataset_name, "imagesTr"),
        dataset_json["file_ending"],
    )
    random.shuffle(all_cases)
    if num_cases is None or num_cases == -1:
        sample_cases = all_cases
    else:
        sample_cases = all_cases[:num_cases] 
    input_lists, output_filenames, seg_from_prev_stage_files = predictor._manage_input_and_output_lists(
        sample_cases,
        "tmp",
        None,
        True,
        0,
        1,
        False,
    )
    data_loader = predictor._internal_get_data_iterator_from_lists_of_filenames(
        input_lists,
        seg_from_prev_stage_files,
        output_filenames,
        2,
    )
    
    return model, dataset_name, loss_fn, data_loader, num_train, mini_batch_size, patch_size


def center_crop_or_pad(x: torch.Tensor, patch_size: tuple[int, int], pad_value: float = 0):
    """
    x: (N, C, H, W) or (C, H, W) or (H, W)
    patch_size: (H, W)
    returns: torch.Tensor
    """
    if x.ndim == 4:   # N, C, H, W
        h, w = x.shape[2:]
    elif x.ndim == 3: # C, H, W
        h, w = x.shape[1:]
    elif x.ndim == 2: # H, W
        h, w = x.shape
    else:
        raise ValueError(f"Unsupported shape: {x.shape}")

    # pad (center)
    pad_h = max(patch_size[0] - h, 0)
    pad_w = max(patch_size[1] - w, 0)

    # split padding: extra goes to the right/bottom
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        # F.pad uses (left, right, top, bottom) for 2D spatial
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=pad_value)

    # crop (center)
    if x.ndim == 4:
        h, w = x.shape[2:]
        hs = (h - patch_size[0]) // 2
        ws = (w - patch_size[1]) // 2
        return x[:, :, hs:hs+patch_size[0], ws:ws+patch_size[1]]
    elif x.ndim == 3:
        h, w = x.shape[1:]
        hs = (h - patch_size[0]) // 2
        ws = (w - patch_size[1]) // 2
        return x[:, hs:hs+patch_size[0], ws:ws+patch_size[1]]
    else:
        h, w = x.shape
        hs = (h - patch_size[0]) // 2
        ws = (w - patch_size[1]) // 2
        return x[hs:hs+patch_size[0], ws:ws+patch_size[1]]


def score_config(args, cfg, device, metric_set):
    num_cases, num_samples = get_num_samples(args.num_samples)
    model, dataset_name, loss_fn, data_loader, num_train, mini_batch_size, patch_size = load_nnunet_model(
        args.train_dataset_id,
        args.plans,
        args.trainer,
        cfg,
        args.fold,
        device,
        num_cases=num_cases,
    )
    
    print("num_train", num_train)
    print("num_samples", num_samples)
    print("mini_batch_size", mini_batch_size)
    print("patch_size", patch_size)
    print("cfg", cfg)

    swap_scores = []
    naswot_scores = []
    ncd_naswot_scores = []
    ncd_swap_scores = []
    az_nas_scores = []
    jacobian_scores = []
    batch_rows = []
    packed_codes = {}
    packed_nbits = {}
    swap_packed_codes = {}
    swap_packed_nbits = {}
    ncd_swap_packed_codes = {}
    ncd_swap_packed_nbits = {}
    ncd_naswot_packed_codes = {}
    ncd_naswot_packed_nbits = {}
    ncd_model = None

    for i, batch in tqdm(enumerate(data_loader), total=num_cases if num_cases is not None else num_train):
        imgs = batch['data']
        meta = batch['ofile']
        
        imgs = imgs.permute(1, 0, 2, 3)
        # get the center slice
        center_idx = imgs.shape[0] // 2
        imgs = imgs[center_idx : center_idx + 1]
        imgs = center_crop_or_pad(imgs, patch_size)
        
        dataset_json = get_dataset_json(args.train_dataset_id)
        target = get_target(meta, dataset_name, dataset_json, slice_idx=center_idx)
        target = center_crop_or_pad(target, patch_size)

        x = imgs.float().to(device)
        
        # Compute Jacobian score
        if "jacobian" in metric_set:
            jac = jacobian_score(model, x, loss_fn=loss_fn)
            jacobian_scores.append(jac)
            if args.save_batch_jacobian:
                img_ids = os.path.basename(meta)
                if isinstance(img_ids, (list, tuple)):
                    img_ids = ";".join(img_ids)
                batch_rows.append(
                    {
                        "dataset": dataset_name,
                        "cfg": cfg,
                        "batch": i,
                        "jacobian": jac,
                        "img_ids": img_ids,
                        "seed": args.seed,
                    }
                )
        
        # Compute other NAS metrics
        if "swap" in metric_set:
            if args.save_swap_codes:
                swap_codes, swap_nbits = collect_swap_packed_codes(model, x)
                for k, v in swap_codes.items():
                    swap_packed_codes.setdefault(k, []).append(v)
                    swap_packed_nbits[k] = swap_nbits[k]
            else:
                swap_scores.append(swap_score(model, x))
        if "ncd_swap" in metric_set:
            if args.save_ncd_swap_codes:
                if ncd_model is None:
                    import copy
                    ncd_model = copy.deepcopy(model).to(device)
                    from at_init_metrics import swap_bn_to_ln
                    swap_bn_to_ln(ncd_model)
                ncd_swap_codes, ncd_swap_nbits = collect_ncd_swap_packed_codes(
                    ncd_model, x, alpha=args.ncd_alpha
                )
                for k, v in ncd_swap_codes.items():
                    ncd_swap_packed_codes.setdefault(k, []).append(v)
                    ncd_swap_packed_nbits[k] = ncd_swap_nbits[k]
            else:
                ncd_swap_scores.append(ncd_swap_score(model, x, alpha=args.ncd_alpha))
        if "ncd_naswot" in metric_set:
            if args.save_ncd_naswot_codes:
                if ncd_model is None:
                    import copy
                    ncd_model = copy.deepcopy(model).to(device)
                    from at_init_metrics import swap_bn_to_ln
                    swap_bn_to_ln(ncd_model)
                ncd_nas_codes, ncd_nas_nbits = collect_ncd_naswot_packed_codes(
                    ncd_model, x, alpha=args.ncd_alpha
                )
                for k, v in ncd_nas_codes.items():
                    ncd_naswot_packed_codes.setdefault(k, []).append(v)
                    ncd_naswot_packed_nbits[k] = ncd_nas_nbits[k]
            else:
                ncd_naswot_scores.append(ncd_naswot_score(model, x, alpha=args.ncd_alpha))
        if "naswot" in metric_set:
            if args.save_naswot_codes:
                codes, nbits = collect_naswot_packed_codes(model, x)
                for k, v in codes.items():
                    packed_codes.setdefault(k, []).append(v)
                    packed_nbits[k] = nbits[k]
            else:
                naswot_scores.append(naswot_score(model, x))
        if "az_nas" in metric_set:
            az_nas_scores.append(az_nas_score(model, x, offload_to_cpu=True))
    
    # Aggregate Jacobian score
    if jacobian_scores:
        jac_arr = np.asarray(jacobian_scores, dtype=np.float64)
        jac_arr = jac_arr[np.isfinite(jac_arr)]
        jacobian_avg = float(np.sqrt(np.mean(jac_arr * jac_arr, dtype=np.float64))) if jac_arr.size else float("nan")
    else:
        jacobian_avg = float("nan")
    
    # Aggregate other NAS metrics    
    if args.save_swap_codes:
        merged_swap_codes = {k: np.concatenate(v, axis=0) for k, v in swap_packed_codes.items()}
        swap_codes_path = join(args.out_dir, f"{dataset_name}_{cfg}_K{num_samples}_swap_codes.npz")
        np.savez(swap_codes_path, **{f"{k}__packed": merged_swap_codes[k] for k in merged_swap_codes},
                 **{f"{k}__nbits": np.array(swap_packed_nbits[k]) for k in swap_packed_nbits})
        swap_avg = swap_from_packed(merged_swap_codes, swap_packed_nbits)
    else:
        swap_avg = float(np.nanmean(swap_scores)) if swap_scores else float("nan")
    if args.save_naswot_codes:
        merged_codes = {k: np.concatenate(v, axis=0) for k, v in packed_codes.items()}
        codes_path = join(args.out_dir, f"{dataset_name}_{cfg}_K{num_samples}_naswot_codes.npz")
        np.savez(codes_path, **{f"{k}__packed": merged_codes[k] for k in merged_codes},
                 **{f"{k}__nbits": np.array(packed_nbits[k]) for k in packed_nbits})
        naswot_avg = naswot_from_packed(merged_codes, packed_nbits)
    else:
        naswot_avg = float(np.nanmean(naswot_scores)) if naswot_scores else float("nan")
    if args.save_ncd_naswot_codes:
        merged_ncd_nas = {k: np.concatenate(v, axis=0) for k, v in ncd_naswot_packed_codes.items()}
        ncd_nas_path = join(args.out_dir, f"{dataset_name}_{cfg}_K{num_samples}_ncd_naswot_codes.npz")
        np.savez(ncd_nas_path, **{f"{k}__packed": merged_ncd_nas[k] for k in merged_ncd_nas},
                 **{f"{k}__nbits": np.array(ncd_naswot_packed_nbits[k]) for k in ncd_naswot_packed_nbits})
        ncd_naswot_avg = naswot_from_packed(merged_ncd_nas, ncd_naswot_packed_nbits)
    else:
        ncd_naswot_avg = float(np.nanmean(ncd_naswot_scores)) if ncd_naswot_scores else float("nan")
    if args.save_ncd_swap_codes:
        merged_ncd_swap = {k: np.concatenate(v, axis=0) for k, v in ncd_swap_packed_codes.items()}
        ncd_swap_path = join(args.out_dir, f"{dataset_name}_{cfg}_K{num_samples}_ncd_swap_codes.npz")
        np.savez(ncd_swap_path, **{f"{k}__packed": merged_ncd_swap[k] for k in merged_ncd_swap},
                 **{f"{k}__nbits": np.array(ncd_swap_packed_nbits[k]) for k in ncd_swap_packed_nbits})
        ncd_swap_avg = swap_from_packed(merged_ncd_swap, ncd_swap_packed_nbits)
    else:
        ncd_swap_avg = float(np.nanmean(ncd_swap_scores)) if ncd_swap_scores else float("nan")
    az_nas_avg = float(np.nanmean(az_nas_scores)) if az_nas_scores else float("nan")

    # Compute model parameters
    params = sum(p.numel() for p in model.parameters())

    # Compile and save model NAS metrics results
    line = (
        f"params={params} swap={swap_avg} naswot={naswot_avg} "
        f"ncd_naswot={ncd_naswot_avg} ncd_swap={ncd_swap_avg} "
        f"az_nas={az_nas_avg} jacobian={jacobian_avg}"
    )
    print("\n")
    print(line)
    out_file = join(args.out_dir, f"{dataset_name}_metrics_K{num_samples}_seed{args.seed}.csv")
    print("out_file", out_file)
    need_header = not os.path.exists(out_file) or os.path.getsize(out_file) == 0
    with open(out_file, "a", encoding="utf-8") as f:
        if need_header:
            f.write("cfg,params,jacobian,swap,naswot,ncd_naswot,ncd_swap,az_nas\n")
        f.write(
            f"{cfg},{params},{jacobian_avg},{swap_avg},{naswot_avg},"
            f"{ncd_naswot_avg},{ncd_swap_avg},{az_nas_avg}\n"
        )
    
    if args.save_batch_jacobian and batch_rows:
        batch_path = join(args.out_dir, f"{dataset_name}_batch_jacobian_K{num_samples}.csv")
        need_header = not os.path.exists(batch_path) or os.path.getsize(batch_path) == 0
        with open(batch_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write("dataset,cfg,batch,jacobian,img_ids,seed\n")
            for row in batch_rows:
                f.write(
                    f"{row['dataset']},{row['cfg']},{row['batch']},"
                    f"{row['jacobian']},{row['img_ids']},{row['seed']}\n"
                )

    print("Done!")
    print("--------------------------------------------------\n\n")


def main(args):
    set_seed(args.seed)
    device = torch.device("cpu" if args.gpu < 0 else "cuda")
    metric_set = {m.strip().lower() for m in args.metrics if m.strip()}
    os.makedirs(args.out_dir, exist_ok=True)

    dataset_name = get_dataset_name(args.train_dataset_id)
    cfgs = [args.cfg] if args.cfg is not None else get_xtiny_configs(dataset_name, args.plans)
    if not cfgs:
        raise RuntimeError("No 2d_xtiny configs found. Run src/generate_candidate_configs.py first.")
    for cfg in cfgs:
        score_config(args, cfg, device, metric_set)


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)
