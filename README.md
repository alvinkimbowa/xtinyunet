# XTinyU-Net: Training-Free U-Net Scaling via Initialization-Time Sensitivity

## XTinyU-Net workflow:

1. set up a Python environment,
2. prepare a dataset in nnU-Net format,
3. run standard nnU-Net preprocessing,
4. derive XTinyU-Net configuration from the generated nnU-Net plans,
5. train with nnU-Net using the XTinyU-Net configuration,
6. run nnU-Net inference with the XTinyU-Net configuration.

## 1. Set Up Environment

Create a project environment with `uv`:

```bash
uv init
```

Install PyTorch. Choose the wheel index that matches your CUDA setup. For CUDA 12.6:

```bash
uv add torch torchvision --index=pytorch-cu126=https://download.pytorch.org/whl/cu126
```

Install nnU-Net v2:

```bash
uv add nnunetv2==2.7.0
```

Set the standard nnU-Net environment variables:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

## 2. Prepare Dataset

Prepare your dataset following the official nnU-Net v2 dataset format:

```text
nnUNet_raw/
  DatasetXXX_MyDataset/
    dataset.json
    imagesTr/
    labelsTr/
    imagesTs/   # optional, for inference/test data
```

Use the normal nnU-Net naming conventions for case IDs, image channels, labels, and file endings.

## 3. Run nnU-Net Preprocessing

Run nnU-Net planning and preprocessing:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity -c 2d
```

This writes the preprocessed dataset and generates an `nnUNetPlans.json` file under:

```text
$nnUNet_preprocessed/DatasetXXX_MyDataset/
```

## 4. Get XTinyU-Net Config

```bash
uv run python src/get_xtinyunet_config.py --plans /path/to/nnUNetPlans.json
```

Example:

```bash
uv run python src/get_xtinyunet_config.py \
  --plans "$nnUNet_preprocessed/Dataset300_MyDataset/nnUNetPlans.json"
```

This returns the XTinyU-Net config.

## 5. Train With nnU-Net

Train normally with nnU-Net, but use the selected XTinyU-Net config:

```bash
nnUNetv2_train DATASET_ID XTINY_CONFIG 0 -p nnUNetPlans -tr nnUNetTrainer --val_on_end
```

Example:

```bash
nnUNetv2_train 300 2d_xtiny8 0 -p nnUNetPlans -tr nnUNetTrainer --val_on_end
```

## 6. Run Inference

Run nnU-Net inference normally, using the same XTinyU-Net config and plans used for training:

```bash
nnUNetv2_predict \
  -i /path/to/imagesTs \
  -o /path/to/output_predictions \
  -d DATASET_ID \
  -c XTINY_CONFIG \
  -f 0 \
  -p nnUNetPlans \
  -tr nnUNetTrainer
```

Example:

```bash
nnUNetv2_predict \
  -i "$nnUNet_raw/Dataset300_MyDataset/imagesTs" \
  -o predictions/Dataset300_MyDataset_xtiny \
  -d 300 \
  -c 2d_xtiny8 \
  -f 0 \
  -p nnUNetPlans \
  -tr nnUNetTrainer
```

## Notes

- The selected XTinyU-Net config must be used consistently for training and inference.
- The generated nnU-Net plans from preprocessing are the source of truth for patch size, spacing, and dataset-specific preprocessing.
- Public cleanup is in progress; scripts and command names may be simplified before release.

## Citation

If you use this work, please cite:

```bibtex
@misc{kimbowa_xtinyu-net_2026,
	title = {{XTinyU}-{Net}: {Training}-{Free} {U}-{Net} {Scaling} via {Initialization}-{Time} {Sensitivity}},
	shorttitle = {{XTinyU}-{Net}},
	url = {http://arxiv.org/abs/2605.09639},
	doi = {10.48550/arXiv.2605.09639},
	urldate = {2026-05-16},
	publisher = {arXiv},
	author = {Kimbowa, Alvin and Heidari, Moein and Liu, David and Hacihaliloglu, Ilker},
	month = may,
	year = {2026},
	note = {arXiv:2605.09639 [eess.IV]},
	keywords = {Computer Science - Computer Vision and Pattern Recognition, Electrical Engineering and Systems Science - Image and Video Processing}
}
```
