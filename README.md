# Frequency-guided adaptive computation for efficient single-image super-resolution
---
[[arXiv](https://arxiv.org/abs/2505.06975)] [[PDF](https://arxiv.org/pdf/2505.06975)]

## Dependencies
- Python 3.8
- NVIDIA GPU with CUDA
- PyTorch 1.13.1 and torchvision 0.14.1 ([CUDA installation instructions](https://pytorch.org/get-started/previous-versions/#v1131))

```bash
python -m pip install numpy opencv-python pandas tqdm fast_pytorch_kmeans imageio einops matplotlib timm==0.6.13
```

## Datasets
Please download the DIV2K dataset following [ClassSR](https://github.com/XPixelGroup/ClassSR?tab=readme-ov-file).

## Download Pre-trained Models
The ×4 checkpoints for FSRCNN, CARN, SRResNet, and SwinIR are included in [Pixelwise_SR_experiments](Pixelwise_SR_experiments/).

## Getting Started
### Test
Run the commands from the repository root. Replace `/path/to/test2k` with the directory containing `LR/X4/` and `HR/X4/`. Keep the paired images in the same order when sorted by filename.

**FSRCNN**

```bash
python test_mask.py \
  --data_path /path/to/test2k \
  --model_path Pixelwise_SR_experiments/fsrcnn_dyconv_v1_mask_new/models/3500000_G.pth \
  --model_name fsrcnn --dilate 5 --save_image \
  --result_path results/fsrcnn
```

Use the same script for CARN (`--model_name carn --dilate 5`) or SRResNet (`--model_name srresnet --dilate 7`), with the corresponding checkpoint.

**SwinIR**

```bash
python test_SwinIR_mask.py \
  --data_path /path/to/test2k \
  --model_path Pixelwise_SR_experiments/SwinIR_mask_new_progress_retrain/best_G.pth \
  --thres 0.5 --dilate 11 \
  --result_path results/swinir
```

### Train
### 1) Data preprocessing
We use the DIV2K training set as our base. Set the input and output paths in the preprocessing scripts before running them.
```
cd data/scripts
python data_augmentation.py
python extract_mask.py
python extract_subimages.py
python extract_submasks.py
python generate_mod_LR_bic.py
python generate_mod_mask_bic.py
cd ../..
```
### 2) Run training scripts (example usage)
Set the paths under `path` and `datasets`, and choose your `gpu_ids` in the selected configuration under `options/`.
```
python -m torch.distributed.launch  --nproc_per_node=1  --master_port=1432   train_mask_progress.py  --opt options/train_fsrcnn_dyconv_v1_mask_new.json  --dist True
python -m torch.distributed.launch  --nproc_per_node=1  --master_port=1432   train_mask_progress.py  --opt options/train_carn_dyconv_v1_mask_new_progress.json  --dist True
python -m torch.distributed.launch  --nproc_per_node=1  --master_port=1432   train_mask_progress.py  --opt options/train_srresnet_dyconv_v1_mask_new_da_progress.json  --dist True
python -m torch.distributed.launch  --nproc_per_node=1  --master_port=1432   train_mask_progress.py  --opt options/train_swinir_mask_new_progress.json  --dist True
```

## Cite
If you use any part of our code, or AMSR is useful for your research, please consider citing:
```
@article{shang2026frequency,
  title={Frequency-guided adaptive computation for efficient single-image super-resolution},
  author={Shang, Wei and Zhang, Wanying and Li, Wenwen and Wu, Xiaohe and Wu, Huaming and Ren, Dongwei},
  journal={Pattern Recognition},
  pages={},
  year={2026},
  publisher={Elsevier}
}
```

## Contact
If you have any questions, please contact csweishang@gmail.com.

