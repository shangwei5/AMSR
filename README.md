# Frequency-guided adaptive computation for efficient single-image super-resolution （Pattern Recognition 2026）
---
[[arXiv](https://arxiv.org/abs/2505.06975)] [[PDF](https://arxiv.org/pdf/2505.06975)]

## Dependencies
- Python 3.8<br>
- Pytorch 1.13<br>
- NVIDIA GPU + CUDA<br>
- Python packages: `pip install numpy opencv-python pandas tqdm fast_pytorch_kmeans`

## Datasets
Please download the DIV2K dataset following [ClassSR](https://github.com/XPixelGroup/ClassSR?tab=readme-ov-file).

## Download Pre-trained Models
Please download the pre-trained models from [link](https://drive.google.com/file/d/1EJjQySoEWBkyBTt8Ejskv5LHcQyD8JuJ/view?usp=sharing). 

## Getting Started
### Test
```
bash test.sh
```
Please change `--data_path`, `--model_path` and `--result_path` in the file according to yours.

### Train
### 1) Data preprocessing
We use the DIV2K training set as our base.
```
cd data/scripts
python data_augmentation.py
python extract_mask.py
python extract_subimages.py
python extract_submasks.py
python generate_mod_LR_bic.py
python generate_mod_mask_bic.py
```
### 2) Run training scripts (example usage)
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

