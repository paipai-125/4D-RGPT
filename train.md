## 环境配置 (Environment Setup)

### 1. 安装Grounding DINO和sam2依赖

```bash
echo $CUDA_HOME

# 如果没有输出
which nvcc  # 如果输出是/usr/local/cuda/bin/nvcc
export CUDA_HOME=/usr/local/cuda
source ~/.bashrc
echo $CUDA_HOME # 确保有输出
```

```bash
cd GroundingDINO/
pip install -e . --no-build-isolation

mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..
```

```bash
cd sam2/
pip install -e . --no-build-isolation

cd checkpoints/
bash download_ckpts.sh
mv sam2.1_hiera_base_plus.pt ../../../4D-Data/sam2_weights
cd ../..
```


## 数据集准备 (Dataset Preparation)

### 下载 SIMS-VSI 数据集 (HuggingFace)

```bash
cd 4D-Data
huggingface-cli download ellisbrown/SIMS-VSI \
  --repo-type dataset \
  --local-dir ./SIMS-VSI
tar --zstd -xvf video_shard_000.tar.zst
tar --zstd -xvf video_shard_001.tar.zst
tar --zstd -xvf video_shard_002.tar.zst
```

## 运行训练 (Training)

```bash
cd 4D-RGPT
torchrun --nproc_per_node=8 all_pipeline.py

# nproc_per_node 运行时使用卡的数量
```

训练权重保存在'4D-Data/checkpoints_all'目录下