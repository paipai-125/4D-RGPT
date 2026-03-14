# 4D-RGPT Distillation Training

This repository contains the training code for distilling a 4D Perception Teacher (L4P) into a Student model (Qwen-VL).

## 环境配置 (Environment Setup)

### 1. 创建 Conda 虚拟环境

We recommend using Python 3.10 or higher.

```bash
conda create -n 4d-rgpt python=3.10 -y
conda activate 4d-rgpt
```

### 2. 安装依赖 (Install Dependencies)

Install the required Python packages using pip:

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

apt install ffmpeg
```

## 数据集准备 (Dataset Preparation)

### 下载 RoboFAC 数据集 (HuggingFace)

```bash
mkdir -p /4D-Data
cd 4D-Data
```

Download the dataset from HuggingFace and place it in the `RoboFAC/` directory.

```bash
# Install git-lfs if not already installed
git lfs install

# Clone the dataset repository
git clone https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset RoboFAC
```

Verify the structure:
```
RoboFAC/
├── simulation_data/
│   └── ... (video files)
└── training_qa.json
```

## 模型准备 (Model Preparation)

### 下载 Qwen3-VL-2B-Instruct or Qwen3-VL-8B-Instruct

The training script expects the model at `4D-Data/models/Qwen3-VL-2B-Instruct`.

```bash
# Install huggingface-cli
pip install -U "huggingface_hub[cli]"

# Download model (You can change this to Qwen/Qwen2-VL-7B-Instruct if needed)
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir models/Qwen3-VL-2B-Instruct --local-dir-use-symlinks False
```

> **Note**: If you intend to use a different base model, please update `local_model_path` in `qwen_distill_all_ji.py` or download that specific model to the path above.

### 下载 L4P 模型权重

Download l4p_depth_flow_2d3dtrack_camray_dynseg_v1.ckpt

```bash
cd 4D-RGPT/L4P-main/weights
bash download.sh
```

Put `l4p_depth_flow_2d3dtrack_camray_dynseg_v1.ckpt` at `4D-Data/l4p-weights`.


Verify the structure:
```
4D-RGPT/
├── L4P-main/
└── qwen_distill_all_ji.py
└── ... (other code files)
4D-Data/
├── l4p-weights/
└── models/
└── RoboFAC/
```


## 运行训练 (Training)

Usually, you can run the distillation training script directly:

```bash
cd 4D-RGPT
torchrun --nproc_per_node=8 qwen_distill_all_ji.py
```

### Notes on Configuration
- **Batch Size / CUDA Memory**: Adjust `parameters` in `qwen_distill_all_ji.py` if running out of memory.
- **L4P Integration**: Ensure that the necessary weights of L4P is saved in `4D-Data/l4p-weights/`.

## Output

Training logs will be saved to `output_dis.txt`.
Checkpoints are saved in `4D-Data/checkpoints_distill/`.
