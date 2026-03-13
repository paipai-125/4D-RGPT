# 4D-RGPT Distillation Training

This repository contains the training code for distilling a 4D Perception Teacher (L4P) into a Student model (Qwen-VL).

## 环境配置 (Environment Setup)

### 1. 创建 Conda 虚拟环境

We recommend using Python 3.10 or higher.

```bash
conda create -n 4d-rgpt python=3.10 -y
conda activate 4d-rgpt
cd 4D-RGPT
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

### 下载 Qwen3-VL-8B-Instruct

The training script expects the model at `./src/models/Qwen3-VL-8B-Instruct`.

```bash
# Install huggingface-cli
pip install -U "huggingface_hub[cli]"

# Create target directory
mkdir -p src/models/Qwen3-VL-8B-Instruct

# Download model (You can change this to Qwen/Qwen2-VL-7B-Instruct if needed)
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir src/models/Qwen3-VL-8B-Instruct --local-dir-use-symlinks False
```

> **Note**: If you intend to use a different base model, please update `local_model_path` in `qwen_distill_all.py` or download that specific model to the path above.

### 下载 L4P 模型权重

download l4p_depth_flow_2d3dtrack_camray_dynseg_v1.ckpt

```bash
cd L4P-main/weights
bash download.sh
```

## 运行训练 (Training)

Usually, you can run the distillation training script directly:

```bash
python qwen_distill_all.py
```

### Notes on Configuration
- **Batch Size / CUDA Memory**: Adjust `parameters` in `qwen_distill_all.py` if running out of memory.
- **L4P Integration**: Ensure `L4P-main` contains the necessary weights in `L4P-main/weights/`. If L4P weights are missing, download them as per L4P instructions.

## Output

Training logs will be saved to `output_dis.txt`.
Checkpoints are saved in `checkpoints_distill/`.
