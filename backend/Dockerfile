# ============================================================
# TRELLIS.2 + FastAPI - Docker 构建文件
# 基于 NVIDIA CUDA 12.4 + Python 3.10
# ============================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN python -m pip install --upgrade pip setuptools wheel

# 安装 PyTorch (CUDA 12.4)
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# ============================================================
# 克隆 TRELLIS.2 仓库并安装依赖
# ============================================================
WORKDIR /app

RUN git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive /app/TRELLIS.2

WORKDIR /app/TRELLIS.2

# 安装基础 Python 依赖
RUN pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
    trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard \
    && pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 \
    && pip install pillow-simd kornia timm

# 安装 flash-attention
RUN pip install flash-attn==2.7.3

# 安装 nvdiffrast
RUN mkdir -p /tmp/extensions \
    && git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast \
    && pip install /tmp/extensions/nvdiffrast --no-build-isolation

# 安装 nvdiffrec
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec \
    && pip install /tmp/extensions/nvdiffrec --no-build-isolation

# 安装 CuMesh
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive \
    && pip install /tmp/extensions/CuMesh --no-build-isolation

# 安装 FlexGEMM
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive \
    && pip install /tmp/extensions/FlexGEMM --no-build-isolation

# 安装 o-voxel (从仓库子目录)
RUN cp -r o-voxel /tmp/extensions/o-voxel \
    && pip install /tmp/extensions/o-voxel --no-build-isolation

# 清理编译缓存
RUN rm -rf /tmp/extensions

# ============================================================
# 运行时配置
# ============================================================

# 安装 FastAPI 依赖
COPY requirements.txt /app/TRELLIS.2/requirements.txt
RUN pip install -r requirements.txt

# 复制应用代码
COPY main.py /app/TRELLIS.2/main.py

# 创建必要目录
RUN mkdir -p /app/TRELLIS.2/static /app/TRELLIS.2/logs

# 环境变量
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# HuggingFace 缓存目录（用于模型权重挂载）
ENV HF_HOME=/app/models

# 启动命令
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
