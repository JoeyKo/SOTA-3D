# TRELLIS.2 部署指南 - Docker 方式

本指南帮助你在 RTX 4090 GPU 服务器上通过 Docker 完成 TRELLIS.2 + FastAPI 的部署。

---

## 前置要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| OS | Ubuntu 22.04 LTS（或其他支持 Docker 的 Linux） |
| NVIDIA 驱动 | ≥ 550（支持 CUDA 12.4） |
| Docker | ≥ 24.0 |
| NVIDIA Container Toolkit | 已安装 |

---

## 1. 安装 Docker & NVIDIA Container Toolkit

如果服务器上还没有 Docker 和 NVIDIA Container Toolkit：

```bash
# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# 安装 NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证 GPU 在 Docker 中可见
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 2. 上传项目文件

将 `backend/` 目录上传到 GPU 服务器：

```bash
# 在本地 Mac
scp -r "/Users/kojoey/SOTA 3D/backend" user@gpu-server:~/trellis-api/

# 或者使用 rsync
rsync -avz "/Users/kojoey/SOTA 3D/backend/" user@gpu-server:~/trellis-api/
```

---

## 3. 构建 Docker 镜像

```bash
ssh user@gpu-server
cd ~/trellis-api

# 构建镜像（首次约 30-60 分钟，需要编译 CUDA 扩展）
docker compose build
```

> ⚠️ 构建过程会：
> - 克隆 TRELLIS.2 仓库
> - 安装 PyTorch 2.6.0 + CUDA 12.4
> - 编译 flash-attn、nvdiffrast、nvdiffrec、CuMesh、FlexGEMM、o-voxel
> - 安装 FastAPI 依赖
>
> 首次构建较慢，后续只要 Dockerfile 不变就会使用缓存。

---

## 4. 启动服务

```bash
# 前台启动（查看日志）
docker compose up

# 后台启动
docker compose up -d

# 查看日志
docker compose logs -f
```

首次启动时，模型权重会自动从 Hugging Face 下载到 Docker Volume `trellis-model-cache` 中（约 15-20GB）。后续重启不需要重新下载。

启动成功后会看到：
```
✅ 模型加载完成，显存占用: ~18.5 GB
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 5. 测试 API

```bash
# 健康检查
curl http://localhost:8000/

# 生成 3D 模型
curl -X POST http://localhost:8000/api/generate \
  -F "image=@test_image.png"

# 响应:
# {"success":true,"model_url":"/static/xxx.glb","generation_time":45.2}

# 下载 GLB 文件
curl -O http://localhost:8000/static/xxx.glb

# 或直接在 output/ 目录查看
ls ~/trellis-api/output/
```

API 文档：http://localhost:8000/docs

---

## 6. 常用运维命令

```bash
# 停止服务
docker compose down

# 重启服务
docker compose restart

# 查看资源使用（GPU + 内存）
docker stats trellis-3d-api
nvidia-smi

# 查看日志
docker compose logs -f --tail 100

# 进入容器调试
docker exec -it trellis-3d-api bash

# 清理旧的 GLB 文件（48 小时前）
find ~/trellis-api/output -name "*.glb" -mmin +2880 -delete

# 重新构建（代码更新后）
docker compose build --no-cache
docker compose up -d
```

---

## 7. (可选) 定时清理 Cron

```bash
crontab -e
# 每天凌晨 3 点清理 48 小时前的 GLB 文件
0 3 * * * find ~/trellis-api/output -name "*.glb" -mmin +2880 -delete
```

---

## 8. (可选) Cloudflare Tunnel 公网访问

```bash
# 在宿主机上安装 cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared

# 快速启动（临时域名）
cloudflared tunnel --url http://localhost:8000

# 使用自定义域名
cloudflared tunnel login
cloudflared tunnel create trellis-3d
cloudflared tunnel route dns trellis-3d api.yourdomain.com
cloudflared tunnel run --url http://localhost:8000 trellis-3d
```

---

## 目录结构

部署到服务器后的文件结构：

```
~/trellis-api/
├── Dockerfile           # Docker 构建文件
├── docker-compose.yml   # Docker Compose 编排
├── .dockerignore        # 构建忽略文件
├── main.py              # FastAPI 核心服务
├── requirements.txt     # FastAPI 额外依赖
├── start.sh             # 非 Docker 启动脚本（备用）
├── output/              # 生成的 GLB 文件（Docker Volume 映射）
└── logs/                # 日志文件（Docker Volume 映射）
```
