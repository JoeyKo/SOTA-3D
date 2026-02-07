#!/bin/bash

# 3D AI 生成平台 - 启动脚本

set -e

echo "=== 3D AI 生成平台启动脚本 ==="

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 Python3，请先安装 Python 3.10+"
    exit 1
fi

# 检查 CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ 警告: 未找到 nvidia-smi，GPU 可能不可用"
else
    echo "✅ GPU 状态:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
fi

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "📦 安装依赖..."
pip install -r requirements.txt --quiet

# 创建必要目录
mkdir -p static logs

# 启动服务
echo "🚀 启动服务..."
echo "   访问地址: http://localhost:8000"
echo "   API 文档: http://localhost:8000/docs"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
