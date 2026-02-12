# 3D AI 生成平台 - 后端服务

基于 TRELLIS.2-4B 模型的图片转 3D 模型 API 服务。

## 快速启动 (Docker)

> **前提**: GPU 服务器上已安装 Docker 和 NVIDIA Container Toolkit（详见 [DEPLOY.md](./DEPLOY.md)）

```bash
# 上传到 GPU 服务器后
cd ~/trellis-api
docker compose build    # 首次约 30-60 分钟
docker compose up -d    # 后台启动
```

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| OS | Ubuntu 22.04 LTS |
| CUDA | 12.4 |
| Python | 3.10+ (conda trellis2 环境) |

## API 接口

### 健康检查
```
GET /
```
返回 GPU 状态、显存使用、队列信息。

### 生成 3D 模型
```
POST /api/generate
Content-Type: multipart/form-data
Body: { image: File }
```

**支持格式**: JPG, PNG, WebP  
**最大文件**: 10MB  
**推理耗时**: 30-90 秒

**响应示例**:
```json
{
  "success": true,
  "model_url": "/static/550e8400-e29b-41d4-a716-446655440000.glb",
  "generation_time": 45.2
}
```

### 查询任务状态
```
GET /api/status/{task_id}
```

### SSE 实时状态流
```
GET /api/status/{task_id}/stream
```

### 取消任务
```
DELETE /api/task/{task_id}
```

## 访问地址

- 服务: http://localhost:8000
- API 文档: http://localhost:8000/docs
- 静态文件: http://localhost:8000/static/

## 目录结构

```
backend/
├── Dockerfile           # Docker 构建文件
├── docker-compose.yml   # Docker Compose 编排
├── .dockerignore        # 构建忽略
├── main.py              # FastAPI 核心服务 + TRELLIS.2 推理
├── requirements.txt     # FastAPI 额外依赖（核心依赖由 Docker 构建）
├── start.sh             # 非 Docker 启动脚本（备用）
├── DEPLOY.md            # 完整 Docker 部署指南
├── output/              # 生成的 GLB 模型存放（Docker Volume 映射）
└── logs/                # 日志目录
```

## 注意事项

1. **必须在 TRELLIS.2 仓库目录下运行**：推理依赖 `trellis2` 和 `o_voxel` 包
2. **并发限制**: 同时只处理 1 个请求，队列最多 5 个
3. **显存**: 模型常驻约 18.5GB，推理时峰值更高
4. **文件清理**: 建议配置 cron 定时清理 `static/` 下的旧文件