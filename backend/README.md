# 3D AI 生成平台 - 后端服务

基于 TRELLIS.2 模型的图片转 3D 模型 API 服务。

## 快速启动

```bash
cd /Users/kojoey/SOTA\ 3D/backend
chmod +x start.sh
./start.sh
```

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| OS | Ubuntu 22.04 LTS |
| CUDA | 12.4 |
| Python | 3.10+ |

## API 接口

### 健康检查
```
GET /
```

### 生成 3D 模型
```
POST /api/generate
Content-Type: multipart/form-data
Body: { image: File }
```

**支持格式**: JPG, PNG, WebP  
**最大文件**: 10MB

**响应示例**:
```json
{
  "success": true,
  "model_url": "/static/xxx.glb",
  "generation_time": 45.2
}
```

### 查询任务状态
```
GET /api/status/{task_id}
```

### SSE 实时状态
```
GET /api/status/{task_id}/stream
```

## 访问地址

- 服务: http://localhost:8000
- API 文档: http://localhost:8000/docs
- 静态文件: http://localhost:8000/static/

## 目录结构

```
backend/
├── main.py          # 核心代码
├── requirements.txt # 依赖
├── start.sh         # 启动脚本
├── static/          # 生成的模型存放
└── logs/            # 日志目录
```

## ⚠️ 注意事项

1. **模型集成**: 在 `main.py` 中搜索 `TODO`，替换为实际的 TRELLIS.2 模型代码
2. **当前为模拟模式**: 生成会延迟 5 秒，输出占位符文件
3. **并发限制**: 同时只处理 1 个请求，队列最多 5 个