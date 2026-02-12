"""
3D AI 生成平台 - FastAPI 后端
基于 TRELLIS.2 模型，实现图片转 3D 模型功能
"""

import os
# TRELLIS.2 所需环境变量（必须在其他导入之前设置）
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from logging.handlers import TimedRotatingFileHandler

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time

# TRELLIS.2 核心依赖
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# ==================== 配置 ====================

# 目录配置
STATIC_DIR = Path("./static")
LOGS_DIR = Path("./logs")
STATIC_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# 队列配置
MAX_QUEUE_SIZE = 5
REQUEST_TIMEOUT = 180  # 秒

# 支持的图片格式
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# GLB 导出配置
GLB_DECIMATION_TARGET = 1000000
GLB_TEXTURE_SIZE = 4096

# ==================== 日志配置 ====================

log_handler = TimedRotatingFileHandler(
    LOGS_DIR / "app.log",
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        log_handler,
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 数据模型 ====================

class GenerationResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    success: bool
    model_url: Optional[str] = None
    generation_time: Optional[float] = None
    error: Optional[str] = None

class TaskStatusModel(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    task_id: str
    status: str  # queued, processing, completed, failed
    queue_position: Optional[int] = None
    progress: Optional[int] = None
    model_url: Optional[str] = None
    error: Optional[str] = None

# ==================== 全局状态 ====================

class AppState:
    def __init__(self):
        self.pipeline = None  # TRELLIS.2 推理管线
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.tasks: dict[str, TaskStatusModel] = {}
        self.processing_lock = asyncio.Lock()
        self.is_processing = False

app_state = AppState()

# ==================== 模型加载 ====================

def load_trellis_model() -> Trellis2ImageTo3DPipeline:
    """加载 TRELLIS.2-4B 模型到 GPU（Warm-up）"""
    logger.info("正在加载 TRELLIS.2-4B 模型...")
    
    try:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        pipeline.cuda()
        
        vram_used = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"✅ 模型加载完成，显存占用: {vram_used:.2f} GB")
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
        raise

def generate_3d_model(image_path: Path, output_path: Path) -> bool:
    """
    调用 TRELLIS.2 生成 3D 模型并导出 GLB
    
    Args:
        image_path: 输入图片路径
        output_path: 输出 GLB 文件路径
    
    Returns:
        是否生成成功
    """
    try:
        logger.info(f"开始生成 3D 模型: {image_path}")
        start_time = time.time()
        
        # 1. 加载图片
        image = Image.open(image_path)
        logger.info(f"图片尺寸: {image.size}, 模式: {image.mode}")
        
        # 2. 运行推理管线
        logger.info("正在运行 TRELLIS.2 推理...")
        mesh = app_state.pipeline.run(image)[0]
        
        # 3. 简化网格（nvdiffrast 限制）
        mesh.simplify(16777216)
        logger.info("网格简化完成")
        
        # 4. 导出 GLB
        logger.info("正在导出 GLB 文件...")
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=GLB_DECIMATION_TARGET,
            texture_size=GLB_TEXTURE_SIZE,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True
        )
        glb.export(str(output_path), extension_webp=True)
        
        elapsed = time.time() - start_time
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ 3D 模型生成完成，耗时: {elapsed:.2f}s，文件大小: {file_size_mb:.2f} MB")
        return True
        
    except torch.cuda.OutOfMemoryError:
        logger.warning("⚠️ 显存不足，正在清理并重试...")
        torch.cuda.empty_cache()
        
        try:
            # 重试一次
            image = Image.open(image_path)
            mesh = app_state.pipeline.run(image)[0]
            mesh.simplify(16777216)
            
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=GLB_DECIMATION_TARGET,
                texture_size=GLB_TEXTURE_SIZE,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=True
            )
            glb.export(str(output_path), extension_webp=True)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ 重试成功，3D 模型生成完成，耗时: {elapsed:.2f}s")
            return True
            
        except Exception as retry_err:
            logger.error(f"❌ 重试仍然失败: {retry_err}", exc_info=True)
            torch.cuda.empty_cache()
            return False
        
    except Exception as e:
        logger.error(f"❌ 3D 模型生成失败: {e}", exc_info=True)
        torch.cuda.empty_cache()
        return False

# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    logger.info("=== 服务启动 ===")
    app_state.pipeline = load_trellis_model()
    
    # 启动后台任务处理器
    asyncio.create_task(process_queue())
    
    yield
    
    # 关闭时清理
    logger.info("=== 服务关闭 ===")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="3D AI 生成平台",
    description="基于 TRELLIS.2-4B 的图片转 3D 模型服务",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ==================== 队列处理 ====================

async def process_queue():
    """后台任务：处理生成队列"""
    logger.info("队列处理器已启动")
    
    while True:
        try:
            # 等待队列中的任务
            task_id, image_path = await app_state.queue.get()
            
            async with app_state.processing_lock:
                app_state.is_processing = True
                
                # 更新任务状态
                if task_id in app_state.tasks:
                    app_state.tasks[task_id].status = "processing"
                    app_state.tasks[task_id].queue_position = None
                    app_state.tasks[task_id].progress = 0
                
                # 更新队列中其他任务的位置
                update_queue_positions()
                
                try:
                    # 生成模型
                    output_filename = f"{task_id}.glb"
                    output_path = STATIC_DIR / output_filename
                    
                    # 在线程池中运行同步的模型生成
                    loop = asyncio.get_event_loop()
                    success = await loop.run_in_executor(
                        None, generate_3d_model, image_path, output_path
                    )
                    
                    if success and task_id in app_state.tasks:
                        app_state.tasks[task_id].status = "completed"
                        app_state.tasks[task_id].progress = 100
                        app_state.tasks[task_id].model_url = f"/static/{output_filename}"
                    elif task_id in app_state.tasks:
                        app_state.tasks[task_id].status = "failed"
                        app_state.tasks[task_id].error = "Generation failed"
                        
                except Exception as e:
                    logger.error(f"任务 {task_id} 处理失败: {e}", exc_info=True)
                    if task_id in app_state.tasks:
                        app_state.tasks[task_id].status = "failed"
                        app_state.tasks[task_id].error = str(e)
                
                finally:
                    # 清理临时图片
                    if image_path.exists():
                        image_path.unlink()
                    
                    app_state.is_processing = False
                    app_state.queue.task_done()
                    
        except Exception as e:
            logger.error(f"队列处理器异常: {e}", exc_info=True)
            await asyncio.sleep(1)

def update_queue_positions():
    """更新队列中任务的位置"""
    position = 1
    for task_id, task in app_state.tasks.items():
        if task.status == "queued":
            task.queue_position = position
            position += 1

# ==================== API 端点 ====================

@app.get("/")
async def root():
    """健康检查"""
    gpu_info = "N/A"
    vram_info = "N/A"
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_mem / 1024**3
        vram_info = f"{vram_used:.1f}GB / {vram_total:.1f}GB"
    
    return {
        "status": "running",
        "model": "TRELLIS.2-4B",
        "gpu": gpu_info,
        "vram": vram_info,
        "queue_size": app_state.queue.qsize(),
        "is_processing": app_state.is_processing
    }

@app.post("/api/generate", response_model=GenerationResponse)
async def generate(image: UploadFile = File(...)):
    """
    上传图片并生成 3D 模型
    
    - 支持格式: JPG, PNG, WebP
    - 最大文件大小: 10MB
    - 返回生成的 GLB 文件下载链接
    """
    start_time = time.time()
    
    # 验证文件扩展名
    ext = Path(image.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # 读取文件内容
    content = await image.read()
    
    # 验证文件大小
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large, max {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # 检查队列是否已满
    if app_state.queue.full():
        raise HTTPException(
            status_code=503,
            detail="Server busy, retry later"
        )
    
    # 生成任务 ID
    task_id = str(uuid.uuid4())
    
    # 保存临时图片
    temp_image_path = STATIC_DIR / f"temp_{task_id}{ext}"
    temp_image_path.write_bytes(content)
    
    logger.info(f"收到生成请求: task_id={task_id}, size={len(content)/1024:.1f}KB, format={ext}")
    
    # 创建任务状态
    queue_position = app_state.queue.qsize() + 1
    app_state.tasks[task_id] = TaskStatusModel(
        task_id=task_id,
        status="queued",
        queue_position=queue_position
    )
    
    # 加入队列
    await app_state.queue.put((task_id, temp_image_path))
    
    # 等待任务完成（最多等待 REQUEST_TIMEOUT 秒）
    for _ in range(REQUEST_TIMEOUT * 2):
        await asyncio.sleep(0.5)
        task = app_state.tasks.get(task_id)
        if task and task.status in ("completed", "failed"):
            break
    
    task = app_state.tasks.get(task_id)
    if task:
        if task.status == "completed":
            generation_time = time.time() - start_time
            return GenerationResponse(
                success=True,
                model_url=task.model_url,
                generation_time=round(generation_time, 2)
            )
        elif task.status == "failed":
            return GenerationResponse(
                success=False,
                error=task.error or "Generation failed"
            )
    
    # 超时 - 返回任务 ID 让客户端轮询
    return GenerationResponse(
        success=True,
        model_url=None,
        error=f"Generation timeout. Use /api/status/{task_id} to check progress."
    )

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """获取任务状态"""
    task = app_state.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task

@app.get("/api/status/{task_id}/stream")
async def get_status_stream(task_id: str):
    """SSE 端点：实时推送任务状态"""
    async def event_generator():
        while True:
            task = app_state.tasks.get(task_id)
            if not task:
                yield f"data: {{\"error\": \"Task not found\"}}\n\n"
                break
            
            yield f"data: {task.model_dump_json()}\n\n"
            
            if task.status in ("completed", "failed"):
                break
            
            await asyncio.sleep(1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.delete("/api/task/{task_id}")
async def cancel_task(task_id: str):
    """取消任务（仅对排队中的任务有效）"""
    task = app_state.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != "queued":
        raise HTTPException(status_code=400, detail="Can only cancel queued tasks")
    
    task.status = "failed"
    task.error = "Cancelled by user"
    return {"success": True, "message": "Task cancelled"}

# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
