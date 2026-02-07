"""
3D AI 生成平台 - FastAPI 后端
基于 TRELLIS.2 模型，实现图片转 3D 模型功能
"""

import os
import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time

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

# ==================== 日志配置 ====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "app.log"),
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

class TaskStatus(BaseModel):
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
        self.model = None
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.tasks: dict[str, TaskStatus] = {}
        self.processing_lock = asyncio.Lock()
        self.is_processing = False

app_state = AppState()

# ==================== 模型加载 ====================

def load_trellis_model():
    """加载 TRELLIS.2 模型到 GPU"""
    logger.info("正在加载 TRELLIS.2 模型...")
    
    try:
        # TODO: 替换为实际的 TRELLIS.2 模型加载代码
        # from trellis import TrellisModel
        # model = TrellisModel.from_pretrained("trellis-2-4b")
        # model = model.to("cuda")
        # model.eval()
        
        # 模拟模型加载（实际使用时删除）
        logger.info("⚠️ 当前为模拟模式，请替换为实际 TRELLIS.2 模型加载代码")
        
        logger.info(f"模型加载完成，显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        return None  # 返回实际模型对象
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

def generate_3d_model(image_path: Path, output_path: Path) -> bool:
    """
    调用 TRELLIS.2 生成 3D 模型
    
    Args:
        image_path: 输入图片路径
        output_path: 输出 GLB 文件路径
    
    Returns:
        是否生成成功
    """
    try:
        logger.info(f"开始生成 3D 模型: {image_path}")
        start_time = time.time()
        
        # TODO: 替换为实际的 TRELLIS.2 推理代码
        # from PIL import Image
        # image = Image.open(image_path)
        # mesh = app_state.model.generate(image)
        # mesh.export(output_path)
        
        # 模拟生成过程（实际使用时删除）
        import time
        time.sleep(5)  # 模拟 5 秒生成时间
        
        # 创建一个空的 GLB 文件作为占位符（实际使用时删除）
        output_path.write_bytes(b"placeholder")
        
        elapsed = time.time() - start_time
        logger.info(f"3D 模型生成完成，耗时: {elapsed:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"3D 模型生成失败: {e}")
        return False

# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    logger.info("=== 服务启动 ===")
    app_state.model = load_trellis_model()
    
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
    description="基于 TRELLIS.2 的图片转 3D 模型服务",
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
                    logger.error(f"任务 {task_id} 处理失败: {e}")
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
            logger.error(f"队列处理器异常: {e}")
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
    return {
        "status": "running",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "queue_size": app_state.queue.qsize(),
        "is_processing": app_state.is_processing
    }

@app.post("/api/generate", response_model=GenerationResponse)
async def generate(image: UploadFile = File(...)):
    """
    上传图片并生成 3D 模型
    
    - 支持格式: JPG, PNG, WebP
    - 最大文件大小: 10MB
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
    app_state.tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="queued",
        queue_position=queue_position if not app_state.is_processing else queue_position
    )
    
    # 加入队列
    await app_state.queue.put((task_id, temp_image_path))
    
    # 如果没有正在处理的任务，等待当前任务完成
    if not app_state.is_processing or app_state.queue.qsize() == 1:
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
                    generation_time=generation_time
                )
            elif task.status == "failed":
                return GenerationResponse(
                    success=False,
                    error=task.error or "Generation failed"
                )
    
    # 如果队列中有其他任务，返回任务 ID 让客户端轮询
    return GenerationResponse(
        success=True,
        model_url=None,
        error=f"Task queued. Use /api/status/{task_id} to check progress."
    )

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """
    获取任务状态（支持 SSE）
    """
    task = app_state.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task

@app.get("/api/status/{task_id}/stream")
async def get_status_stream(task_id: str):
    """
    SSE 端点：实时推送任务状态
    """
    async def event_generator():
        while True:
            task = app_state.tasks.get(task_id)
            if not task:
                yield f"data: {{'error': 'Task not found'}}\n\n"
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
