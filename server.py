import os
import json
import arxiv
import datetime
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv
from arxiv_subjects import search_subjects, get_all_subjects, get_subject_code
from call_jina import JinaReaderClient
from call_llm import LLMClient

# 加载环境变量
load_dotenv()

app = FastAPI()

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
WEB_DIR = os.path.join(BASE_DIR, "web")

# 确保目录存在
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)

# 服务器配置
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
# ===========================================


# 数据模型
class Paper(BaseModel):
    title: str
    authors: List[str]
    first_author: str
    abs_url: str
    pdf_url: str
    trans_abs: str
    bibtex: str
    published: Optional[str] = None
    summary: Optional[str] = None
    affiliation: Optional[str] = "Unknown"


# API 路由
@app.get("/api/dates")
async def get_dates():
    """获取所有可用的日报日期"""
    index_path = os.path.join(DATABASE_DIR, "index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.get("/api/report")
async def get_report(date: str):
    """获取指定日期的日报数据"""
    file_path = os.path.join(DATABASE_DIR, f"{date}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/search")
async def search_arxiv(query: str = Query(..., min_length=1)):
    """调用 ArXiv 库进行搜索"""
    logger.info(f"Searching ArXiv for: {query}")
    try:
        # 使用 arxiv 库搜索
        client = arxiv.Client()
        search = arxiv.Search(
            query=query, max_results=50, sort_by=arxiv.SortCriterion.SubmittedDate
        )

        results = []
        for r in client.results(search):
            authors = [a.name for a in r.authors]
            first_author = authors[0] if authors else "Unknown"

            results.append(
                {
                    "title": r.title,
                    "authors": authors,
                    "first_author": first_author,
                    "abs_url": r.entry_id,
                    "pdf_url": r.pdf_url,
                    "trans_abs": "（实时搜索结果，暂无翻译）",
                    "bibtex": f"@article{{{r.get_short_id()},\n  title={{{r.title}}},\n  author={{{' and '.join(authors)}}},\n  journal={{arXiv preprint arXiv:{r.get_short_id()}}},\n  year={{{r.published.year}}}\n}}",
                    "published": r.published.strftime("%Y-%m-%d"),
                    "summary": r.summary,
                    "affiliation": "ArXiv Metadata 不包含单位信息",
                }
            )
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/subjects/search")
async def api_search_subjects(q: str = Query(..., min_length=1)):
    """搜索 ArXiv 主题"""
    return search_subjects(q, limit=20)


@app.get("/api/subjects/all")
async def api_get_all_subjects():
    """获取所有 ArXiv 主题"""
    return get_all_subjects()


# 全局任务状态存储
task_status = {}


class FetchRequest(BaseModel):
    subject: str
    date: Optional[str] = None


def process_arxiv_fetch(task_id: str, subject_code: str, target_date: str):
    """后台任务：抽取和翻译 ArXiv 论文"""
    try:
        task_status[task_id] = {
            "status": "running",
            "progress": 0,
            "message": "正在初始化...",
        }

        # 1. 初始化客户端
        jina_client = JinaReaderClient()
        llm_client = LLMClient()

        task_status[task_id]["progress"] = 10
        task_status[task_id]["message"] = f"正在获取 {subject_code} 论文列表..."

        # 2. 获取 Jina 数据
        jina_data = jina_client.fetch_arxiv_list(subject=subject_code, skip=0, show=250)
        if not jina_data:
            task_status[task_id] = {"status": "error", "message": "Jina API 请求失败"}
            return

        task_status[task_id]["progress"] = 20
        task_status[task_id]["message"] = "正在解析论文 ID..."

        # 3. 解析 ID
        arxiv_ids = jina_client.parse_arxiv_ids(jina_data, target_date)
        if not arxiv_ids:
            task_status[task_id] = {
                "status": "error",
                "message": f"未找到 {target_date} 的论文",
            }
            return

        # 限制数量
        max_papers = min(len(arxiv_ids), 50)
        arxiv_ids = arxiv_ids[:max_papers]

        task_status[task_id]["progress"] = 30
        task_status[task_id]["message"] = (
            f"找到 {len(arxiv_ids)} 篇论文，正在获取元数据..."
        )

        # 4. 获取 ArXiv 元数据
        client = arxiv.Client()
        search = arxiv.Search(id_list=arxiv_ids)
        results = list(client.results(search))

        task_status[task_id]["progress"] = 40
        task_status[task_id]["total_papers"] = len(results)

        # 5. 处理每篇论文
        processed_papers = []
        for i, paper in enumerate(results):
            progress = 40 + int((i + 1) / len(results) * 50)
            task_status[task_id]["progress"] = progress
            task_status[task_id]["message"] = (
                f"正在处理 [{i + 1}/{len(results)}]: {paper.title[:50]}..."
            )

            # 翻译摘要
            trans_abs = llm_client.translate_abstract(paper.summary, domain="CS")

            # 生成 BibTeX
            authors = [a.name for a in paper.authors]
            bibtex = (
                f"@article{{{paper.get_short_id()},\n"
                f"  title={{{paper.title}}},\n"
                f"  author={{{' and '.join(authors)}}},\n"
                f"  journal={{arXiv preprint arXiv:{paper.get_short_id()}}},\n"
                f"  year={{{paper.published.year}}},\n"
                f"  url={{{paper.entry_id}}}\n"
                f"}}"
            )

            processed_papers.append(
                {
                    "title": paper.title,
                    "authors": authors,
                    "first_author": authors[0] if authors else "Unknown",
                    "abs_url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "trans_abs": trans_abs,
                    "bibtex": bibtex,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "summary": paper.summary,
                }
            )

        task_status[task_id]["progress"] = 90
        task_status[task_id]["message"] = "正在保存数据..."

        # 6. 保存数据
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = f"{date_str}_{subject_code.replace('.', '_')}.json"
        json_path = os.path.join(DATABASE_DIR, filename)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(processed_papers, f, ensure_ascii=False, indent=2)

        task_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "完成！",
            "result_file": filename,
            "paper_count": len(processed_papers),
        }

    except Exception as e:
        logger.exception(f"Task {task_id} failed: {e}")
        task_status[task_id] = {"status": "error", "message": str(e)}


@app.post("/api/fetch")
async def api_fetch_papers(request: FetchRequest, background_tasks: BackgroundTasks):
    """手动触发论文抽取"""
    # 获取主题代码
    subject_code = get_subject_code(request.subject)
    if not subject_code:
        subject_code = request.subject  # 如果直接传入了代码

    # 确定目标日期
    if request.date:
        target_date = request.date
    else:
        target_date = datetime.datetime.now().strftime("%a, %d %b %Y")

    # 生成任务 ID
    task_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{subject_code}"

    # 启动后台任务
    background_tasks.add_task(process_arxiv_fetch, task_id, subject_code, target_date)

    return {"task_id": task_id, "message": "任务已启动"}


@app.get("/api/task/{task_id}")
async def api_get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_status[task_id]


# 挂载静态文件
# /database 用于访问 JSON 数据（如果前端直接请求文件）
app.mount("/database", StaticFiles(directory=DATABASE_DIR), name="database")

# / 访问前端页面
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server at http://{SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
