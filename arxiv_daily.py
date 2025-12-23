import os
import json
import arxiv
import datetime
from typing import Optional
from dotenv import load_dotenv
from loguru import logger

# 导入自定义工具模块
from dataclasses import dataclass, field

from tools.call_llm import LLMClient
from tools.call_jina import JinaReaderClient
from tools.report2md import papers_to_markdown
from tools.upload_md_feishu import FeishuDriveUploadError, upload_file

# 加载环境变量
load_dotenv()

# ================= 配置区域 =================
ARXIV_SUBJECT = os.getenv("ARXIV_SUBJECT", "cs.AI")
DATE_OFFSET_DAYS = int(os.getenv("ARXIV_DATE_OFFSET_DAYS", "1"))
CURRENT_DATE = datetime.datetime.now() - datetime.timedelta(days=DATE_OFFSET_DAYS)
TARGET_DATE_STR = CURRENT_DATE.strftime("%a, %d %b %Y")
MAX_PAPERS = int(os.getenv("MAX_PAPERS", "200"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

FEISHU_DOCX_FOLDER_TOKEN = os.getenv("FEISHU_DOCX_FOLDER_TOKEN")
FEISHU_DOCX_BASE_URL = os.getenv("FEISHU_DOCX_BASE_URL")
# ===========================================


def _build_docx_url(*, document_id: str) -> Optional[str]:
    # 文档访问 URL 可能因飞书部署/租户域名而不同，因此允许通过环境变量覆盖
    base = (FEISHU_DOCX_BASE_URL or "https://ai.feishu.cn/docx").strip().rstrip("/")
    return f"{base}/{document_id}"


def upload_markdown_via_drive(*, md_path: str, file_name: str) -> Optional[str]:
    """兼容旧函数名：实际写入飞书 Docx 文档并返回文档链接。"""

    if upload_file is None:
        logger.error("缺少模块 feishu_drive_upload（或依赖 lark-oapi），无法上传")
        return None

    folder_token = (FEISHU_DOCX_FOLDER_TOKEN or "").strip()
    if not folder_token and not (os.getenv("FEISHU_DOCX_DOCUMENT_ID") or "").strip():
        logger.error(
            "缺少环境变量 FEISHU_DOCX_FOLDER_TOKEN（或 FEISHU_DOCX_DOCUMENT_ID）"
        )
        return None

    if not os.path.exists(md_path):
        logger.error(f"待上传文件不存在: {md_path}")
        return None

    try:
        document_id = upload_file(
            file_path=md_path,
            file_name=file_name,
            parent_node=folder_token,
        )
        return _build_docx_url(document_id=document_id)
    except FeishuDriveUploadError as e:
        logger.exception(f"写入 md 到飞书 Docx 失败: {e}")
        return None


@dataclass
class ArxivPaperProcessor:
    """ArXiv 论文处理器（dataclass 版本）"""

    llm_client: LLMClient = field(default_factory=LLMClient)
    jina_client: JinaReaderClient = field(default_factory=JinaReaderClient)

    def fetch_jina_data(self, subject: str = "cs.AI"):
        """调用 Jina Reader API 获取 ArXiv 列表数据"""
        return self.jina_client.fetch_arxiv_list(subject=subject, skip=0, show=250)

    def parse_jina_response(self, json_data, target_date):
        """从 Jina Reader 的 JSON 响应中提取指定日期的 ArXiv ID"""
        return self.jina_client.parse_arxiv_ids(json_data, target_date)

    def fetch_arxiv_metadata(self, arxiv_ids):
        """使用 arxiv 库批量获取元数据"""
        if not arxiv_ids:
            return []
        logger.info("正在从 ArXiv 获取元数据...")
        arxiv_ids = arxiv_ids[:MAX_PAPERS]  # 限制最大数量

        client = arxiv.Client()
        search = arxiv.Search(id_list=arxiv_ids)
        results = list(client.results(search))
        return results

    def generate_bibtex(self, paper):
        """根据 arxiv 元数据生成 BibTeX"""
        year = paper.published.year
        title = paper.title
        author_text = " and ".join([a.name for a in paper.authors])
        url = paper.entry_id
        doi = paper.doi
        eprint = paper.get_short_id()

        doi_line = f"  doi={{{doi}}},\n" if doi else ""
        bibtex = (
            f"@article{{{eprint},\n"
            f"  title={{{title}}},\n"
            f"  author={{{author_text}}},\n"
            f"  journal={{arXiv preprint arXiv:{eprint}}},\n"
            f"  year={{{year}}},\n"
            f"  url={{{url}}},\n"
            f"{doi_line}"
            f"}}"
        )
        return bibtex

    def translate_abstract(self, title, abstract):
        """调用 LLM 翻译摘要并分析"""
        combined_text = f"Title: {title}\nAbstract: {abstract}"
        return self.llm_client.translate_abstract(combined_text, domain="AI")


def save_markdown(markdown_text: str, *, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    return os.path.abspath(path)


def save_text(text: str, *, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return os.path.abspath(path)


def main():
    # 1. 初始化
    processor = ArxivPaperProcessor()

    # 2. 获取 Jina 数据
    logger.info(f"正在获取 ArXiv 主题: {ARXIV_SUBJECT}")
    jina_data = processor.fetch_jina_data(subject=ARXIV_SUBJECT)
    if not jina_data:
        return

    # 3. 解析 ID
    ids = processor.parse_jina_response(jina_data, TARGET_DATE_STR)
    if not ids:
        logger.warning("未提取到论文 ID")
        return

    # 4. 获取元数据并处理
    arxiv_results = processor.fetch_arxiv_metadata(ids)
    processed_papers = []

    for i, paper in enumerate(arxiv_results):
        logger.info(f"[{i + 1}/{len(arxiv_results)}] 处理: {paper.title}")
        analysis_json = processor.translate_abstract(paper.title, paper.summary)
        try:
            analysis = json.loads(analysis_json)
        except Exception as e:
            logger.error(f"解析 JSON 失败: {e}, 内容: {analysis_json}")
            analysis = {
                "trans_abs": analysis_json,
                "keywords": ["未知", "未知", "未知"],
                "sub_topic": "未知",
                "recommendation": "一般推荐",
            }

        bib = processor.generate_bibtex(paper)

        # 获取作者信息 (保留在 JSON 中供数据库使用)
        authors_list = [author.name for author in paper.authors]

        processed_papers.append(
            {
                "title": paper.title,
                "authors": authors_list,
                "abs_url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "trans_abs": analysis.get("trans_abs", "翻译失败"),
                "keywords": analysis.get("keywords", ["未知", "未知", "未知"]),
                "sub_topic": analysis.get("sub_topic", "未知"),
                "recommendation": analysis.get("recommendation", "一般推荐"),
                "bibtex": bib,
                "doi": paper.doi,
                "doi_url": f"https://doi.org/{paper.doi}" if paper.doi else None,
                "published": paper.published.strftime("%Y-%m-%d"),
                "summary": paper.summary,
            }
        )

    # 5. 排序：按推荐程度从高到低排序
    recommendation_order = {
        "极度推荐": 5,
        "很推荐": 4,
        "推荐": 3,
        "一般推荐": 2,
        "不推荐": 1,
    }
    processed_papers.sort(
        key=lambda x: recommendation_order.get(x["recommendation"], 0), reverse=True
    )

    # 6. 保存数据
    # 确保 database 目录存在
    output_dir = os.path.join(os.path.dirname(__file__), "database")
    os.makedirs(output_dir, exist_ok=True)

    # 保存 JSON 数据
    json_filename = f"{CURRENT_DATE.strftime('%Y-%m-%d')}.json"
    json_path = os.path.join(output_dir, json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(processed_papers, f, ensure_ascii=False, indent=2)
    logger.success(f"JSON 数据已保存: {json_path}")

    # 更新 index.json
    index_path = os.path.join(output_dir, "index.json")
    index_data = []
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        except:  # noqa: E722
            pass

    # 添加当前日期（如果不存在）
    current_date_str = CURRENT_DATE.strftime("%Y-%m-%d")
    if current_date_str not in index_data:
        index_data.insert(0, current_date_str)  # 最新日期排前面
        # 限制保留最近 365 天
        index_data = index_data[:365]
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    # 生成 Markdown（本地留存）
    markdown_text = papers_to_markdown(current_date_str, processed_papers)
    md_output_dir = OUTPUT_DIR or output_dir
    md_filename = f"{current_date_str}.md"
    md_path = save_markdown(
        markdown_text, output_dir=md_output_dir, filename=md_filename
    )
    logger.success(f"Markdown 已保存: {md_path}")

    # 强制 docx convert 走 Markdown（以 md 作为文档块转换源）
    os.environ["FEISHU_DOCX_CONTENT_TYPE"] = "markdown"
    file_url = upload_markdown_via_drive(md_path=md_path, file_name=md_filename)
    if not file_url:
        logger.error("写入飞书 Docx 失败，跳过飞书推送")
        return

    logger.success(f"飞书 Docx 写入成功: {file_url}")


if __name__ == "__main__":
    main()
