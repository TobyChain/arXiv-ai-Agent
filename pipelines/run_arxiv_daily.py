import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import arxiv
from dotenv import load_dotenv
from loguru import logger

from clients.call_llm import LLMClient
from clients.fetch_crossref import fetch_bibtex_by_doi
from clients.fetch_jina import JinaReaderClient
from integrations.feishu.write_docx import FeishuDriveUploadError, upload_file
from renderers.render_markdown import papers_to_markdown

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


def _project_root() -> str:
    # pipelines/ 的上一级目录就是项目根目录 arxiv-ai-agent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _build_docx_url(*, document_id: str) -> Optional[str]:
    base = (FEISHU_DOCX_BASE_URL or "https://ai.feishu.cn/docx").strip().rstrip("/")
    return f"{base}/{document_id}"


def upload_markdown_via_drive(*, md_path: str, file_name: str) -> Optional[str]:
    """把 Markdown 写入飞书 Docx 并返回文档链接。"""
    if upload_file is None:
        logger.error("缺少模块 integrations.feishu.write_docx（或依赖 lark-oapi），无法上传")
        return None

    folder_token = (FEISHU_DOCX_FOLDER_TOKEN or "").strip()
    if not folder_token and not (os.getenv("FEISHU_DOCX_DOCUMENT_ID") or "").strip():
        logger.error("缺少环境变量 FEISHU_DOCX_FOLDER_TOKEN（或 FEISHU_DOCX_DOCUMENT_ID）")
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
        return self.jina_client.fetch_arxiv_list(subject=subject, skip=0, show=250)

    def parse_jina_response(self, json_data, target_date):
        return self.jina_client.parse_arxiv_ids(json_data, target_date)

    def fetch_arxiv_metadata(self, arxiv_ids):
        if not arxiv_ids:
            return []
        logger.info("正在从 ArXiv 获取元数据...")
        arxiv_ids = arxiv_ids[:MAX_PAPERS]

        client = arxiv.Client()
        search = arxiv.Search(id_list=arxiv_ids)
        results = list(client.results(search))
        return results

    def generate_bibtex(self, paper):
        """优先使用 DOI 从 Crossref 获取规范 BibTeX，失败再回退到 arXiv 模板。"""
        year = paper.published.year
        title = paper.title
        author_text = " and ".join([a.name for a in paper.authors])
        url = paper.entry_id
        doi = paper.doi
        eprint = paper.get_short_id()

        if doi:
            bib = fetch_bibtex_by_doi(doi)
            if bib:
                return bib

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
        combined_text = f"Title: {title}\nAbstract: {abstract}"
        return self.llm_client.translate_abstract(combined_text, domain="AI")


def save_markdown(markdown_text: str, *, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    return os.path.abspath(path)


def main() -> None:
    processor = ArxivPaperProcessor()

    logger.info(f"正在获取 ArXiv 主题: {ARXIV_SUBJECT}")
    jina_data = processor.fetch_jina_data(subject=ARXIV_SUBJECT)
    if not jina_data:
        return

    ids = processor.parse_jina_response(jina_data, TARGET_DATE_STR)
    if not ids:
        logger.warning("未提取到论文 ID")
        return

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

    output_dir = os.path.join(_project_root(), "database")
    os.makedirs(output_dir, exist_ok=True)

    json_filename = f"{CURRENT_DATE.strftime('%Y-%m-%d')}.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(processed_papers, f, ensure_ascii=False, indent=2)
    logger.success(f"JSON 数据已保存: {json_path}")

    index_path = os.path.join(output_dir, "index.json")
    index_data = []
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        except Exception:
            pass

    current_date_str = CURRENT_DATE.strftime("%Y-%m-%d")
    if current_date_str not in index_data:
        index_data.insert(0, current_date_str)
        index_data = index_data[:365]
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    markdown_text = papers_to_markdown(current_date_str, processed_papers)
    md_output_dir = OUTPUT_DIR or output_dir
    md_filename = f"{current_date_str}.md"
    md_path = save_markdown(markdown_text, output_dir=md_output_dir, filename=md_filename)
    logger.success(f"Markdown 已保存: {md_path}")

    os.environ["FEISHU_DOCX_CONTENT_TYPE"] = "markdown"
    file_url = upload_markdown_via_drive(md_path=md_path, file_name=md_filename)
    if not file_url:
        logger.error("写入飞书 Docx 失败，跳过飞书推送")
        return

    logger.success(f"飞书 Docx 写入成功: {file_url}")


