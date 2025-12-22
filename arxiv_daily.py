import os
import json
import arxiv
import datetime
from typing import Optional
from dotenv import load_dotenv
from loguru import logger

# 导入自定义工具模块
from call_llm import LLMClient
from call_feishu_card import FeishuCardNotifier
from call_jina import JinaReaderClient
from md_report import papers_to_markdown

try:
    from feishu_drive_upload import FeishuDriveUploadError, upload_file
except ModuleNotFoundError:  # pragma: no cover
    FeishuDriveUploadError = Exception  # type: ignore[assignment]
    upload_file = None  # type: ignore[assignment]

# 加载环境变量
load_dotenv()

# ================= 配置区域 =================
ARXIV_SUBJECT = os.getenv("ARXIV_SUBJECT", "cs.AI")
DATE_OFFSET_DAYS = int(os.getenv("ARXIV_DATE_OFFSET_DAYS", "1"))
CURRENT_DATE = datetime.datetime.now() - datetime.timedelta(days=DATE_OFFSET_DAYS)
TARGET_DATE_STR = CURRENT_DATE.strftime("%a, %d %b %Y")
MAX_PAPERS = int(os.getenv("MAX_PAPERS", "200"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

FEISHU_DRIVE_PARENT_NODE = os.getenv("FEISHU_DRIVE_PARENT_NODE")
FEISHU_DRIVE_BASE_URL = os.getenv("FEISHU_DRIVE_BASE_URL")
# ===========================================


def _build_drive_file_url(*, file_token: str) -> Optional[str]:
    # 优先使用环境变量中的基础 URL，若无则使用飞书云空间默认文件路径格式
    base = (FEISHU_DRIVE_BASE_URL or "https://ai.feishu.cn/file").strip().rstrip("/")
    return f"{base}/{file_token}"


def upload_markdown_via_drive(*, md_path: str, file_name: str) -> Optional[str]:
    """使用 drive.v1.file.upload_all 上传 Markdown 文件到飞书云空间。"""

    if upload_file is None:
        logger.error("缺少模块 feishu_drive_upload（或依赖 lark-oapi），无法上传")
        return None

    parent_node = (FEISHU_DRIVE_PARENT_NODE or "").strip()
    if not parent_node:
        logger.error("缺少环境变量 FEISHU_DRIVE_PARENT_NODE")
        return None

    if not os.path.exists(md_path):
        logger.error(f"待上传文件不存在: {md_path}")
        return None

    try:
        file_token = upload_file(
            file_path=md_path,
            file_name=file_name,
            parent_node=parent_node,
        )
        return _build_drive_file_url(file_token=file_token)
    except FeishuDriveUploadError as e:
        logger.exception(f"上传 md 到飞书 Drive 失败: {e}")
        return None


class ArxivPaperProcessor:
    """ArXiv 论文处理器"""

    def __init__(self):
        self.llm_client = LLMClient()
        self.jina_client = JinaReaderClient()

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
        eprint = paper.get_short_id()

        bibtex = (
            f"@article{{{eprint},\n"
            f"  title={{{title}}},\n"
            f"  author={{{author_text}}},\n"
            f"  journal={{arXiv preprint arXiv:{eprint}}},\n"
            f"  year={{{year}}},\n"
            f"  url={{{url}}}\n"
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


def main():
    # 1. 初始化
    processor = ArxivPaperProcessor()
    card_notifier = FeishuCardNotifier()

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

    # 生成 Markdown（日报产物）
    markdown_text = papers_to_markdown(current_date_str, processed_papers)
    md_output_dir = OUTPUT_DIR or output_dir
    md_filename = f"{current_date_str}.md"
    md_path = save_markdown(
        markdown_text, output_dir=md_output_dir, filename=md_filename
    )
    logger.success(f"Markdown 已保存: {md_path}")

    file_url = upload_markdown_via_drive(md_path=md_path, file_name=md_filename)
    if not file_url:
        logger.error("Markdown 上传失败，跳过飞书推送")
        return

    logger.success(f"Markdown 上传成功: {file_url}")
    card_notifier.send_daily_report_card(
        date=TARGET_DATE_STR,
        paper_count=len(processed_papers),
        file_url=file_url,
    )


if __name__ == "__main__":
    main()
