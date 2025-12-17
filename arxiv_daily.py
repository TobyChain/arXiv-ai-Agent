import os
import json
import arxiv
import datetime
from dotenv import load_dotenv
from loguru import logger

# 导入自定义工具模块
from call_llm import LLMClient
from call_feishu_card import FeishuCardNotifier
from call_jina import JinaReaderClient

# 加载环境变量
load_dotenv()

# ================= 配置区域 =================
WEB_SERVER_URL = os.getenv("WEB_SERVER_URL", "http://localhost:8000")
ARXIV_SUBJECT = os.getenv("ARXIV_SUBJECT", "cs.AI")  # ArXiv 主题代码
CURRENT_DATE = datetime.datetime.now()
TARGET_DATE_STR = CURRENT_DATE.strftime("%a, %d %b %Y")
MAX_PAPERS = 200
# ===========================================


class ArxivPaperProcessor:
    """ArXiv 论文处理器"""

    def __init__(self):
        self.llm_client = LLMClient()
        self.jina_client = JinaReaderClient()

    def fetch_jina_data(self, subject: str = "cs.AI"):
        """
        调用 Jina Reader API 获取 ArXiv 列表数据

        Args:
            subject: ArXiv 主题代码
        """
        return self.jina_client.fetch_arxiv_list(subject=subject, skip=0, show=250)

    def parse_jina_response(self, json_data, target_date):
        """
        从 Jina Reader 的 JSON 响应中提取指定日期的 ArXiv ID
        """
        return self.jina_client.parse_arxiv_ids(json_data, target_date)

    def fetch_arxiv_metadata(self, arxiv_ids):
        """
        使用 arxiv 库批量获取元数据
        """
        if not arxiv_ids:
            return []
        logger.info("正在从 ArXiv 获取元数据...")
        arxiv_ids = arxiv_ids[:MAX_PAPERS]  # 限制最大数量

        client = arxiv.Client()
        search = arxiv.Search(id_list=arxiv_ids)
        results = list(client.results(search))
        return results

    def generate_bibtex(self, paper):
        """
        根据 arxiv 元数据生成 BibTeX
        """
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

    def translate_abstract(self, abstract):
        """
        调用 LLM 翻译摘要
        """
        return self.llm_client.translate_abstract(abstract, domain="AI")


class HTMLReportGenerator:
    def generate_html(self, papers_data, date_str, output_filename):
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ArXiv AI Daily Report - {date_str}</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <script>
                tailwind.config = {{
                    theme: {{
                        extend: {{
                            colors: {{
                                primary: '#2563eb',
                                secondary: '#475569',
                            }}
                        }}
                    }}
                }}
            </script>
        </head>
        <body class="bg-gray-50 min-h-screen font-sans text-gray-800">
            <div class="max-w-4xl mx-auto px-4 py-8">
                <!-- Header -->
                <header class="text-center mb-12">
                    <div class="inline-block p-3 rounded-full bg-blue-100 text-blue-600 mb-4">
                        <i class="fas fa-robot text-3xl"></i>
                    </div>
                    <h1 class="text-4xl font-bold text-gray-900 mb-2">ArXiv AI Daily Report</h1>
                    <p class="text-gray-500 text-lg">{date_str}</p>
                </header>

                <!-- Papers List -->
                <div class="space-y-8">
        """

        for item in papers_data:
            html_content += f"""
            <article class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition-shadow duration-300">
                <div class="p-6">
                    <h2 class="text-xl font-bold text-gray-900 mb-2 leading-tight hover:text-blue-600 transition-colors">
                        <a href="{item["abs_url"]}" target="_blank">{item["title"]}</a>
                    </h2>
                    
                    <div class="flex items-center text-sm text-gray-500 mb-4">
                        <i class="fas fa-users mr-2"></i>
                        <span class="italic truncate">{item["authors"]}</span>
                    </div>

                    <div class="flex gap-3 mb-6">
                        <a href="{item["pdf_url"]}" target="_blank" 
                           class="inline-flex items-center px-3 py-1.5 rounded-md bg-red-50 text-red-600 text-sm font-medium hover:bg-red-100 transition-colors">
                            <i class="fas fa-file-pdf mr-2"></i> PDF
                        </a>
                        <a href="{item["abs_url"]}" target="_blank"
                           class="inline-flex items-center px-3 py-1.5 rounded-md bg-gray-100 text-gray-700 text-sm font-medium hover:bg-gray-200 transition-colors">
                            <i class="fas fa-external-link-alt mr-2"></i> Abstract
                        </a>
                    </div>

                    <div class="bg-blue-50 rounded-lg p-4 mb-4 border-l-4 border-blue-500">
                        <h3 class="text-sm font-bold text-blue-900 mb-2 flex items-center">
                            <i class="fas fa-language mr-2"></i> 中文摘要
                        </h3>
                        <p class="text-gray-700 text-sm leading-relaxed text-justify">
                            {item["trans_abs"]}
                        </p>
                    </div>

                    <details class="group">
                        <summary class="flex items-center cursor-pointer text-sm text-gray-500 hover:text-gray-700 select-none">
                            <i class="fas fa-code mr-2 transition-transform group-open:rotate-90"></i>
                            <span>BibTeX</span>
                        </summary>
                        <div class="mt-3 bg-gray-900 rounded-lg p-4 overflow-x-auto">
                            <pre class="text-xs text-gray-300 font-mono whitespace-pre-wrap">{item["bibtex"]}</pre>
                        </div>
                    </details>
                </div>
            </article>
            """

        html_content += """
                </div>

                <!-- Footer -->
                <footer class="text-center mt-12 text-gray-400 text-sm pb-8">
                    <p>Generated by ArXiv AI Agent • Powered by DeepSeek & Jina AI</p>
                </footer>
            </div>
        </body>
        </html>
        """

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.success(f"HTML 报告已生成: {output_filename}")
        return os.path.abspath(output_filename)


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
        trans_abs = processor.translate_abstract(paper.summary)
        bib = processor.generate_bibtex(paper)

        # 获取作者信息
        authors_list = [author.name for author in paper.authors]
        first_author = authors_list[0] if authors_list else "Unknown"

        processed_papers.append(
            {
                "title": paper.title,
                "authors": authors_list,
                "first_author": first_author,
                "abs_url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "trans_abs": trans_abs,
                "bibtex": bib,
                "published": paper.published.strftime("%Y-%m-%d"),
                "summary": paper.summary,
            }
        )

    # 5. 保存数据
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

    # 生成 HTML 报告
    html_gen = HTMLReportGenerator()
    filename = f"Arxiv_Report_{TARGET_DATE_STR.replace(' ', '_').replace(',', '')}.html"
    output_path = os.path.join(output_dir, filename)

    # 转换作者列表为字符串格式用于 HTML 显示
    html_papers = []
    for p in processed_papers:
        p_copy = p.copy()
        p_copy["authors"] = ", ".join(p["authors"])
        html_papers.append(p_copy)

    html_gen.generate_html(html_papers, TARGET_DATE_STR, output_path)

    # 6. 发送飞书通知（使用卡片格式）
    # 构建可访问的 HTTP URL
    html_filename = os.path.basename(output_path)
    report_url = f"{WEB_SERVER_URL}/database/{html_filename}"

    # 同时提供 Web 界面的链接
    web_interface_url = f"{WEB_SERVER_URL}/?date={current_date_str}"

    # 发送交互式卡片消息
    card_notifier.send_daily_report_card(
        date=TARGET_DATE_STR,
        paper_count=len(processed_papers),
        html_url=report_url,
        web_url=web_interface_url,
    )


if __name__ == "__main__":
    main()
