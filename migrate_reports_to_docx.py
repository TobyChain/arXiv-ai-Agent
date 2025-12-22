"""migrate_reports_to_docx.py

一次性迁移脚本：
1) 将已存在的 HTML 报告批量转为 Markdown 文件
2) 使用 Drive v1 `upload_all` 将 Markdown 文件上传至云空间指定目录

也支持：直接把处理后的 JSON（论文列表）渲染成 Markdown 并上传。

用法示例：
    # 迁移历史 HTML：生成 md 并上传
    python migrate_reports_to_docx.py html --database-dir ./database --out-md-dir ./database_md \
        --folder-token LVrTfA0lOlFRPDdKge3c1qBLn5e

    # 基于某一天 JSON 生成 Markdown 并上传
    python migrate_reports_to_docx.py json --json-file ./database/2025-12-20.json --date 2025-12-20 \
        --out-md ./database_md/2025-12-20.md --folder-token LVrTfA0lOlFRPDdKge3c1qBLn5e

环境变量：
    FEISHU_APP_ID / FEISHU_APP_SECRET 必填
    （可选）FEISHU_DRIVE_PARENT_NODE 可作为默认 folder token
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger
from markdownify import markdownify as md

from feishu_drive_upload import FeishuDriveUploadError, upload_file
from md_report import papers_to_markdown

# 加载环境变量
load_dotenv()


def html_to_markdown(html_text: str) -> str:
    # markdownify 会做较通用的 HTML->MD 转换，足够用于历史迁移
    return md(html_text, heading_style="ATX")


def upload_markdown_file(*, md_path: Path, parent_node: Optional[str]) -> Optional[str]:
    try:
        return upload_file(
            file_path=str(md_path),
            file_name=md_path.name,
            parent_node=parent_node,
        )
    except FeishuDriveUploadError as e:
        logger.exception(f"upload_all 上传失败: {md_path.name}: {e}")
        return None


def cmd_html(args: argparse.Namespace) -> int:
    db_dir = Path(args.database_dir)
    out_dir = Path(args.out_md_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(db_dir.glob("*.html"))
    if not html_files:
        logger.warning(f"未找到 html 文件: {db_dir}")
        return 0

    folder_token = args.folder_token

    for i, html_path in enumerate(html_files, start=1):
        logger.info(f"[{i}/{len(html_files)}] 迁移: {html_path.name}")
        html_text = html_path.read_text(encoding="utf-8", errors="ignore")
        markdown_text = html_to_markdown(html_text)

        md_path = out_dir / (html_path.stem + ".md")
        md_path.write_text(markdown_text, encoding="utf-8")

        file_token = upload_markdown_file(md_path=md_path, parent_node=folder_token)
        if file_token:
            logger.success(f"上传成功 file_token={file_token}")

    return 0


def cmd_json(args: argparse.Namespace) -> int:
    json_file = Path(args.json_file)
    if not json_file.exists():
        raise FileNotFoundError(json_file)

    papers: List[Dict[str, Any]] = json.loads(json_file.read_text(encoding="utf-8"))
    markdown_text = papers_to_markdown(args.date, papers)

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(markdown_text, encoding="utf-8")

    md_path: Optional[Path] = None
    if args.out_md:
        md_path = Path(args.out_md)
    else:
        md_path = Path.cwd() / f"{args.date}.md"
        md_path.write_text(markdown_text, encoding="utf-8")

    file_token = upload_markdown_file(md_path=md_path, parent_node=args.folder_token)
    if file_token:
        logger.success(f"上传成功 file_token={file_token}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_html = sub.add_parser("html", help="migrate database/*.html -> md + docx")
    p_html.add_argument("--database-dir", required=True)
    p_html.add_argument("--out-md-dir", required=True)
    p_html.add_argument("--folder-token", default=None)
    p_html.add_argument("--title-prefix", default="ArXiv Report - ")
    p_html.set_defaults(func=cmd_html)

    p_json = sub.add_parser("json", help="json -> md + docx")
    p_json.add_argument("--json-file", required=True)
    p_json.add_argument("--date", required=True, help="YYYY-MM-DD")
    p_json.add_argument("--out-md", default=None)
    p_json.add_argument("--folder-token", default=None)
    p_json.add_argument("--title", default=None)
    p_json.set_defaults(func=cmd_json)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
