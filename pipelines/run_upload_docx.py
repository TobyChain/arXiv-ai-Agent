"""
run_upload_docx.py

把本地 Markdown 文件写入飞书 Docx 云文档（独立小工具）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from integrations.feishu.write_docx import FeishuDriveUploadError, upload_file

load_dotenv()


def _build_docx_url(document_id: str) -> str:
    base = (os.getenv("FEISHU_DOCX_BASE_URL") or "https://ai.feishu.cn/docx").strip().rstrip("/")
    return f"{base}/{document_id}"


def upload_md_to_docx(*, md_file: Path, folder_token: str) -> Optional[str]:
    if not md_file.exists():
        raise FileNotFoundError(md_file)
    if not folder_token:
        raise FeishuDriveUploadError("缺少 FEISHU_DOCX_FOLDER_TOKEN 环境变量或 --folder-token")

    os.environ["FEISHU_DOCX_CONTENT_TYPE"] = "markdown"

    document_id = upload_file(
        file_path=str(md_file),
        file_name=md_file.name,
        parent_node=folder_token,
    )
    return str(document_id)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, help="Markdown 文件路径")
    parser.add_argument("--folder-token", default=None, help="飞书 Docx 文件夹 token（不传则取 env）")
    args = parser.parse_args(argv)

    md_file = Path(args.md)
    folder_token = (args.folder_token or os.getenv("FEISHU_DOCX_FOLDER_TOKEN") or "").strip()

    logger.info(f"准备上传 Markdown 文件: {md_file}")
    try:
        document_id = upload_md_to_docx(md_file=md_file, folder_token=folder_token)
        docx_url = _build_docx_url(document_id)
        logger.success(f"写入成功 document_id={document_id} url={docx_url}")
    except FeishuDriveUploadError as e:
        logger.exception(f"写入 Docx 失败: {e}")



