"""write_docx.py

将 Markdown/HTML 内容写入飞书 Docx 文档，流程对齐官方推荐：
1) create document：创建一篇 docx 文档（若目标文档已存在，可跳过）
2) convert：将 Markdown/HTML 转换为 docx blocks
3) create document block descendants：将 blocks 批量插入目标文档

说明：
- 本模块保留 `upload_file()` 作为对外兼容入口，但语义从“上传文件”变为“把文件内容写入 docx”。
- `upload_file()` 返回值为 docx 的 `document_id`。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import lark_oapi as lark
    from lark_oapi.api.docx.v1 import (
        Block,
        ConvertDocumentRequest,
        ConvertDocumentRequestBody,
        ConvertDocumentResponse,
        CreateDocumentBlockDescendantRequest,
        CreateDocumentBlockDescendantRequestBody,
        CreateDocumentRequest,
        CreateDocumentRequestBody,
        CreateDocumentResponse,
    )
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "未安装依赖 lark-oapi。请先安装：pip install lark-oapi\n"
        "或使用项目依赖管理工具（uv/pip/pdm/poetry）安装 pyproject.toml 依赖。"
    ) from e

DEBUG_DOCX_RESPONSE = os.getenv("FEISHU_DOCX_DEBUG_RESPONSE", "").strip() in ("1", "true", "True")


@dataclass(frozen=True)
class FeishuDriveUploadConfig:
    app_id: str
    app_secret: str
    folder_token: str = ""
    document_id: str = ""
    content_type: str = "markdown"
    log_level: lark.LogLevel = lark.LogLevel.ERROR
    max_retries: int = 6
    retry_backoff_seconds: float = 0.5


class FeishuDriveUploadError(RuntimeError):
    pass


def _get_env_required(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise FeishuDriveUploadError(f"缺少环境变量 {name}")
    return value


def default_config(*, parent_node: Optional[str] = None) -> FeishuDriveUploadConfig:
    folder_token = (parent_node or os.getenv("FEISHU_DOCX_FOLDER_TOKEN") or "").strip()
    document_id = (os.getenv("FEISHU_DOCX_DOCUMENT_ID") or "").strip()
    content_type = (os.getenv("FEISHU_DOCX_CONTENT_TYPE") or "markdown").strip().lower()
    if content_type not in ("markdown", "html"):
        raise FeishuDriveUploadError("非法 FEISHU_DOCX_CONTENT_TYPE：仅支持 markdown/html")

    return FeishuDriveUploadConfig(
        app_id=_get_env_required("FEISHU_APP_ID"),
        app_secret=_get_env_required("FEISHU_APP_SECRET"),
        folder_token=folder_token,
        document_id=document_id,
        content_type=content_type,
    )


def _new_client(*, app_id: str, app_secret: str, log_level: lark.LogLevel) -> lark.Client:
    return (
        lark.Client.builder()
        .app_id(app_id)
        .app_secret(app_secret)
        .log_level(log_level)
        .build()
    )


def _safe_raw_json(response: Any) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(response.raw.content)
    except Exception:
        return None


def _raise_lark_error(*, action: str, response: Any) -> None:
    code = getattr(response, "code", None)
    msg = getattr(response, "msg", None)
    try:
        log_id = response.get_log_id()
    except Exception:
        log_id = None

    raw_json = _safe_raw_json(response)
    if raw_json is not None:
        raise FeishuDriveUploadError(
            f"{action} failed: code={code} msg={msg} log_id={log_id} resp={raw_json}"
        )
    raise FeishuDriveUploadError(f"{action} failed: code={code} msg={msg} log_id={log_id}")


def create_docx_document(*, client: lark.Client, folder_token: str, title: str) -> str:
    request: CreateDocumentRequest = (
        CreateDocumentRequest.builder()
        .request_body(
            CreateDocumentRequestBody.builder().folder_token(folder_token).title(title).build()
        )
        .build()
    )

    response: CreateDocumentResponse = client.docx.v1.document.create(request)
    if not response.success():
        _raise_lark_error(action="docx.document.create", response=response)

    raw_json = _safe_raw_json(response)
    if DEBUG_DOCX_RESPONSE and raw_json is not None:
        logger.info(f"docx.document.create raw preview: {raw_json}")

    document_id: Optional[str] = None
    if isinstance(raw_json, dict):
        data = raw_json.get("data")
        if isinstance(data, dict):
            document_id = data.get("document_id") or data.get("documentId")
            if not document_id and isinstance(data.get("document"), dict):
                document_id = data["document"].get("document_id") or data["document"].get("documentId")

    if not document_id and isinstance(getattr(response, "data", None), dict):
        data2 = response.data
        document_id = data2.get("document_id") or data2.get("documentId")
        if not document_id and isinstance(data2.get("document"), dict):
            document_id = data2["document"].get("document_id")

    if not document_id:
        raise FeishuDriveUploadError("docx.document.create success but missing document_id")
    return str(document_id)


def convert_markdown_to_blocks(
    *, client: lark.Client, content_type: str, content: str
) -> Dict[str, Any]:
    request: ConvertDocumentRequest = (
        ConvertDocumentRequest.builder()
        .request_body(ConvertDocumentRequestBody.builder().content_type(content_type).content(content).build())
        .build()
    )

    response: ConvertDocumentResponse = client.docx.v1.document.convert(request)
    if not response.success():
        _raise_lark_error(action="docx.document.convert", response=response)

    raw_json = _safe_raw_json(response)
    if DEBUG_DOCX_RESPONSE and raw_json is not None:
        logger.info(f"docx.document.convert raw preview: {raw_json}")

    data_dict: Dict[str, Any] = {}
    if isinstance(raw_json, dict) and isinstance(raw_json.get("data"), dict):
        data_dict = raw_json["data"]
    elif hasattr(response, "data") and response.data:
        data_dict = response.data if isinstance(response.data, dict) else response.data.__dict__

    blocks_raw = data_dict.get("blocks") or data_dict.get("block_list") or []
    children_ids = data_dict.get("first_level_block_ids") or data_dict.get("children") or []

    if not blocks_raw:
        if isinstance(data_dict.get("children"), list) and data_dict["children"]:
            first = data_dict["children"][0]
            if isinstance(first, dict) and "block_id" in first:
                blocks_raw = data_dict["children"]
                children_ids = [str(b["block_id"]) for b in blocks_raw]

    if not blocks_raw:
        raise FeishuDriveUploadError(
            f"docx.document.convert success but missing blocks. Data keys: {list(data_dict.keys())}"
        )

    raw_block_map = {
        str(b["block_id"]): b for b in blocks_raw if isinstance(b, dict) and "block_id" in b
    }

    if len(children_ids) == 1:
        root_id = str(children_ids[0])
        root_block = raw_block_map.get(root_id)
        if root_block and root_block.get("block_type") == 1:
            root_children = root_block.get("children")
            if isinstance(root_children, list) and root_children:
                children_ids = [str(cid) for cid in root_children]

    minimal_blocks: List[Dict[str, Any]] = []
    allowed_fields = {
        "block_type",
        "parent_id",
        "text",
        "heading1",
        "heading2",
        "heading3",
        "heading4",
        "heading5",
        "heading6",
        "bullet",
        "ordered",
        "quote",
        "code",
        "divider",
        "callout",
        "image",
        "equation",
        "todo",
        "bitable",
        "diagram",
        "page",
        "file",
        "mindnote",
        "grid",
        "grid_column",
        "table",
        "table_cell",
        "children",
    }

    for bid, b_dict in raw_block_map.items():
        minimal = {"block_id": bid}
        for k, v in b_dict.items():
            if k in allowed_fields:
                minimal[k] = v
        minimal_blocks.append(minimal)

    return {"blocks": minimal_blocks, "children": [str(cid) for cid in children_ids if cid]}


def insert_blocks_to_document(
    *,
    client: lark.Client,
    document_id: str,
    blocks: List[Dict[str, Any]],
    children_ids: List[str],
    parent_block_id: Optional[str] = None,
    index: Optional[int] = None,
    document_revision_id: int = -1,
) -> None:
    target_parent_block_id = (parent_block_id or "").strip() or document_id
    block_map = {b["block_id"]: b for b in blocks}

    def get_reachable_descendants(root_ids: List[str]) -> List[Block]:
        reachable_blocks = []
        visited = set()
        stack = list(root_ids)
        while stack:
            bid = stack.pop()
            if bid in visited:
                continue
            visited.add(bid)
            b_dict = block_map.get(bid)
            if not b_dict:
                continue
            try:
                blk: Block = lark.JSON.unmarshal(json.dumps(b_dict), Block)
                reachable_blocks.append(blk)
                cids = b_dict.get("children")
                if isinstance(cids, list):
                    for cid in cids:
                        stack.append(str(cid))
            except Exception as e:
                logger.warning(f"unmarshal block failed: {e}, block_id: {bid}")
        return reachable_blocks

    if not children_ids:
        return

    batch_size = 1000
    current_index = index

    for start in range(0, len(children_ids), batch_size):
        end = min(start + batch_size, len(children_ids))
        batch_children_ids = children_ids[start:end]
        batch_descendants = get_reachable_descendants(batch_children_ids)
        if not batch_descendants:
            continue

        rb_builder = (
            CreateDocumentBlockDescendantRequestBody.builder()
            .children_id(batch_children_ids)
            .descendants(batch_descendants)
        )
        if current_index is not None:
            rb_builder.index(current_index)

        request: CreateDocumentBlockDescendantRequest = (
            CreateDocumentBlockDescendantRequest.builder()
            .document_id(document_id)
            .block_id(target_parent_block_id)
            .document_revision_id(document_revision_id)
            .request_body(rb_builder.build())
            .build()
        )
        response = client.docx.v1.document_block_descendant.create(request)
        if not response.success():
            _raise_lark_error(action="docx.document_block_descendant.create", response=response)

        if current_index is not None:
            current_index += len(batch_children_ids)


def _split_markdown_by_papers(content: str, batch_size: int = 10) -> List[str]:
    parts = re.split(r"^---\s*$", content, flags=re.MULTILINE)
    if len(parts) <= 1:
        return [content]

    header = parts[0].strip()
    papers = [p.strip() for p in parts[1:] if p.strip()]

    chunks = []
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        batch_content = "\n\n---\n\n".join(batch)
        if i == 0:
            if header:
                chunks.append(header + "\n\n---\n\n" + batch_content)
            else:
                chunks.append("---\n\n" + batch_content)
        else:
            chunks.append("---\n\n" + batch_content)

    return chunks


def upload_markdown_to_docx(
    *,
    file_path: str,
    file_name: str,
    app_id: str,
    app_secret: str,
    folder_token: str,
    document_id: str = "",
    content_type: str = "markdown",
    log_level: lark.LogLevel = lark.LogLevel.ERROR,
    max_retries: int = 6,
    retry_backoff_seconds: float = 0.5,
) -> str:
    if not os.path.exists(file_path):
        raise FeishuDriveUploadError(f"内容文件不存在: {file_path}")

    content = open(file_path, "r", encoding="utf-8", errors="ignore").read()
    if not content.strip():
        raise FeishuDriveUploadError("不可写入空内容")

    if content_type not in ("markdown", "html"):
        raise FeishuDriveUploadError("content_type 仅支持 markdown/html")

    if not document_id and not folder_token:
        raise FeishuDriveUploadError("缺少目标文档：请提供 document_id 或 folder_token（用于创建文档）")

    client = _new_client(app_id=app_id, app_secret=app_secret, log_level=log_level)

    content_chunks = [content]
    if content_type == "markdown":
        content_chunks = _split_markdown_by_papers(content, batch_size=10)
        logger.info(f"Markdown 内容已拆分为 {len(content_chunks)} 个分片，根据---拆分")

    backoff = retry_backoff_seconds
    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            target_document_id = document_id
            if not target_document_id:
                target_document_id = create_docx_document(
                    client=client,
                    folder_token=folder_token,
                    title=os.path.splitext(file_name)[0] or file_name,
                )

            for idx, chunk in enumerate(content_chunks):
                result = convert_markdown_to_blocks(
                    client=client, content_type=content_type, content=chunk
                )
                insert_blocks_to_document(
                    client=client,
                    document_id=target_document_id,
                    blocks=result["blocks"],
                    children_ids=result["children"],
                    parent_block_id=target_document_id,
                    index=None,
                    document_revision_id=-1,
                )
                if len(content_chunks) > 1:
                    logger.info(f"已完成分片 {idx + 1}/{len(content_chunks)} 的写入")

            return str(target_document_id)

        except FeishuDriveUploadError:
            raise
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                logger.warning(f"docx 写入异常重试 {attempt}/{max_retries}，sleep={backoff}s: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue
            raise FeishuDriveUploadError(f"docx 写入异常且重试耗尽: {e}")

    raise FeishuDriveUploadError(f"docx 写入失败且重试耗尽: {last_error}")


def upload_file(
    *,
    file_path: str,
    file_name: str,
    parent_node: Optional[str] = None,
    config: Optional[FeishuDriveUploadConfig] = None,
) -> str:
    cfg = config or default_config(parent_node=parent_node)
    return upload_markdown_to_docx(
        file_path=file_path,
        file_name=file_name,
        app_id=cfg.app_id,
        app_secret=cfg.app_secret,
        folder_token=cfg.folder_token,
        document_id=cfg.document_id,
        content_type=cfg.content_type,
        log_level=cfg.log_level,
        max_retries=cfg.max_retries,
        retry_backoff_seconds=cfg.retry_backoff_seconds,
    )


