"""feishu_drive_upload.py

基于飞书开放平台 Drive v1 的 upload_all 接口上传文件。

对齐官方文档约束：
- 文件大小 <= 20MB，且不可为空（否则需使用分片上传接口）
- parent_type 固定 explorer
- size 单位字节
- checksum 可不传（本实现按需求不传，即为空）
- 可能返回 code=1061045（can retry），建议重试

环境变量：
- FEISHU_APP_ID / FEISHU_APP_SECRET: 必填（SDK 会自动获取 token）
- FEISHU_DRIVE_PARENT_NODE: 默认上传目标文件夹 token（可选；调用方也可显式传入）
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger

try:
    import lark_oapi as lark
    from lark_oapi.api.drive.v1 import (
        UploadAllFileRequest,
        UploadAllFileRequestBody,
        UploadAllFileResponse,
    )
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "未安装依赖 lark-oapi。请先安装：pip install lark-oapi\n"
        "或使用项目依赖管理工具（uv/pip/pdm/poetry）安装 pyproject.toml 依赖。"
    ) from e


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEBUG_UPLOAD_RESPONSE = os.getenv("FEISHU_DRIVE_DEBUG_UPLOAD_RESPONSE", "").strip() in (
    "1",
    "true",
    "True",
)


@dataclass(frozen=True)
class FeishuDriveUploadConfig:
    app_id: str
    app_secret: str
    parent_node: str
    parent_type: str = "explorer"
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
    return FeishuDriveUploadConfig(
        app_id=_get_env_required("FEISHU_APP_ID"),
        app_secret=_get_env_required("FEISHU_APP_SECRET"),
        parent_node=(
            parent_node or os.getenv("FEISHU_DRIVE_PARENT_NODE") or ""
        ).strip(),
    )


def upload_file_upload_all(
    *,
    file_path: str,
    file_name: str,
    parent_node: str,
    app_id: str,
    app_secret: str,
    parent_type: str = "explorer",
    log_level: lark.LogLevel = lark.LogLevel.ERROR,
    max_retries: int = 6,
    retry_backoff_seconds: float = 0.5,
) -> str:
    """上传文件并返回 file_token。

    注意：upload_all 成功响应体里只保证包含 file_token，不保证有可直接访问的 URL。
    """

    if not os.path.exists(file_path):
        raise FeishuDriveUploadError(f"待上传文件不存在: {file_path}")

    size = os.path.getsize(file_path)
    if size <= 0:
        raise FeishuDriveUploadError("不可上传空文件（size=0）")
    if size > MAX_UPLOAD_BYTES:
        raise FeishuDriveUploadError(
            f"文件大小超限（{size} bytes > {MAX_UPLOAD_BYTES} bytes），请使用分片上传接口"
        )

    client = (
        lark.Client.builder()
        .app_id(app_id)
        .app_secret(app_secret)
        .log_level(log_level)
        .build()
    )

    backoff = retry_backoff_seconds
    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            with open(file_path, "rb") as f:
                body_builder = (
                    UploadAllFileRequestBody.builder()
                    .file_name(file_name)
                    .parent_type(parent_type)
                    .parent_node(parent_node)
                    .size(str(size))
                    .file(f)
                )
                request: UploadAllFileRequest = (
                    UploadAllFileRequest.builder()
                    .request_body(body_builder.build())
                    .build()
                )

                response: UploadAllFileResponse = client.drive.v1.file.upload_all(
                    request
                )

            if not response.success():
                code = response.code
                msg = response.msg
                log_id = response.get_log_id()
                last_error = f"code={code} msg={msg} log_id={log_id}"

                # 官方文档：1061045 can retry
                if code == 1061045 and attempt < max_retries:
                    logger.warning(
                        f"upload_all can retry(1061045)，重试 {attempt}/{max_retries}，sleep={backoff}s"
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    continue

                try:
                    raw = json.loads(response.raw.content)
                    raise FeishuDriveUploadError(
                        f"upload_all failed: {last_error} resp={json.dumps(raw, ensure_ascii=False)}"
                    )
                except Exception:
                    raise FeishuDriveUploadError(f"upload_all failed: {last_error}")

            # 优先解析 raw JSON（更贴近官方响应：{"code":0,"msg":"success","data":{"file_token":"..."}}）
            file_token = None
            raw_json = None
            try:
                raw_json = json.loads(response.raw.content)
            except Exception:
                raw_json = None

            if DEBUG_UPLOAD_RESPONSE:
                try:
                    safe_preview = (
                        raw_json if isinstance(raw_json, dict) else str(raw_json)
                    )
                    logger.info(f"upload_all raw preview: {safe_preview}")
                except Exception:
                    pass

            if isinstance(raw_json, dict):
                data2 = raw_json.get("data")
                if isinstance(data2, dict):
                    file_token = data2.get("file_token") or data2.get("fileToken")
                if not file_token:
                    file_token = raw_json.get("file_token") or raw_json.get("fileToken")

            # 兜底：从 SDK 的 response.data 再尝试一次
            if not file_token:
                data = response.data
                if isinstance(data, dict):
                    file_token = data.get("file_token") or data.get("fileToken")
                    if not file_token and isinstance(data.get("data"), dict):
                        file_token = data["data"].get("file_token")

                    if not file_token and isinstance(data.get("file"), dict):
                        file_token = data["file"].get("token") or data["file"].get(
                            "file_token"
                        )

            if not file_token:
                raise FeishuDriveUploadError(
                    "upload_all success but missing file_token"
                )

            return str(file_token)

        except FeishuDriveUploadError:
            raise
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                logger.warning(
                    f"upload_all 异常重试 {attempt}/{max_retries}，sleep={backoff}s: {e}"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue
            raise FeishuDriveUploadError(f"upload_all 异常且重试耗尽: {e}")

    raise FeishuDriveUploadError(f"upload_all 失败且重试耗尽: {last_error}")


def upload_file(
    *,
    file_path: str,
    file_name: str,
    parent_node: Optional[str] = None,
    config: Optional[FeishuDriveUploadConfig] = None,
) -> str:
    cfg = config or default_config(parent_node=parent_node)
    if not cfg.parent_node:
        raise FeishuDriveUploadError(
            "缺少 parent_node：请传入 parent_node 或设置 FEISHU_DRIVE_PARENT_NODE"
        )

    return upload_file_upload_all(
        file_path=file_path,
        file_name=file_name,
        parent_node=cfg.parent_node,
        app_id=cfg.app_id,
        app_secret=cfg.app_secret,
        parent_type=cfg.parent_type,
        log_level=cfg.log_level,
        max_retries=cfg.max_retries,
        retry_backoff_seconds=cfg.retry_backoff_seconds,
    )
