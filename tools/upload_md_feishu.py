"""兼容层：保留 tools.upload_md_feishu 的导入路径。

真实实现已迁移至 `integrations.feishu.write_docx`。
"""

from integrations.feishu.write_docx import (  # noqa: F401
    FeishuDriveUploadConfig,
    FeishuDriveUploadError,
    default_config,
    upload_file,
    upload_markdown_to_docx,
)

__all__ = [
    "FeishuDriveUploadConfig",
    "FeishuDriveUploadError",
    "default_config",
    "upload_markdown_to_docx",
    "upload_file",
]



