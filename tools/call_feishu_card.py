"""兼容层：保留 tools.call_feishu_card 的导入路径，实际实现迁移至 integrations.feishu.send_card。"""

from integrations.feishu.send_card import (  # noqa: F401
    FeishuNotifier,
    send_daily_report,
)

__all__ = ["FeishuNotifier", "send_daily_report"]
