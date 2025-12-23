"""
飞书通知工具模块
支持发送多种类型的飞书卡片消息
"""

import os
import time
import hmac
import hashlib
import base64
import requests
from typing import Any, Dict, Optional
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class FeishuNotifier:
    """飞书消息通知器"""

    def __init__(self, webhook_url: Optional[str] = None, secret: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("FEISHU_WEBHOOK_URL")
        self.secret = secret or os.getenv("FEISHU_SECRET")

    def _gen_sign(self, timestamp: str) -> str:
        """生成飞书签名"""
        if not self.secret:
            return ""
        string_to_sign = "{}\n{}".format(timestamp, self.secret)
        hmac_code = hmac.new(
            string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
        ).digest()
        sign = base64.b64encode(hmac_code).decode("utf-8")
        return sign

    def send_card(self, card_content: Dict[str, Any]) -> bool:
        """
        发送自定义卡片消息

        Args:
            card_content: 飞书卡片 JSON 内容
        """
        if not self.webhook_url:
            logger.warning("未配置 FEISHU_WEBHOOK_URL，跳过发送")
            return False

        timestamp = str(int(time.time()))

        msg = {
            "msg_type": "interactive",
            "card": card_content,
        }

        if self.secret:
            msg["timestamp"] = timestamp
            msg["sign"] = self._gen_sign(timestamp)

        try:
            response = requests.post(self.webhook_url, json=msg, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result.get("code") == 0:
                logger.success("飞书卡片消息发送成功")
                return True
            else:
                logger.error(f"飞书卡片消息发送失败: {result}")
                return False
        except Exception as e:
            logger.exception(f"发送飞书卡片请求异常: {e}")
            return False

    def send_daily_report(
        self,
        date: str,
        paper_count: int,
        file_url: str,
        title: str = "ArXiv AI Daily Report",
        template: str = "blue",
    ) -> bool:
        """
        发送每日报告卡片（内置模板）
        """
        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": template,
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {"content": f"**📅 报告日期**\n{date}", "tag": "lark_md"},
                },
                {
                    "tag": "div",
                    "text": {
                        "content": f"**📚 今日论文数量**\n{paper_count} 篇",
                        "tag": "lark_md",
                    },
                },
                {
                    "tag": "div",
                    "text": {
                        "content": "💡 点击下方按钮查看详细的论文翻译与分析报告（Markdown 格式）。",
                        "tag": "lark_md",
                    },
                },
                {"tag": "hr"},
                {
                    "tag": "action",
                    "actions": [
                        {
                            "tag": "button",
                            "text": {
                                "tag": "plain_text",
                                "content": "查看 Markdown 报告",
                            },
                            "type": "primary",
                            "url": file_url,
                        }
                    ],
                },
            ],
        }
        return self.send_card(card)


# 兼容性快捷函数
def send_daily_report(date: str, paper_count: int, file_url: str) -> bool:
    notifier = FeishuNotifier()
    return notifier.send_daily_report(date, paper_count, file_url)
