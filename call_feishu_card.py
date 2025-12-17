"""
增强版飞书通知工具
支持富文本卡片格式的消息推送
"""

import os
import time
import hmac
import hashlib
import base64
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class FeishuCardNotifier:
    """飞书卡片消息通知器"""

    def __init__(self, webhook_url=None, secret=None):
        self.webhook_url = webhook_url or os.getenv("FEISHU_WEBHOOK_URL")
        self.secret = secret or os.getenv("FEISHU_SECRET")

    def gen_sign(self, timestamp: str, secret: str) -> str:
        """生成飞书签名"""
        string_to_sign = "{}\n{}".format(timestamp, secret)
        hmac_code = hmac.new(
            string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
        ).digest()
        sign = base64.b64encode(hmac_code).decode("utf-8")
        return sign

    def send_daily_report_card(
        self, date: str, paper_count: int, html_url: str, web_url: str
    ) -> bool:
        """
        发送每日报告卡片消息

        Args:
            date: 报告日期
            paper_count: 论文数量
            html_url: HTML 报告链接
            web_url: Web 界面链接

        Returns:
            是否发送成功
        """
        timestamp = str(int(time.time()))
        sign = self.gen_sign(timestamp, self.secret)

        # 构建交互式卡片
        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "blue",
                "title": {"content": "🤖 ArXiv AI Daily Report", "tag": "plain_text"},
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {"content": f"**📅 日期**\n{date}", "tag": "lark_md"},
                },
                {
                    "tag": "div",
                    "text": {
                        "content": f"**📚 今日更新论文**\n{paper_count} 篇",
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
                                "content": "📄 查看 HTML 报告",
                                "tag": "plain_text",
                            },
                            "url": html_url,
                            "type": "primary",
                        },
                        {
                            "tag": "button",
                            "text": {
                                "content": "🌐 打开 Web 界面",
                                "tag": "plain_text",
                            },
                            "url": web_url,
                            "type": "default",
                        },
                    ],
                },
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": "💡 提示：请确保 Web 服务器已启动 (运行 ./run_server.sh)",
                        }
                    ],
                },
            ],
        }

        msg = {
            "timestamp": timestamp,
            "sign": sign,
            "msg_type": "interactive",
            "card": card,
        }

        try:
            response = requests.post(self.webhook_url, json=msg)
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


def send_daily_report(date: str, paper_count: int, html_url: str, web_url: str) -> bool:
    """快捷发送每日报告"""
    notifier = FeishuCardNotifier()
    return notifier.send_daily_report_card(date, paper_count, html_url, web_url)
