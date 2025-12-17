"""
飞书 Webhook 调用工具
用于向飞书群发送消息通知
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


class FeishuNotifier:
    """飞书消息通知器"""

    def __init__(self, webhook_url=None, secret=None):
        self.webhook_url = webhook_url or os.getenv("FEISHU_WEBHOOK_URL")
        self.secret = secret or os.getenv("FEISHU_SECRET")

    def gen_sign(self, timestamp: str, secret: str) -> str:
        """
        生成飞书签名

        Args:
            timestamp: 时间戳字符串
            secret: 密钥

        Returns:
            签名字符串
        """
        # 拼接 timestamp 和 secret
        string_to_sign = "{}\n{}".format(timestamp, secret)
        hmac_code = hmac.new(
            string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
        ).digest()
        # 对结果进行 base64 编码
        sign = base64.b64encode(hmac_code).decode("utf-8")
        return sign

    def send_text(self, title: str, text: str, url: str = None) -> bool:
        """
        发送文本消息

        Args:
            title: 消息标题
            text: 消息内容
            url: 可选的链接 URL

        Returns:
            是否发送成功
        """
        timestamp = str(int(time.time()))
        sign = self.gen_sign(timestamp, self.secret)

        content_text = text
        if url:
            content_text += f"\n\n📄 查看完整报告: {url}"

        msg = {
            "timestamp": timestamp,
            "sign": sign,
            "msg_type": "text",
            "content": {"text": f"{title}\n\n{content_text}"},
        }

        try:
            response = requests.post(self.webhook_url, json=msg)
            response.raise_for_status()
            result = response.json()
            if result.get("code") == 0:
                logger.success("飞书消息发送成功")
                return True
            else:
                logger.error(f"飞书消息发送失败: {result}")
                return False
        except Exception as e:
            logger.exception(f"发送飞书请求异常: {e}")
            return False

    def send_rich_text(self, title: str, content: list) -> bool:
        """
        发送富文本消息

        Args:
            title: 消息标题
            content: 富文本内容列表

        Returns:
            是否发送成功
        """
        timestamp = str(int(time.time()))
        sign = self.gen_sign(timestamp, self.secret)

        msg = {
            "timestamp": timestamp,
            "sign": sign,
            "msg_type": "post",
            "content": {"post": {"zh_cn": {"title": title, "content": content}}},
        }

        try:
            response = requests.post(self.webhook_url, json=msg)
            response.raise_for_status()
            result = response.json()
            if result.get("code") == 0:
                logger.success("飞书富文本消息发送成功")
                return True
            else:
                logger.error(f"飞书富文本消息发送失败: {result}")
                return False
        except Exception as e:
            logger.exception(f"发送飞书富文本请求异常: {e}")
            return False


# 单例模式
_default_notifier = None


def get_notifier() -> FeishuNotifier:
    """获取默认飞书通知器"""
    global _default_notifier
    if _default_notifier is None:
        _default_notifier = FeishuNotifier()
    return _default_notifier


def send_message(title: str, text: str, url: str = None) -> bool:
    """快捷发送文本消息"""
    return get_notifier().send_text(title, text, url)
