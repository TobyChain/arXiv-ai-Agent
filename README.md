# ArXiv AI Agent

一个基于 AI 的 ArXiv 论文每日速递脚本：自动抓取、翻译/分析、生成 Markdown，并上传到飞书云空间后通过机器人卡片推送链接。

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-success)](https://tobychain.github.io/arXiv-ai-Agent/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## ✨ 功能特性

- 🤖 **自动抓取**：每日自动获取 ArXiv 最新论文
- 🌐 **智能翻译**：使用 DeepSeek 模型翻译摘要为中文
- 📝 **生成 Markdown**：输出结构化日报（含关键词/子主题/推荐度）
- ☁️ **飞书云空间上传**：使用 Drive v1 `upload_all` 上传 Markdown 文件
- 📱 **飞书推送**：通过群机器人卡片推送可点击链接

## 项目结构

```
arXiv-ai-Agent/
├── arxiv_daily.py         # 主程序（自动化抓取）
├── call_llm.py           # LLM 翻译模块
├── call_jina.py          # Jina API 客户端
├── call_feishu.py        # 飞书通知（基础，已弃用）
├── call_feishu_card.py   # 飞书卡片通知
├── arxiv_subjects.py     # 主题配置
├── feishu_drive_upload.py # Drive upload_all 上传封装
├── md_report.py          # 论文列表 -> Markdown 渲染
├── database/             # 数据存储
├── .env                  # 环境变量（不提交）
└── README.md
```

## 快速开始

### 1. 环境配置

创建 `.env` 文件并配置以下变量：

```env
# LLM API 配置
API_KEY="your_openai_api_key"
BASE_URL="https://api.openai.com/v1"
MODEL_NAME="gpt-3.5-turbo"

# Jina Reader API
JINA_API_KEY="your_jina_api_key"

# 飞书 Webhook
FEISHU_WEBHOOK_URL="your_feishu_webhook_url"
FEISHU_SECRET="your_feishu_secret"

# 飞书开放平台（自建应用）：用于 Drive upload_all 文件上传
FEISHU_APP_ID="cli_xxx"
FEISHU_APP_SECRET="xxx"

# Drive 上传目标文件夹 token（parent_node）
FEISHU_DRIVE_PARENT_NODE="LVrTfA0lOlFRPDdKge3c1qBLn5e"

# 生成可点击链接用的前缀（不同租户可能不同；如果不确定，可先留空，仅上传不推送）
FEISHU_DRIVE_BASE_URL="https://your-tenant.feishu.cn/drive/file"
```

### 2. 安装依赖

本项目使用 `uv` 进行 Python 包管理。依赖已在项目根目录的 `pyproject.toml` 中配置。

```bash
# 如果没有安装 uv，先安装
curl -LsSf https://astral.sh/uv/install.sh | sh

# uv 会自动使用 pyproject.toml 中的依赖
```

### 3. 运行任务

```bash
# 使用 uv 执行每日论文抓取和处理
uv run python arxiv_daily.py

# 或使用提供的快捷脚本
./run.sh
```

## 模块说明

### call_llm.py - LLM 调用工具

提供大语言模型调用功能，支持论文摘要翻译和通用对话。

```python
from call_llm import LLMClient, translate_text

# 使用默认客户端
client = LLMClient()
translation = client.translate_abstract("Original abstract...", domain="AI")

# 或使用快捷函数
translation = translate_text("Original abstract...")
```

### call_feishu.py - 飞书推送工具

封装飞书 Webhook API，支持文本和富文本消息推送。

```python
from call_feishu import FeishuNotifier, send_message

# 发送简单文本消息
notifier = FeishuNotifier()
notifier.send_text("标题", "消息内容", "https://link.url")

# 或使用快捷函数
send_message("标题", "消息内容")
```

### call_jina.py - Jina Reader API 工具

调用 Jina Reader API 抓取和解析 ArXiv 论文列表。

```python
from call_jina import JinaReaderClient, fetch_arxiv_papers

# 获取指定日期的论文
client = JinaReaderClient()
data = client.fetch_arxiv_list(skip=0, show=250)
ids = client.parse_arxiv_ids(data, "Wed, 17 Dec 2025")

# 或使用快捷函数
ids = fetch_arxiv_papers("Wed, 17 Dec 2025", max_papers=250)
```

## 定时任务设置

使用 cron 设置每日自动执行：

```bash
# 每天早上 9:00 执行
0 9 * * * cd /path/to/arxiv-ai-agent && uv run python arxiv_daily.py
```

## 日志系统

项目使用 `loguru` 进行日志管理，所有日志会自动输出到控制台并按级别着色：

- `logger.info()` - 一般信息（蓝色）
- `logger.success()` - 成功信息（绿色）
- `logger.warning()` - 警告信息（黄色）
- `logger.error()` - 错误信息（红色）
- `logger.exception()` - 异常信息（红色，包含堆栈）

日志输出示例：
```
2025-12-17 10:30:45 | INFO     | 正在请求 Jina Reader API: https://r.jina.ai/...
2025-12-17 10:30:46 | SUCCESS  | 共提取到 154 篇文献。
2025-12-17 10:30:50 | INFO     | [1/10] 处理: Context-Picker...
2025-12-17 10:31:00 | SUCCESS  | JSON 数据已保存: database/2025-12-17.json
```

## 飞书消息推送

### 消息格式

项目支持两种飞书消息格式：

1. **文本消息** (`call_feishu.py`) - 简单的纯文本通知
2. **交互式卡片** (`call_feishu_card.py`) - 美观的卡片格式，带按钮链接

默认使用**交互式卡片格式**，消息包含：
- 📅 报告日期
- 📚 论文数量统计
- 📝 Markdown 文件链接

## License

MIT
