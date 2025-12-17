# ArXiv AI Agent

一个基于 AI 的 ArXiv 论文每日速递系统，支持自动抓取、翻译和推送。

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-success)](https://tobychain.github.io/arXiv-ai-Agent/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Vue.js](https://img.shields.io/badge/Vue.js-2.6-brightgreen.svg)](https://v2.vuejs.org/)

## ✨ 功能特性

- 🤖 **自动抓取**：每日自动获取 ArXiv 最新论文
- 🌐 **智能翻译**：使用 DeepSeek 模型翻译摘要为中文
- 📱 **飞书推送**：支持飞书群机器人消息通知
- 🔍 **全文搜索**：支持 ArXiv 全库关键词搜索
- 🎯 **主题定制**：支持 40+ CS 主题自定义抓取
- 💻 **Web 界面**：Vue.js 构建的现代化前端
- 📊 **实时进度**：手动抓取时显示实时进度条
- 🚀 **一键部署**：支持 Screen、Cron、Nginx 等部署方案

## 🌟 在线体验

访问 GitHub Pages 查看演示：[https://tobychain.github.io/arXiv-ai-Agent/](https://tobychain.github.io/arXiv-ai-Agent/)

## 项目结构

```
arXiv-ai-Agent/
├── arxiv_daily.py         # 主程序（自动化抓取）
├── server.py              # FastAPI 后端
├── call_llm.py           # LLM 翻译模块
├── call_jina.py          # Jina API 客户端
├── call_feishu.py        # 飞书通知（基础）
├── call_feishu_card.py   # 飞书卡片通知
├── arxiv_subjects.py     # 主题配置
├── web/                  # 前端文件
│   ├── index.html
│   └── app.js
├── database/             # 数据存储
├── .env                  # 环境变量（不提交）
└── README.md
```
🔍 **全库搜索**：支持实时搜索 ArXiv 全库论文
📱 **响应式界面**：现代化 Web 界面，支持移动端访问

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

# Web 服务器地址（用于生成可访问的链接）
# 本地测试使用 localhost，部署后改为公网地址
WEB_SERVER_URL="http://localhost:8000"
# 或使用公网地址：WEB_SERVER_URL="https://arxiv.ai.agent.com"
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
uv run python arxiv_daily_job.py

# 或使用提供的快捷脚本
./run.sh
```

### 4. 启动 Web 服务

```bash
# 使用 uv 启动服务器
uv run python server.py

# 或使用提供的快捷脚本
./run_server.sh

# 访问 http://localhost:8000
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

## API 接口

### GET /api/dates
获取所有可用日期列表

### GET /api/report?date=YYYY-MM-DD
获取指定日期的论文报告

### GET /api/search?query=关键词
搜索 ArXiv 论文库

## 定时任务设置

使用 cron 设置每日自动执行：

```bash
# 每天早上 9:00 执行
0 9 * * * cd /path/to/arxiv-ai-agent && uv run python arxiv_daily_job.py
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
- 📄 **HTML 报告按钮** - 直接打开静态 HTML 页面
- 🌐 **Web 界面按钮** - 打开交互式 Web 应用

### 配置可访问的 URL

为确保飞书消息中的链接可以直接打开，需要配置 `WEB_SERVER_URL`：

**本地测试**：
```env
WEB_SERVER_URL="http://localhost:8000"
```

**内网部署**：
```env
WEB_SERVER_URL="http://192.168.1.100:8000"
```

**公网部署**（推荐）：
```env
WEB_SERVER_URL="https://arxiv.ai.agent.com"
```

### 使用内网穿透（可选）

如果需要在没有公网 IP 的情况下让飞书消息可访问，可使用内网穿透工具：

**使用 ngrok**：
```bash
# 安装 ngrok
brew install ngrok  # macOS
# 或从 https://ngrok.com 下载

# 启动内网穿透
ngrok http 8000

# 将生成的 URL 配置到 .env 中
# WEB_SERVER_URL="https://xxxx-xx-xx-xx-xx.ngrok-free.app"
```

**使用 frp** 或其他内网穿透工具同理。

## License

MIT
