#!/usr/bin/env bash
# ArXiv AI Agent 启动脚本

# 使用 uv 运行 Python 脚本
uv run python arxiv_daily.py "$@"
