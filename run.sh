#!/usr/bin/env bash
# ArXiv AI Agent 启动脚本

# 使用 uv 运行 Python 脚本
nohup uv run python arxiv_daily.py "$@" >./logs/arxiv_ai_agent_$(date +%Y%m%d).log 2>&1 &
