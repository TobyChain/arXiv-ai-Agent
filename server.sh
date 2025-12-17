#!/usr/bin/env bash
# ArXiv AI Agent Web 服务器启动脚本

# 使用 uv 运行服务器
uv run python server.py "$@"
