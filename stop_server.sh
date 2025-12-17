#!/usr/bin/env bash
# 停止运行在 Screen 中的服务器

SCREEN_NAME="arxiv-server"

if screen -list | grep -q "$SCREEN_NAME"; then
    echo "🛑 正在停止 Screen 会话 '$SCREEN_NAME'..."
    screen -S "$SCREEN_NAME" -X quit
    sleep 1
    echo "✅ 服务器已停止"
else
    echo "⚠️  未找到运行中的 Screen 会话 '$SCREEN_NAME'"
fi
