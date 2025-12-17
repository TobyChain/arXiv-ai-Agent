#!/usr/bin/env bash
# 使用 Screen 启动 ArXiv AI Agent Web 服务器

SCREEN_NAME="arxiv-server"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查 screen 是否已在运行
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "⚠️  Screen 会话 '$SCREEN_NAME' 已在运行"
    echo "使用以下命令查看: screen -r $SCREEN_NAME"
    echo "使用以下命令停止: screen -S $SCREEN_NAME -X quit"
    exit 1
fi

# 创建新的 screen 会话并启动服务器
echo "🚀 正在启动 Web 服务器（Screen 会话: $SCREEN_NAME）..."
cd "$SCRIPT_DIR"
screen -dmS "$SCREEN_NAME" bash -c "uv run python server.py; exec bash"

# 等待服务器启动
sleep 2

# 检查是否启动成功
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "✅ 服务器已在后台启动"
    echo ""
    echo "📝 常用命令："
    echo "  查看服务器输出: screen -r $SCREEN_NAME"
    echo "  离开但保持运行: Ctrl+A, 然后按 D"
    echo "  停止服务器: screen -S $SCREEN_NAME -X quit"
    echo ""
    echo "🌐 访问地址: http://localhost:8000"
else
    echo "❌ 服务器启动失败，请检查日志"
    exit 1
fi
