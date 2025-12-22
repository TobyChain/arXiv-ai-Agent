#!/usr/bin/env bash
# 设置 ArXiv 论文抓取的定时任务

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# cron 环境很“干净”，常见问题是找不到 uv/python 或未加载 .env。
# 这里显式设置 PATH，并优先用绝对路径运行脚本。
# 针对 uv 虚拟环境，先 source activate 再运行脚本。
CRON_LOG="$SCRIPT_DIR/logs/cron.log"
VENV_ACTIVATE="/home/mi/guanbingtao/.venv/bin/activate"
CRON_JOB="0 8 * * * PATH=/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$HOME/.cargo/bin source $VENV_ACTIVATE && cd $SCRIPT_DIR && /usr/bin/env bash $SCRIPT_DIR/run.sh >> $CRON_LOG 2>&1"

echo "📋 将添加以下 cron 任务："
echo "   每天上午 8:00 执行 ArXiv 论文抓取"
echo ""
echo "   $CRON_JOB"
echo ""

# 创建日志目录
mkdir -p "$SCRIPT_DIR/logs"

# 确保脚本可执行
chmod +x "$SCRIPT_DIR/run.sh" 2>/dev/null || true

# 幂等更新：删除旧任务（包含 run.sh 路径的行），再追加新任务
current_crontab=""
if crontab -l >/dev/null 2>&1; then
    current_crontab="$(crontab -l)"
fi

filtered_crontab="$(printf '%s\n' "$current_crontab" | grep -v "$SCRIPT_DIR/run.sh" || true)"

(printf '%s\n' "$filtered_crontab"; echo "$CRON_JOB") | crontab -

echo ""
echo "✅ Cron 任务已添加成功！"
echo ""
echo "📝 验证任务："
echo "   查看所有任务: crontab -l"
echo "   查看日志: tail -f $CRON_LOG"
echo ""
echo "⏰ 下次执行时间: 明天上午 8:00"
echo ""
echo "💡 提示："
echo "   - 如需立即测试，运行: $SCRIPT_DIR/run.sh"
echo "   - 如需删除任务，运行: crontab -e"
