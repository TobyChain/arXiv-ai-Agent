#!/usr/bin/env bash
# 设置 ArXiv 论文抓取的定时任务

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRON_JOB="0 8 * * * cd $SCRIPT_DIR && $SCRIPT_DIR/run.sh >> $SCRIPT_DIR/logs/cron.log 2>&1"

echo "📋 将添加以下 cron 任务："
echo "   每天上午 8:00 执行 ArXiv 论文抓取"
echo ""
echo "   $CRON_JOB"
echo ""

# 创建日志目录
mkdir -p "$SCRIPT_DIR/logs"

# 检查是否已存在相同的 cron 任务
if crontab -l 2>/dev/null | grep -q "$SCRIPT_DIR/run.sh"; then
    echo "⚠️  检测到已存在的 cron 任务"
    echo ""
    echo "当前的 crontab:"
    crontab -l | grep "$SCRIPT_DIR/run.sh"
    echo ""
    read -p "是否要删除旧任务并重新添加? (y/N): " confirm
    if [[ $confirm != [yY] ]]; then
        echo "❌ 已取消"
        exit 0
    fi
    # 删除旧任务
    crontab -l | grep -v "$SCRIPT_DIR/run.sh" | crontab -
fi

# 添加新的 cron 任务
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo ""
echo "✅ Cron 任务已添加成功！"
echo ""
echo "📝 验证任务："
echo "   查看所有任务: crontab -l"
echo "   查看日志: tail -f $SCRIPT_DIR/logs/cron.log"
echo ""
echo "⏰ 下次执行时间: 明天上午 8:00"
echo ""
echo "💡 提示："
echo "   - 如需立即测试，运行: $SCRIPT_DIR/run.sh"
echo "   - 如需删除任务，运行: crontab -e"
