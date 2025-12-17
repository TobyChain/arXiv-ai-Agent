# ArXiv AI Agent 部署指南

## 一、服务器长期运行（使用 Screen）

### 1. 启动服务器

```bash
# 赋予执行权限
chmod +x start_server_screen.sh stop_server_screen.sh

# 启动服务器（在 screen 会话中运行）
./start_server_screen.sh
```

服务器将在后台运行，访问地址：http://localhost:8000

### 2. 管理服务器

```bash
# 查看服务器输出
screen -r arxiv-server

# 离开但保持运行（在 screen 会话内）
# 按 Ctrl+A，然后按 D

# 停止服务器
./stop_server_screen.sh

# 查看所有 screen 会话
screen -ls
```

### 3. Screen 常用命令

| 命令 | 说明 |
|------|------|
| `screen -r arxiv-server` | 重新连接到会话 |
| `Ctrl+A, D` | 离开但保持运行 |
| `Ctrl+A, K` | 杀死当前窗口 |
| `screen -ls` | 列出所有会话 |
| `screen -S arxiv-server -X quit` | 远程关闭会话 |

---

## 二、定时任务配置（每天 8:00 执行）

### 1. 自动设置 Cron 任务

```bash
# 赋予执行权限
chmod +x setup_cron.sh

# 运行设置脚本
./setup_cron.sh
```

这将自动添加以下 cron 任务：
```
0 8 * * * cd /path/to/arxiv-ai-agent && ./run.sh >> logs/cron.log 2>&1
```

### 2. 手动配置 Cron 任务

如果你更喜欢手动配置：

```bash
# 编辑 crontab
crontab -e

# 添加以下行（替换为实际路径）
0 8 * * * cd /home/mi/guanbingtao/arxiv-ai-agent && /home/mi/guanbingtao/arxiv-ai-agent/run.sh >> /home/mi/guanbingtao/arxiv-ai-agent/logs/cron.log 2>&1
```

### 3. Cron 时间格式说明

```
# ┌───────────── 分钟 (0 - 59)
# │ ┌───────────── 小时 (0 - 23)
# │ │ ┌───────────── 日期 (1 - 31)
# │ │ │ ┌───────────── 月份 (1 - 12)
# │ │ │ │ ┌───────────── 星期 (0 - 7, 0 和 7 都是周日)
# │ │ │ │ │
# * * * * * 要执行的命令
```

示例：
- `0 8 * * *` - 每天 8:00
- `0 8 * * 1` - 每周一 8:00
- `0 8 1 * *` - 每月 1 号 8:00
- `0 8,20 * * *` - 每天 8:00 和 20:00
- `*/30 * * * *` - 每 30 分钟

### 4. 验证和管理 Cron 任务

```bash
# 查看当前用户的所有 cron 任务
crontab -l

# 查看 cron 日志
tail -f logs/cron.log

# 删除所有 cron 任务（谨慎使用！）
crontab -r

# 删除特定任务
crontab -e  # 手动删除对应行
```

---

## 三、完整部署流程

### 步骤 1: 启动 Web 服务器

```bash
# 启动服务器（使用 screen 后台运行）
./start_server_screen.sh
```

### 步骤 2: 设置定时任务

```bash
# 配置每天 8:00 自动抓取论文
./setup_cron.sh
```

### 步骤 3: 测试运行

```bash
# 手动运行一次，验证功能正常
./run.sh

# 查看输出日志
tail -f logs/cron.log
```

### 步骤 4: 验证部署

```bash
# 检查 screen 会话
screen -ls

# 检查 cron 任务
crontab -l

# 访问 Web 界面
# 浏览器打开: http://localhost:8000
```

---

## 四、日志管理

### 日志文件位置

- Cron 执行日志: `logs/cron.log`
- Loguru 日志: 控制台输出（被重定向到 cron.log）

### 查看日志

```bash
# 实时查看 cron 日志
tail -f logs/cron.log

# 查看最近 100 行
tail -n 100 logs/cron.log

# 搜索错误信息
grep -i error logs/cron.log
```

### 日志轮转（可选）

如果日志文件增长过快，可以配置 logrotate：

```bash
# 创建 logrotate 配置
sudo tee /etc/logrotate.d/arxiv-agent << EOF
/home/mi/guanbingtao/arxiv-ai-agent/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOF
```

---

## 五、故障排查

### 问题 1: Cron 任务未执行

```bash
# 检查 cron 服务状态
sudo systemctl status cron

# 查看系统 cron 日志
sudo tail -f /var/log/syslog | grep CRON

# 确认路径正确
which uv
echo $PATH
```

### 问题 2: Screen 会话丢失

```bash
# 查看所有 screen 会话
screen -ls

# 如果会话丢失，重新启动
./start_server_screen.sh
```

### 问题 3: 环境变量未加载

确保 `.env` 文件存在且包含所有必要配置：

```bash
# 检查 .env 文件
cat .env

# 必需的环境变量:
# - JINA_API_KEY
# - API_KEY (LLM)
# - BASE_URL (LLM)
# - MODEL_NAME
# - FEISHU_WEBHOOK_URL
# - FEISHU_SECRET
# - WEB_SERVER_URL
```

---

## 六、系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Python**: 3.11+
- **依赖工具**:
  - `screen` - 终端复用器
  - `cron` - 定时任务调度
  - `uv` - Python 包管理器

### 安装依赖

```bash
# 安装 screen
sudo apt install screen -y

# 安装 uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 验证安装
screen -version
uv --version
```

---

## 七、生产环境建议

1. **使用 systemd 替代 screen**（更稳定）
2. **配置日志轮转**避免磁盘占满
3. **监控服务状态**（可使用 healthcheck 脚本）
4. **备份数据库目录**定期备份 JSON 数据
5. **使用 Nginx 反向代理**提供 HTTPS 支持
6. **设置防火墙规则**限制端口访问

---

## 八、快速命令参考

```bash
# 服务器管理
./start_server_screen.sh          # 启动服务器
./stop_server_screen.sh           # 停止服务器
screen -r arxiv-server            # 查看服务器输出

# 定时任务
./setup_cron.sh                   # 设置定时任务
crontab -l                        # 查看定时任务
tail -f logs/cron.log             # 查看执行日志

# 手动运行
./run.sh                          # 手动抓取论文
./run_server.sh                   # 前台运行服务器（调试用）

# 测试
curl http://localhost:8000/api/dates     # 测试 API
curl http://localhost:8000               # 测试前端
```
