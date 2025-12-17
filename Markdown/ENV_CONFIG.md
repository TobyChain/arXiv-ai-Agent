# 环境变量配置说明

## API 配置

### LLM API（DeepSeek）
- `API_KEY`: DeepSeek API 密钥
- `BASE_URL`: API 基础 URL（默认: https://yunwu.ai/v1）
- `MODEL_NAME`: 使用的模型名称（默认: deepseek-chat）

### Jina Reader API
- `JINA_API_KEY`: Jina AI API 密钥

### 飞书机器人
- `FEISHU_WEBHOOK_URL`: 飞书群机器人 Webhook 地址
- `FEISHU_SECRET`: 飞书机器人签名密钥

## 服务器配置

### 监听配置
- `SERVER_HOST`: 服务器监听地址
  - `0.0.0.0` - 监听所有网卡（允许外部访问）
  - `127.0.0.1` - 仅本地访问
  - 具体 IP - 绑定到特定网卡
  - **默认**: `0.0.0.0`

- `SERVER_PORT`: 服务器监听端口
  - **默认**: `8000`
  - 范围: 1024-65535

### 外部访问地址
- `WEB_SERVER_URL`: 外部访问的完整 URL
  - 用于飞书通知等外部服务
  - 示例: 
    - 本地: `http://localhost:8000`
    - 局域网: `http://192.168.1.100:8000`
    - 公网: `https://arxiv.yourdomain.com`

## ArXiv 配置

- `ARXIV_SUBJECT`: 默认抓取的主题代码
  - **默认**: `cs.AI`
  - 可选值参见 arxiv_subjects.py

## 配置示例

### 场景 1: 本地开发
```bash
SERVER_HOST="127.0.0.1"
SERVER_PORT="8000"
WEB_SERVER_URL="http://localhost:8000"
```

### 场景 2: 局域网服务器
```bash
SERVER_HOST="0.0.0.0"
SERVER_PORT="8080"
WEB_SERVER_URL="http://192.168.1.100:8080"
```

### 场景 3: 公网服务器（带 Nginx 反向代理）
```bash
SERVER_HOST="127.0.0.1"  # Nginx 转发到本地
SERVER_PORT="8000"
WEB_SERVER_URL="https://arxiv.yourdomain.com"
```

### 场景 4: Docker 容器
```bash
SERVER_HOST="0.0.0.0"
SERVER_PORT="8000"
WEB_SERVER_URL="http://your-server-ip:8000"
```

## 验证配置

```bash
# 检查环境变量
cat .env

# 测试服务器启动
uv run python server.py

# 测试访问
curl http://localhost:8000/api/dates
```

## 安全建议

1. **生产环境**：不要将 .env 文件提交到 Git
2. **防火墙**：限制 SERVER_PORT 的访问权限
3. **HTTPS**：生产环境使用 Nginx + SSL 证书
4. **密钥管理**：定期轮换 API 密钥
