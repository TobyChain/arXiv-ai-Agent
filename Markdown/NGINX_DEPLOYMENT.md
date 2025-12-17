# Nginx 公网部署指南

本指南将帮助你使用 Nginx 作为反向代理，将 ArXiv AI Agent 部署到公网服务器。

## 📋 前置要求

- ✅ 一台公网服务器（云服务器、VPS 等）
- ✅ 域名（可选，推荐用于 HTTPS）
- ✅ 域名已解析到服务器 IP
- ✅ 服务器已开放 80 和 443 端口

## 🚀 部署步骤

### 步骤 1: 安装 Nginx

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install nginx -y
```

#### CentOS/RHEL
```bash
sudo yum install epel-release -y
sudo yum install nginx -y
```

#### 验证安装
```bash
nginx -v
sudo systemctl status nginx
```

### 步骤 2: 配置后端服务

修改 `.env` 文件，让服务仅监听本地：

```bash
# 服务器配置
SERVER_HOST="127.0.0.1"  # 仅本地访问，由 Nginx 转发
SERVER_PORT="8000"
WEB_SERVER_URL="https://arxiv.yourdomain.com"  # 替换为你的域名
```

### 步骤 3: 配置 Nginx 反向代理

#### 方案 A: HTTP 配置（仅用于测试）

创建配置文件：
```bash
sudo nano /etc/nginx/sites-available/arxiv-agent
```

添加以下内容：
```nginx
server {
    listen 80;
    server_name arxiv.yourdomain.com;  # 替换为你的域名

    # 访问日志
    access_log /var/log/nginx/arxiv_access.log;
    error_log /var/log/nginx/arxiv_error.log;

    # 客户端请求大小限制
    client_max_body_size 10M;

    # 代理到 FastAPI 后端
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        
        # WebSocket 支持
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        
        # 传递真实客户端信息
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;  # 处理长时间请求（如手动抓取）
    }
}
```

启用配置：
```bash
sudo ln -s /etc/nginx/sites-available/arxiv-agent /etc/nginx/sites-enabled/
sudo nginx -t  # 测试配置
sudo systemctl reload nginx
```

#### 方案 B: HTTPS 配置（推荐，生产环境）

##### B1. 安装 Certbot（Let's Encrypt）

```bash
# Ubuntu/Debian
sudo apt install certbot python3-certbot-nginx -y

# CentOS/RHEL
sudo yum install certbot python3-certbot-nginx -y
```

##### B2. 获取 SSL 证书

```bash
sudo certbot --nginx -d arxiv.yourdomain.com
```

按提示操作：
1. 输入邮箱
2. 同意服务条款
3. 选择是否重定向 HTTP 到 HTTPS（建议选择 Yes）

##### B3. 完整 HTTPS 配置

创建配置文件：
```bash
sudo nano /etc/nginx/sites-available/arxiv-agent
```

添加以下内容：
```nginx
# HTTP 自动重定向到 HTTPS
server {
    listen 80;
    server_name arxiv.yourdomain.com;
    
    # 强制跳转 HTTPS
    return 301 https://$server_name$request_uri;
}

# HTTPS 主配置
server {
    listen 443 ssl http2;
    server_name arxiv.yourdomain.com;

    # SSL 证书配置（Certbot 会自动添加）
    ssl_certificate /etc/letsencrypt/live/arxiv.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/arxiv.yourdomain.com/privkey.pem;
    
    # SSL 优化配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # 安全头
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # 日志
    access_log /var/log/nginx/arxiv_access.log;
    error_log /var/log/nginx/arxiv_error.log;

    # 客户端请求大小限制
    client_max_body_size 10M;

    # Gzip 压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss;

    # 反向代理配置
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        
        # WebSocket 支持
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_cache_bypass $http_upgrade;
        
        # 传递客户端信息
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时设置（适配长时间任务）
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 600s;  # 10 分钟，用于长时间抓取任务
        
        # 缓冲设置
        proxy_buffering off;
    }

    # 静态文件缓存优化（可选）
    location ~* \.(jpg|jpeg|png|gif|ico|css|js|svg|woff|woff2|ttf|eot)$ {
        proxy_pass http://127.0.0.1:8000;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

启用配置：
```bash
sudo ln -s /etc/nginx/sites-available/arxiv-agent /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 步骤 4: 启动后端服务

使用 Screen 在后台运行：
```bash
cd /home/mi/guanbingtao/arxiv-ai-agent
./start_server_screen.sh
```

### 步骤 5: 配置防火墙

#### UFW (Ubuntu)
```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw status
```

#### Firewalld (CentOS)
```bash
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### 步骤 6: 设置 SSL 证书自动续期

Certbot 会自动添加 cron 任务，验证自动续期：
```bash
sudo certbot renew --dry-run
```

查看自动续期定时任务：
```bash
sudo systemctl status certbot.timer
```

## 🔒 安全加固

### 1. 限制访问速率（防 DDoS）

在 Nginx 配置中添加：
```nginx
# 在 http 块中添加
http {
    # 限制请求速率
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=fetch_limit:10m rate=2r/m;

    server {
        # ... 其他配置 ...
        
        # 一般 API 限速
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            proxy_pass http://127.0.0.1:8000;
            # ... 其他配置 ...
        }
        
        # 抓取任务严格限速
        location /api/fetch {
            limit_req zone=fetch_limit burst=5 nodelay;
            proxy_pass http://127.0.0.1:8000;
            # ... 其他配置 ...
        }
    }
}
```

### 2. IP 白名单（可选）

仅允许特定 IP 访问管理接口：
```nginx
location /api/fetch {
    allow 192.168.1.0/24;  # 允许局域网
    allow 123.456.789.0;   # 允许特定 IP
    deny all;              # 拒绝其他所有
    
    proxy_pass http://127.0.0.1:8000;
}
```

### 3. 隐藏 Nginx 版本号

编辑主配置：
```bash
sudo nano /etc/nginx/nginx.conf
```

在 `http` 块中添加：
```nginx
http {
    server_tokens off;
    # ... 其他配置 ...
}
```

## 🔍 故障排查

### 查看 Nginx 日志
```bash
# 访问日志
sudo tail -f /var/log/nginx/arxiv_access.log

# 错误日志
sudo tail -f /var/log/nginx/arxiv_error.log
```

### 检查后端服务
```bash
# 查看服务是否运行
screen -ls

# 重新连接到服务
screen -r arxiv-server

# 测试后端直接访问
curl http://127.0.0.1:8000/api/dates
```

### 测试 Nginx 配置
```bash
sudo nginx -t
```

### 重启服务
```bash
# 重启 Nginx
sudo systemctl restart nginx

# 重启后端
./stop_server_screen.sh
./start_server_screen.sh
```

### 检查端口监听
```bash
sudo netstat -tlnp | grep -E '(80|443|8000)'
# 或
sudo ss -tlnp | grep -E '(80|443|8000)'
```

## 🎯 完整部署检查清单

- [ ] 域名已解析到服务器 IP
- [ ] 防火墙已开放 80 和 443 端口
- [ ] Nginx 已安装并运行
- [ ] SSL 证书已配置（Let's Encrypt）
- [ ] `.env` 配置正确（SERVER_HOST=127.0.0.1）
- [ ] 后端服务在 Screen 中运行
- [ ] Nginx 反向代理配置正确
- [ ] 可以通过域名访问网站
- [ ] HTTPS 自动重定向正常
- [ ] 日志记录正常
- [ ] SSL 证书自动续期已设置

## 📊 性能优化建议

### 1. 启用 HTTP/2
已在配置中包含 `http2` 参数

### 2. 配置缓存
```nginx
# 在 http 块中添加
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m max_size=1g inactive=60m;

# 在 location 中使用
location /api/dates {
    proxy_cache api_cache;
    proxy_cache_valid 200 5m;  # 缓存 5 分钟
    proxy_pass http://127.0.0.1:8000;
}
```

### 3. 连接池优化
```nginx
upstream backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

location / {
    proxy_pass http://backend;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
}
```

## 🌐 访问测试

配置完成后，访问以下地址测试：

```bash
# 测试 HTTPS
curl -I https://arxiv.yourdomain.com

# 测试 API
curl https://arxiv.yourdomain.com/api/dates

# 浏览器访问
https://arxiv.yourdomain.com
```

## 🔄 更新 .env 配置

最终的 `.env` 配置应该是：
```bash
SERVER_HOST="127.0.0.1"  # 仅本地，由 Nginx 转发
SERVER_PORT="8000"
WEB_SERVER_URL="https://arxiv.yourdomain.com"  # 你的域名
```

## 📞 常见问题

### Q: 502 Bad Gateway
**A:** 检查后端服务是否运行：`screen -ls`

### Q: 证书错误
**A:** 确认域名解析正确，重新运行 `certbot --nginx`

### Q: 访问超时
**A:** 检查防火墙设置，确保 80/443 端口开放

### Q: 抓取任务超时
**A:** 增加 `proxy_read_timeout` 值到 600s 或更高

---

完成部署后，你的服务将通过 HTTPS 安全访问，享受 Nginx 的高性能反向代理和负载均衡能力！
