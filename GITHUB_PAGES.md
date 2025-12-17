# GitHub Pages 配置指南

本指南将帮助你使用 GitHub Pages 托管 ArXiv AI Agent 的前端界面。

## 🎯 GitHub Pages 部署方案

GitHub Pages 是静态网站托管服务，我们将使用它托管 Web 前端，而后端 API 仍需部署在自己的服务器上。

## 📋 部署步骤

### 方式 1: 使用 `gh-pages` 分支（推荐）

#### 步骤 1: 在 GitHub 仓库中启用 Pages

1. 访问你的仓库：https://github.com/TobyChain/arXiv-ai-Agent
2. 点击 **Settings** → **Pages**
3. 在 **Source** 中选择：
   - Branch: `gh-pages`
   - Folder: `/ (root)`
4. 点击 **Save**

#### 步骤 2: 创建 `gh-pages` 分支并推送前端文件

```bash
cd /home/mi/guanbingtao/arxiv-ai-agent

# 创建独立的 gh-pages 分支
git checkout --orphan gh-pages

# 删除所有文件（保留 web 目录）
git rm -rf .
git clean -fd

# 仅保留前端文件
git checkout main -- web/
git checkout main -- database/index.json
mv web/* .
rm -rf web/

# 创建 CNAME 文件（如果有自定义域名）
# echo "arxiv.yourdomain.com" > CNAME

# 修改 index.html 中的 API 地址为你的服务器
# 稍后会用脚本自动处理
```

#### 步骤 3: 配置 API 地址

创建一个配置文件来指定后端 API 地址：

```bash
cat > config.js << 'EOF'
// GitHub Pages 配置
const API_BASE_URL = 'https://your-server.com';  // 替换为你的服务器地址

// 在应用启动前设置全局 API 地址
window.API_BASE_URL = API_BASE_URL;
EOF
```

修改 `index.html`，在加载 `app.js` 之前添加：

```html
<script src="config.js"></script>
```

修改 `app.js` 中的 API 调用，使用配置的地址：

```javascript
// 替换所有 API 调用的基础路径
const API_BASE = window.API_BASE_URL || '';

// 例如：
axios.get(`${API_BASE}/api/dates`)
```

#### 步骤 4: 提交并推送到 GitHub Pages

```bash
git add .
git commit -m "Deploy to GitHub Pages"
git push -f origin gh-pages

# 切回 main 分支继续开发
git checkout main
```

### 方式 2: 使用 `docs/` 目录

#### 步骤 1: 在 main 分支创建 docs 目录

```bash
cd /home/mi/guanbingtao/arxiv-ai-agent

# 创建 docs 目录并复制前端文件
mkdir -p docs
cp -r web/* docs/
cp database/index.json docs/

# 创建配置文件
cat > docs/config.js << 'EOF'
const API_BASE_URL = 'https://your-server.com';
window.API_BASE_URL = API_BASE_URL;
EOF

# 提交
git add docs/
git commit -m "Add GitHub Pages docs folder"
git push origin main
```

#### 步骤 2: 在 GitHub 仓库设置中

1. Settings → Pages
2. Source: `main` branch
3. Folder: `/docs`
4. Save

### 方式 3: 使用 GitHub Actions 自动部署

创建 `.github/workflows/deploy.yml`：

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Prepare deployment
        run: |
          mkdir -p _site
          cp -r web/* _site/
          cp database/index.json _site/
          
          # 创建配置文件
          echo "const API_BASE_URL = 'https://your-server.com';" > _site/config.js
          echo "window.API_BASE_URL = API_BASE_URL;" >> _site/config.js
          
          # 修改 index.html 添加 config.js 引用
          sed -i 's/<script src="https:\/\/cdn.jsdelivr.net\/npm\/vue@2.6.14\/dist\/vue.js"><\/script>/<script src="config.js"><\/script>\n    <script src="https:\/\/cdn.jsdelivr.net\/npm\/vue@2.6.14\/dist\/vue.js"><\/script>/' _site/index.html

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

提交并推送：

```bash
git add .github/workflows/deploy.yml
git commit -m "Add GitHub Actions deployment"
git push origin main
```

## 🔧 跨域问题解决

由于前端托管在 GitHub Pages，后端在你的服务器，需要配置 CORS。

在 `server.py` 中（已配置）：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🌐 自定义域名（可选）

### 步骤 1: 创建 CNAME 文件

在 `gh-pages` 分支或 `docs/` 目录中创建 `CNAME` 文件：

```bash
echo "arxiv.yourdomain.com" > CNAME
```

### 步骤 2: 配置 DNS

在域名提供商处添加 DNS 记录：

```
类型: CNAME
名称: arxiv
值: tobychain.github.io
```

### 步骤 3: 在 GitHub 仓库设置中

Settings → Pages → Custom domain → 输入 `arxiv.yourdomain.com` → Save

## 📊 部署架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户浏览器                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ├─────────────────┬──────────────────┐
                          │                 │                  │
                    静态资源请求        API 请求         WebSocket
                          │                 │                  │
                          ▼                 ▼                  ▼
┌──────────────────────────────┐  ┌────────────────────────────┐
│      GitHub Pages            │  │   Your Server              │
│  https://user.github.io      │  │   https://api.your.com     │
│                              │  │                            │
│  - index.html                │  │   - FastAPI Backend        │
│  - app.js                    │  │   - Jina/LLM/Feishu       │
│  - config.js                 │  │   - Database              │
│  - database/index.json       │  │   - Nginx (optional)      │
└──────────────────────────────┘  └────────────────────────────┘
```

## 🚀 完整部署脚本

创建自动化部署脚本 `deploy_gh_pages.sh`：

```bash
#!/bin/bash
# GitHub Pages 自动部署脚本

API_SERVER="https://your-server.com"  # 修改为你的服务器地址

echo "📦 准备部署到 GitHub Pages..."

# 保存当前分支
CURRENT_BRANCH=$(git branch --show-current)

# 创建临时目录
TEMP_DIR=$(mktemp -d)
echo "📁 临时目录: $TEMP_DIR"

# 复制前端文件
cp -r web/* "$TEMP_DIR/"
cp database/index.json "$TEMP_DIR/" 2>/dev/null || echo "[]" > "$TEMP_DIR/index.json"

# 创建配置文件
cat > "$TEMP_DIR/config.js" << EOF
const API_BASE_URL = '$API_SERVER';
window.API_BASE_URL = API_BASE_URL;
console.log('API Base URL:', API_BASE_URL);
EOF

# 修改 index.html 添加 config.js
sed -i 's/<script src="https:\/\/cdn.jsdelivr.net\/npm\/vue@2.6.14\/dist\/vue.js">/<script src="config.js"><\/script>\n    <script src="https:\/\/cdn.jsdelivr.net\/npm\/vue@2.6.14\/dist\/vue.js">/' "$TEMP_DIR/index.html"

# 修改 app.js 使用 API_BASE_URL
sed -i "s|'/api/|window.API_BASE_URL + '/api/|g" "$TEMP_DIR/app.js"
sed -i "s|'/database/|window.API_BASE_URL + '/database/|g" "$TEMP_DIR/app.js"

# 切换到 gh-pages 分支
git checkout --orphan gh-pages-new 2>/dev/null || git checkout gh-pages

# 清空当前内容
git rm -rf . 2>/dev/null
rm -rf *

# 复制新内容
cp -r "$TEMP_DIR/"* .

# 创建 README
cat > README.md << 'EOF'
# ArXiv AI Agent - GitHub Pages

这是 ArXiv AI Agent 的前端页面，托管在 GitHub Pages 上。

后端 API 部署在独立服务器。

访问：https://tobychain.github.io/arXiv-ai-Agent/
EOF

# 提交并推送
git add .
git commit -m "Deploy: $(date '+%Y-%m-%d %H:%M:%S')"
git branch -D gh-pages 2>/dev/null
git branch -m gh-pages
git push -f origin gh-pages

# 切回原分支
git checkout "$CURRENT_BRANCH"

# 清理
rm -rf "$TEMP_DIR"

echo "✅ 部署完成！"
echo "🌐 访问: https://tobychain.github.io/arXiv-ai-Agent/"
echo ""
echo "⚠️  请确保后端服务器已启动: $API_SERVER"
```

使用脚本：

```bash
chmod +x deploy_gh_pages.sh
./deploy_gh_pages.sh
```

## ✅ 验证部署

1. 访问 GitHub Pages：https://tobychain.github.io/arXiv-ai-Agent/
2. 打开浏览器开发者工具 (F12) → Console
3. 检查是否有 CORS 错误
4. 确认 API 请求发送到正确的服务器

## 🔍 故障排查

### 问题 1: 404 Not Found
**原因**: GitHub Pages 未正确配置或文件未上传
**解决**: 检查 Settings → Pages 设置，确认分支和目录正确

### 问题 2: API 请求失败
**原因**: CORS 未配置或后端服务未启动
**解决**: 
- 检查 server.py 的 CORS 配置
- 确认后端服务运行中
- 检查 config.js 中的 API_BASE_URL

### 问题 3: 样式或脚本加载失败
**原因**: 相对路径问题
**解决**: 使用 CDN 或绝对路径

## 📝 最佳实践

1. **分离前后端**: GitHub Pages 仅托管静态文件，API 在独立服务器
2. **使用 CDN**: 所有库文件使用 CDN 链接
3. **配置文件管理**: 使用 config.js 管理环境配置
4. **自动化部署**: 使用 GitHub Actions 实现 CI/CD
5. **版本控制**: gh-pages 分支仅包含构建产物

---

完成配置后，你的前端将托管在 GitHub Pages，免费、稳定、快速！
