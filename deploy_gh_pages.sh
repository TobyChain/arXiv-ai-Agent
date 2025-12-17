#!/bin/bash
# GitHub Pages 自动部署脚本

# ==================== 配置区 ====================
API_SERVER="${1:-http://localhost:8000}"  # 从参数获取或使用默认值
# ==============================================

echo "🚀 GitHub Pages 部署工具"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📌 API 服务器: $API_SERVER"
echo ""

# 保存当前分支
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 当前分支: $CURRENT_BRANCH"

# 创建临时目录
TEMP_DIR=$(mktemp -d)
echo "📁 临时目录: $TEMP_DIR"

# 复制前端文件
echo "📦 复制前端文件..."
cp -r web/* "$TEMP_DIR/"
cp database/index.json "$TEMP_DIR/" 2>/dev/null || echo "[]" > "$TEMP_DIR/index.json"

# 创建配置文件
echo "⚙️  创建配置文件..."
cat > "$TEMP_DIR/config.js" << EOF
// GitHub Pages API 配置
const API_BASE_URL = '$API_SERVER';
window.API_BASE_URL = API_BASE_URL;
console.log('🔗 API Base URL:', API_BASE_URL);
EOF

# 修改 index.html 添加 config.js 引用
echo "🔧 修改 index.html..."
sed -i '/<script src="https:\/\/cdn.jsdelivr.net\/npm\/vue@2.6.14\/dist\/vue.js"><\/script>/i\    <script src="config.js"><\/script>' "$TEMP_DIR/index.html"

# 修改 app.js 使用配置的 API 地址
echo "🔧 修改 app.js..."
sed -i "s|axios.get('/api/|axios.get((window.API_BASE_URL || '') + '/api/|g" "$TEMP_DIR/app.js"
sed -i "s|axios.get(\`/api/|axios.get(\`\${window.API_BASE_URL || ''}/api/|g" "$TEMP_DIR/app.js"
sed -i "s|axios.post('/api/|axios.post((window.API_BASE_URL || '') + '/api/|g" "$TEMP_DIR/app.js"
sed -i "s|axios.get('/database/|axios.get((window.API_BASE_URL || '') + '/database/|g" "$TEMP_DIR/app.js"
sed -i "s|axios.get(\`/database/|axios.get(\`\${window.API_BASE_URL || ''}/database/|g" "$TEMP_DIR/app.js"

# 切换到 gh-pages 分支
echo "🔀 切换到 gh-pages 分支..."
if git show-ref --verify --quiet refs/heads/gh-pages; then
    git checkout gh-pages
    git rm -rf . 2>/dev/null
else
    git checkout --orphan gh-pages
    git rm -rf . 2>/dev/null
fi

# 复制新内容
echo "📋 复制部署文件..."
cp -r "$TEMP_DIR/"* .

# 创建 README
cat > README.md << 'MDEOF'
# ArXiv AI Agent - GitHub Pages

这是 ArXiv AI Agent 的前端页面，托管在 GitHub Pages 上。

## 🌐 访问地址

https://tobychain.github.io/arXiv-ai-Agent/

## 🔧 技术栈

- Vue.js 2.6
- Tailwind CSS
- Axios
- Font Awesome

## 📡 后端 API

后端服务部署在独立服务器，通过 CORS 跨域访问。

## 📝 最后更新

MDEOF

echo "构建时间: $(date '+%Y-%m-%d %H:%M:%S')" >> README.md
echo "API 服务器: $API_SERVER" >> README.md

# 创建 .nojekyll 文件（禁用 Jekyll 处理）
touch .nojekyll

# 提交并推送
echo "💾 提交更改..."
git add .
git commit -m "Deploy: $(date '+%Y-%m-%d %H:%M:%S') | API: $API_SERVER"

echo "⬆️  推送到 GitHub..."
git push -f origin gh-pages

# 切回原分支
echo "🔙 切回 $CURRENT_BRANCH 分支..."
git checkout "$CURRENT_BRANCH"

# 清理临时目录
rm -rf "$TEMP_DIR"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 部署完成！"
echo ""
echo "🌐 访问地址:"
echo "   https://tobychain.github.io/arXiv-ai-Agent/"
echo ""
echo "⚠️  请确保:"
echo "   1. GitHub Pages 已在仓库设置中启用"
echo "   2. 后端服务已启动: $API_SERVER"
echo "   3. CORS 已正确配置"
echo ""
echo "📖 查看部署指南: GITHUB_PAGES.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
