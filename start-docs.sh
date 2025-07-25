#!/bin/bash
# 启动 MkDocs 文档服务器

echo "🚀 启动 AMinGY 文档服务器..."
echo "📝 文档将在 http://localhost:8000 可访问"
echo "🛑 按 Ctrl+C 停止服务器"
echo ""



# 启动服务器
uv run mkdocs serve --dev-addr=localhost:8000