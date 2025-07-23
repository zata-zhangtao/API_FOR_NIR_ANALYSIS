# NIR API 文档系统

本文档系统使用 MkDocs 框架为 NIR API 创建了完整的 API 文档。

## 🎯 功能特性

- **完整的 API 参考**: 涵盖所有模块和函数
- **详细的教程**: 从基础使用到高级功能
- **实际示例**: 包含完整的工作流程示例
- **中文支持**: 完全中文化的文档界面
- **响应式设计**: 支持桌面和移动设备
- **搜索功能**: 支持中英文搜索
- **代码高亮**: 语法高亮和代码复制功能

## 📁 文档结构

```
nirapi/
├── mkdocs.yml              # MkDocs 配置文件
├── docs/                   # 文档源文件
│   ├── index.md           # 首页
│   ├── getting-started.md # 快速开始
│   ├── changelog.md       # 更新日志
│   ├── api/               # API 参考文档
│   │   ├── overview.md    # API 概览
│   │   ├── load_data.md   # 数据加载模块
│   │   ├── preprocessing.md # 预处理模块
│   │   ├── ml_model.md    # 机器学习模块
│   │   ├── draw.md        # 可视化模块
│   │   ├── analysis.md    # 分析模块
│   │   ├── utils.md       # 工具模块
│   │   └── analysis_class.md # 分析类模块
│   ├── tutorials/         # 教程
│   │   └── basic-usage.md # 基础使用教程
│   └── examples/          # 示例
│       └── complete-workflow.md # 完整工作流程
├── scripts/               # 构建脚本
│   └── build_docs.py     # 文档构建工具
└── site/                 # 构建输出目录
```

## 🚀 快速开始

### 1. 查看文档

文档已经构建完成，您可以通过以下方式查看：

#### 本地查看
```bash
cd nirapi
mkdocs serve
```
然后在浏览器中访问 `http://127.0.0.1:8000`

#### 静态文件
构建的静态文件位于 `site/` 目录，可以直接部署到任何 Web 服务器。

### 2. 使用构建脚本

我们提供了一个便捷的构建脚本：

```bash
# 验证文档
python scripts/build_docs.py validate

# 构建文档
python scripts/build_docs.py build

# 启动开发服务器
python scripts/build_docs.py serve

# 清理构建文件
python scripts/build_docs.py clean
```

## 📖 文档内容

### API 参考文档

- **[数据加载模块](docs/api/load_data.md)**: 从数据库和文件加载光谱数据
- **[预处理模块](docs/api/preprocessing.md)**: 光谱数据预处理方法
- **[机器学习模块](docs/api/ml_model.md)**: 回归、分类和特征选择算法
- **[可视化模块](docs/api/draw.md)**: 数据可视化和图表绘制
- **[分析模块](docs/api/analysis.md)**: 光谱数据分析工具
- **[工具模块](docs/api/utils.md)**: 自动机器学习和实用工具
- **[分析类模块](docs/api/analysis_class.md)**: 高级分析功能和光谱重建

### 教程和示例

- **[基础使用教程](docs/tutorials/basic-usage.md)**: 从数据加载到模型训练的完整流程
- **[完整工作流程示例](docs/examples/complete-workflow.md)**: 血糖检测项目的端到端实现

### 其他文档

- **[快速开始指南](docs/getting-started.md)**: 安装和基本使用
- **[更新日志](docs/changelog.md)**: 版本更新记录

## 🛠️ 开发和维护

### 添加新文档

1. 在 `docs/` 目录下创建新的 `.md` 文件
2. 在 `mkdocs.yml` 的 `nav` 部分添加链接
3. 使用 `mkdocs serve` 预览效果

### 更新现有文档

1. 直接编辑对应的 `.md` 文件
2. MkDocs 开发服务器会自动重新加载
3. 检查链接和格式是否正确

### 文档规范

- 使用标准 Markdown 语法
- 代码示例要完整可运行
- 提供中文注释和说明
- 包含参数说明和返回值
- 添加实际使用场景

## 🎨 主题和样式

文档使用 Material for MkDocs 主题，具有以下特性：

- **深色/浅色模式**: 自动切换或手动选择
- **中文字体**: 优化的中文显示效果
- **代码高亮**: Python 语法高亮
- **搜索功能**: 支持中英文全文搜索
- **导航栏**: 清晰的层级结构
- **响应式**: 适配各种屏幕尺寸

## 📦 部署选项

### 1. GitHub Pages

```bash
# 自动部署到 GitHub Pages
python scripts/build_docs.py deploy
```

### 2. 自定义服务器

```bash
# 构建静态文件
mkdocs build

# 将 site/ 目录内容部署到服务器
rsync -av site/ user@server:/var/www/html/
```

### 3. Docker 部署

```dockerfile
FROM nginx:alpine
COPY site/ /usr/share/nginx/html/
EXPOSE 80
```

## 🔧 配置说明

### mkdocs.yml 主要配置

- `site_name`: 网站标题
- `theme`: Material 主题配置
- `nav`: 导航结构
- `plugins`: 搜索等插件
- `markdown_extensions`: Markdown 扩展功能

### 自定义配置

可以根据需要修改：

- 主题颜色和字体
- 导航结构
- 搜索语言
- 代码高亮样式

## 📊 文档统计

当前文档包含：

- **8个 API 模块文档**: 详细的函数和类说明
- **1个基础教程**: 完整的学习路径
- **1个完整示例**: 实际项目应用
- **100+ 代码示例**: 可运行的示例代码
- **中英文搜索**: 支持全文搜索

## 🤝 贡献指南

欢迎贡献文档内容：

1. Fork 项目仓库
2. 创建新的文档分支
3. 添加或修改文档内容
4. 提交 Pull Request

### 文档贡献规范

- 保持文档结构清晰
- 提供完整的代码示例
- 使用统一的格式和风格
- 添加必要的图片和图表

## 📞 技术支持

如果在使用文档系统时遇到问题：

1. 查看 [MkDocs 官方文档](https://www.mkdocs.org/)
2. 参考 [Material 主题文档](https://squidfunk.github.io/mkdocs-material/)
3. 在项目 Issues 中提问

## 📝 许可证

本文档系统采用 MIT 许可证，与 NIR API 项目保持一致。

---

**NIR API 文档系统** - 为近红外光谱分析提供完整的文档支持
