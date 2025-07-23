# NIR API 文档

这是 NIR API 的官方文档，使用 MkDocs 构建。

## 文档结构

```
docs/
├── index.md                    # 首页
├── getting-started.md          # 快速开始指南
├── changelog.md               # 更新日志
├── api/                       # API 参考文档
│   ├── overview.md            # API 概览
│   ├── load_data.md           # 数据加载模块
│   ├── preprocessing.md       # 数据预处理模块
│   ├── ml_model.md           # 机器学习模块
│   ├── draw.md               # 数据可视化模块
│   ├── analysis.md           # 数据分析模块
│   ├── utils.md              # 工具模块
│   └── analysis_class.md     # 分析类模块
├── tutorials/                 # 教程
│   └── basic-usage.md        # 基础使用教程
└── examples/                  # 示例
    └── complete-workflow.md   # 完整工作流程示例
```

## 本地开发

### 安装依赖

```bash
pip install mkdocs mkdocs-material
```

### 启动开发服务器

```bash
cd nirapi
mkdocs serve
```

然后在浏览器中访问 `http://127.0.0.1:8000`

### 构建静态文档

```bash
mkdocs build
```

构建的文档将保存在 `site/` 目录中。

## 文档编写指南

### Markdown 语法

文档使用标准的 Markdown 语法，支持以下扩展：

- **代码高亮**: 使用 ```python 代码块
- **警告框**: 使用 !!! note, !!! warning, !!! tip 等
- **表格**: 标准 Markdown 表格语法
- **数学公式**: 使用 $$ 包围 LaTeX 公式

### 代码示例

```python
# 代码示例应该包含完整的导入语句
from nirapi import load_data, preprocessing, ML_model

# 提供实际可运行的示例
X = np.random.randn(100, 1200)
y = np.random.uniform(0, 10, 100)

# 添加注释说明每个步骤
X_processed = preprocessing.SNV(X)  # 标准正态变量变换
```

### 文档链接

- 使用相对路径链接其他文档页面
- API 模块之间相互引用
- 提供返回首页的链接

### 图片和媒体

- 图片保存在 `docs/images/` 目录
- 使用相对路径引用图片
- 提供图片的 alt 文本

## 配置说明

### mkdocs.yml

主要配置项：

- `site_name`: 网站名称
- `theme`: 使用 Material 主题
- `nav`: 导航结构
- `plugins`: 启用的插件
- `markdown_extensions`: Markdown 扩展

### 主题配置

使用 Material for MkDocs 主题，支持：

- 深色/浅色模式切换
- 中文搜索
- 代码复制功能
- 响应式设计

## 部署

### GitHub Pages

1. 推送代码到 GitHub 仓库
2. 在仓库设置中启用 GitHub Pages
3. 使用 GitHub Actions 自动构建和部署

### 自定义服务器

1. 运行 `mkdocs build` 构建静态文件
2. 将 `site/` 目录内容部署到 Web 服务器
3. 配置服务器支持 SPA 路由

## 贡献指南

### 添加新页面

1. 在相应目录创建 `.md` 文件
2. 在 `mkdocs.yml` 的 `nav` 部分添加页面
3. 更新相关页面的链接

### 更新现有页面

1. 直接编辑对应的 `.md` 文件
2. 开发服务器会自动重新加载
3. 检查链接和格式是否正确

### 代码文档

- API 文档应该包含完整的函数签名
- 提供参数说明和返回值说明
- 包含实际可运行的示例代码
- 说明常见的使用场景

## 常见问题

### Q: 如何添加新的 API 模块文档？

A: 
1. 在 `docs/api/` 目录创建新的 `.md` 文件
2. 在 `mkdocs.yml` 的导航中添加链接
3. 参考现有模块文档的格式

### Q: 如何修复链接错误？

A: 
1. 检查 `mkdocs build` 的警告信息
2. 确保链接的目标文件存在
3. 使用相对路径而不是绝对路径

### Q: 如何添加代码高亮？

A: 
使用三个反引号加语言名称：
```python
def example_function():
    return "Hello, World!"
```

### Q: 如何添加数学公式？

A: 
使用 LaTeX 语法：
$$E = mc^2$$

## 技术支持

如果在使用文档系统时遇到问题，请：

1. 查看 MkDocs 官方文档
2. 检查 Material 主题文档
3. 在项目 Issues 中提问

## 相关资源

- [MkDocs 官方文档](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Markdown 语法指南](https://www.markdownguide.org/)
- [Python 文档编写最佳实践](https://docs.python-guide.org/writing/documentation/)
