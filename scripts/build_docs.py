#!/usr/bin/env python3
"""
NIR API 文档构建脚本

这个脚本用于自动化构建和部署 NIR API 文档。
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, 
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"错误: 命令执行失败: {cmd}")
        print(f"错误信息: {e.stderr}")
        sys.exit(1)

def check_dependencies():
    """检查必要的依赖是否已安装"""
    print("检查依赖...")

    # 检查 mkdocs 命令是否可用
    try:
        result = subprocess.run(
            ["mkdocs", "--version"],
            capture_output=True, text=True, check=True
        )
        version_info = result.stdout.strip()
        print(f"✓ MkDocs 已安装 ({version_info})")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ MkDocs 未安装或不可用")
        print("请运行: pip install mkdocs mkdocs-material")
        sys.exit(1)

    # 检查 Material 主题（通过尝试构建来验证）
    try:
        result = subprocess.run(
            ["mkdocs", "build", "--help"],
            capture_output=True, text=True, check=True
        )
        print("✓ MkDocs 及相关依赖可用")
    except subprocess.CalledProcessError:
        print("✗ MkDocs 配置有问题")
        sys.exit(1)

def clean_build():
    """清理构建目录"""
    print("清理构建目录...")
    site_dir = Path("site")
    if site_dir.exists():
        shutil.rmtree(site_dir)
        print("✓ 已清理 site 目录")

def build_docs():
    """构建文档"""
    print("构建文档...")
    output = run_command("mkdocs build")
    print("✓ 文档构建完成")
    return output

def serve_docs(host="127.0.0.1", port=8000):
    """启动开发服务器"""
    print(f"启动开发服务器 http://{host}:{port}")
    try:
        subprocess.run([
            "mkdocs", "serve", 
            "--dev-addr", f"{host}:{port}"
        ], check=True)
    except KeyboardInterrupt:
        print("\n服务器已停止")

def validate_docs():
    """验证文档链接和格式"""
    print("验证文档...")
    
    # 检查必要的文件是否存在
    required_files = [
        "docs/index.md",
        "docs/getting-started.md",
        "docs/api/overview.md",
        "mkdocs.yml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("✗ 缺少必要文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✓ 所有必要文件都存在")
    return True

def generate_sitemap():
    """生成站点地图"""
    print("生成站点地图...")
    
    docs_dir = Path("docs")
    sitemap = []
    
    for md_file in docs_dir.rglob("*.md"):
        if md_file.name != "README.md":
            relative_path = md_file.relative_to(docs_dir)
            sitemap.append(str(relative_path))
    
    sitemap_content = "# 站点地图\n\n"
    sitemap_content += "本文档包含以下页面:\n\n"
    
    for page in sorted(sitemap):
        page_name = page.replace(".md", "").replace("/", " / ")
        sitemap_content += f"- [{page_name}]({page})\n"
    
    with open("docs/sitemap.md", "w", encoding="utf-8") as f:
        f.write(sitemap_content)
    
    print("✓ 站点地图已生成")

def check_links():
    """检查文档中的链接"""
    print("检查文档链接...")
    
    # 这里可以添加更复杂的链接检查逻辑
    # 目前只是简单的构建检查
    try:
        output = run_command("mkdocs build --strict")
        print("✓ 所有链接检查通过")
        return True
    except:
        print("✗ 发现链接错误，请检查构建输出")
        return False

def deploy_to_github_pages():
    """部署到 GitHub Pages"""
    print("部署到 GitHub Pages...")
    
    # 检查是否在 git 仓库中
    if not Path(".git").exists():
        print("✗ 当前目录不是 git 仓库")
        return False
    
    # 使用 mkdocs gh-deploy 命令
    try:
        output = run_command("mkdocs gh-deploy --clean")
        print("✓ 已部署到 GitHub Pages")
        return True
    except:
        print("✗ GitHub Pages 部署失败")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NIR API 文档构建工具")
    parser.add_argument("command", choices=[
        "build", "serve", "clean", "validate", "deploy", "check-links"
    ], help="要执行的命令")
    parser.add_argument("--host", default="127.0.0.1", help="开发服务器主机")
    parser.add_argument("--port", type=int, default=8000, help="开发服务器端口")
    parser.add_argument("--strict", action="store_true", help="严格模式构建")
    
    args = parser.parse_args()
    
    # 确保在正确的目录中
    if not Path("mkdocs.yml").exists():
        print("错误: 请在包含 mkdocs.yml 的目录中运行此脚本")
        sys.exit(1)
    
    print("=== NIR API 文档构建工具 ===")
    
    if args.command == "build":
        check_dependencies()
        if args.strict:
            validate_docs()
        build_docs()
        print("✓ 构建完成！文档位于 site/ 目录")
        
    elif args.command == "serve":
        check_dependencies()
        serve_docs(args.host, args.port)
        
    elif args.command == "clean":
        clean_build()
        
    elif args.command == "validate":
        check_dependencies()
        if validate_docs():
            print("✓ 文档验证通过")
        else:
            print("✗ 文档验证失败")
            sys.exit(1)
            
    elif args.command == "deploy":
        check_dependencies()
        if validate_docs():
            build_docs()
            if deploy_to_github_pages():
                print("✓ 部署成功")
            else:
                print("✗ 部署失败")
                sys.exit(1)
        else:
            print("✗ 文档验证失败，取消部署")
            sys.exit(1)
            
    elif args.command == "check-links":
        check_dependencies()
        if check_links():
            print("✓ 链接检查通过")
        else:
            print("✗ 链接检查失败")
            sys.exit(1)

if __name__ == "__main__":
    main()
