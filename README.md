# linkforfuture

我的 GitHub Pages 站点仓库。

https://linkforfuture.github.io/

## 站点维护

网站使用统一的静态页面外框；Markdown 文档由 Docsify 阅读，交互图解保留 HTML。添加内容前先阅读 [AGENTS.md](AGENTS.md) 和 [设计规范](design/DESIGN_SYSTEM.md)。

- 所有内容登记在 [content-index.json](content-index.json)，模板在 [templates/](templates/)。
- 首页、专题目录、图解目录、集合入口、侧栏和公共导航由生成器维护。
- 全站样式在 `assets/css/`，文档集合共享 `assets/vendor/docsify/` 的原版本地依赖。

```sh
python scripts/build_site.py
python scripts/check_site.py
python -m http.server 8000 --bind 127.0.0.1
```

构建只需 Python 3.10+；检查还需要 Node.js 验证 JS 语法，无需安装 npm 包。通过 `http://127.0.0.1:8000` 预览；不要用双击 HTML 的方式检查 Docsify。CI 检查内容登记、生成结果、站内链接和设计约束，不执行部署。

目录页筛选标题和摘要，文档站搜索当前专题正文。涉及布局的修改还需要检查桌面和手机，并操作实际图解控件；自动检查不能代替视觉与交互验证。

## CUDA 实践学习

[在线学习入口](https://linkforfuture.github.io/cuda/) · [课程目录](cuda/README.md) · [远端 GPU 接续](cuda/remote-gpu.md)

环境搭建、动手课程、CUDA 源码和实验记录统一保存在 `cuda/`，可在本机 WSL 或远端 GPU 服务器拉取并继续学习。
