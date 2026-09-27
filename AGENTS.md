# Linkforfuture 站点约束

这是个人技术网站，使用静态 HTML、Markdown 与 Docsify。不是小说项目；不要创建 `.webnovel`、执行小说初始化或引入无关工作流。

## 开始改动前

1. 阅读 `design/DESIGN_SYSTEM.md`。它是当前生效的设计规范；`design/unified-design-plan.md` 只是历史提案。
2. 查看 `content-index.json`，确认内容所属专题、文档集合与内容类型。
3. 保持原有文章路径、锚点、技术内容和交互语义。不要为调整视觉同时升级 Docsify、引入新框架或替换算法演示。

## 必须遵守

- 全站外框和排版复用 `assets/css/tokens.css`、`site.css`；文档站使用 `docs.css` 与 `assets/js/docs.js`。
- 品牌色、文字颜色、基础字体、圆角、宽度在 tokens 中管理。页面不得另建一套全局色板、导航、Hero 或通用按钮。
- 新文章使用 `templates/article.html`；交互内容使用 `templates/interactive.html`；普通文档优先 Markdown，参考 `templates/article.md`。
- HTML 必须保留 `LF:HEAD`、`LF:HEADER`、`LF:FOOTER` 标记。生成器会更新这些区块；不要手动复制导航到页面。
- 页面专属 CSS 必须放在 `assets/css/`，限定到该页面独有的根类，例如 `.demo-mesh`。不得使用裸 `body`、`:root`、`button`、`header`、`.sidebar` 等选择器污染其他页面。不要新增内联 `<style>` 或内联脚本。
- 图表的语义色可以独立定义，但不能只靠颜色传递信息；沿用图例和文字标签。不要把所有通路、rank 或状态强行改成品牌色。
- `assets/css/pages/` 与 `legacy.css` 是存量图解的兼容边界，不是新页面模板。新增内容不得继续扩展迁移兼容层。
- 所有发布内容登记到 `content-index.json`。MD 填写 collection 和 group，HTML 填写 source 与稳定 URL。更新日期必须真实；未知填 null。
- `index.html`、`topics.html`、`diagrams.html`、三个集合的 `index.html` / `_sidebar.md` / `_404.md`、`assets/js/content-data.js` 是生成文件。修改索引或生成器后运行构建，不直接编辑生成结果。
- 目录页搜索仅筛选标题与摘要；文档搜索限当前集合正文。不得把它描述为全站全文搜索。
- Docsify 内指向独立 HTML 的 Markdown 链接必须加 `:ignore`，例如 `[交互图解](demos/example.html ':ignore')`。
- 允许浏览器缩放。手机正文不能靠整体缩小保持桌面布局；宽代码、表格和复杂图仅在局部滚动。
- 简历保留屏幕编辑、HTML 下载、进度复选框和 A4 打印；不得因抽取样式破坏离线下载。
- 不直接编辑 vendor 文件。本地依赖升级须作为明确的独立工作处理。

## 完成前

```sh
python scripts/build_site.py
python scripts/check_site.py
```

涉及布局或交互时，在本地 HTTP 服务中检查修改页面的桌面与 360px 窄屏，测试实际交互和键盘操作。涉及 Docsify 时检查深链接、刷新、前进后退、目录、正文搜索和 Markdown 图片。检查脚本不代替浏览器验证。

提交说明应写明影响页面、验证结果与未验证事项。未经用户要求，不发布、推送或部署。
