# Linkforfuture 设计与内容规范

状态：现行规范，2026-09-27 起生效。后续维护以本文件、根目录 `AGENTS.md` 和代码中的共享 tokens 为准。

## 网站定位与内容结构

这是 **Linkforfuture 的个人主页**，记录工作、学习与思考。一级导航为「首页 / 工作 / 学习 / 关于我」。学习只是其中一个板块；现有文章以技术为主，不代表网站只能承载技术内容。

首页先回答「这是谁的个人空间」，以本人名称、简短介绍和并列板块入口构成。不能以技术方向、课程、学习路径、内容数量或知识目录作为主视觉。首页不自动汇总技术精选，避免某个板块因文章数量多而主导全站。

| 层级 | 职责 | 当前例子 |
| --- | --- | --- |
| 一级板块 `section` | 个人主页的内容分区，决定全站导航归属 | work 工作、learning 学习、about 关于我 |
| 专题 `topic` | 板块内的主题标签，不能代替一级板块 | 集合通信、GPU 编程、算法、模型 |
| 集合 `collection` | Markdown 文档的阅读目录 | work、cuda、leetcode |
| 类型 `kind` | 单篇内容的呈现方式或用途 | 学习笔记、设计文档、图解 |

`content-index.json` 的 sections 定义一级导航与首页入口；每个条目和集合都必须标注 section。同一技术 topic 可以横跨工作与学习；学习目录只列 section=learning 的内容，工作文档留在工作板块。图解实验是学习的下级入口。为保持外链稳定，学习入口仍使用 `topics.html`。

「关于我」是独立的 `about.html`，介绍本人和小站；含待补全信息的简历草稿作为下级资料明确标注。未来增加随笔、生活、兴趣、作品时，可以扩展与工作、学习并列的板块；不必硬塞进技术分类，也不创建没有内容的占位栏目。个人经历、爱好、照片、动态必须来自用户或既有可靠资料，不替用户编造。

MD 与 HTML 是创作形式，不是用户导航分类。页面以学习笔记、技术文档、设计文档、分析报告、图解、交互图解、实验等类型展示。

`featured` 暂保留为内容元数据，不触发首页展示。没有可靠更新时间时不显示「最近更新」列表。新增内容不得擅自改变站点定位或首页的信息优先级。

## 视觉基础

所有真实值在 `assets/css/tokens.css` 中统一管理：

| 用途 | Token | 当前值 |
| --- | --- | --- |
| 页面 / 内容面 | `--lf-bg` / `--lf-surface` | `#f7f8fa` / `#ffffff` |
| 正文 / 次要文字 | `--lf-text` / `--lf-muted` | `#1f2937` / `#5c6b80` |
| 分隔线 | `--lf-line` | `#dfe5ed` |
| 品牌 / 轻强调 | `--lf-accent` / `--lf-tint` | `#4f46e5` / `#eef2ff` |
| 状态 | `--lf-success` / `--lf-warning` / `--lf-danger` | 成功 / 提醒 / 错误 |
| 字体 | `--lf-sans` / `--lf-mono` | 系统中文字体 / 等宽字体 |
| 圆角 | `--lf-radius` / `--lf-control-radius` | 10px / 6px |
| 内容 / 阅读宽度 | `--lf-width` / `--lf-reading` | 1160px / 780px |
| 间距 | `--lf-space-*` | 4、8、12、16、24、32、48px |

正文默认 17px / 1.85，手机 16px。页面标题 30–46px；文章一级标题约 38px，二级 25px，三级 20px；辅助文字 13–14px。图中标签按图的实际尺寸设计，不能把整个 SVG 缩小到不可读。

第一期提供统一浅色主题。不要在单个页面中另开自动深色模式；未来新增深色需一起覆盖公共界面、代码、图表和打印。

品牌色用于链接、选中状态和主要操作。绿色、橙色等用于数据或状态；图表语义色必须配文字、图例或形状，且保持同页一致。避免大面积渐变、装饰性 Emoji、重阴影和每段文字一个卡片。

## 页面结构与组件

- 所有页面共用生成的 `.lf-header` 与 `.lf-footer`。窄屏导航是原生 details；不依赖脚本也可以展开。
- 首页：`.lf-home-hero`、`.lf-person-name`、`.lf-space-grid`、`.lf-space-card`，姓名和个人介绍优先，板块等权重呈现。
- 目录：`.lf-container`、`.lf-title`、`.lf-lead`、`.lf-entry`；专题、筛选和课程入口只出现在相应板块中。
- 阅读：`.lf-prose`，连续正文与标题层级；表格、代码采用局部滚动。不要把普通段落放进演示卡片。
- 交互：文章头与解释使用共同排版；`.lf-scroll` 容纳必要的宽图；交互控件使用 `.lf-button`，主操作增加 `.lf-button-primary`。
- 提示使用 `.lf-note`；短内容类型使用 `.lf-tag`；按钮必须是 button，导航必须是 a，参数必须有 label。
- Markdown 集合通过 `docs.css` 适配 Docsify：左侧是文章目录，正文可展开的「本页目录」是文章章节，两者职责不同。
- 简历使用同一站点外框，内部保留 A4 专用排版。打印隐藏导航、面包屑及工具栏。下载 HTML 会内联本页依赖，继续支持离线编辑和打印。

推荐页面布局：

```text
共同页头
面包屑
文章标题 + 简述 + 类型 / 真实日期
正文（约 780px）或交互画布（可扩到 1280px）
相关内容 / 下一篇
共同页脚
```

## 文件职责

```text
content-index.json        内容登记，一级板块、专题与集合定义
assets/css/tokens.css     设计变量
assets/css/site.css       外框、通用组件、普通文章
assets/css/docs.css       Docsify 结构适配
assets/css/legacy.css     存量页面的共同外观适配
assets/css/pages/         已迁移图解的隔离样式（@layer legacy）
assets/js/site.js         菜单与目录筛选
assets/js/docs.js         集合配置、文章导航、代码复制
assets/js/pages/          现有图解与简历的独立交互
assets/vendor/docsify/    合并后的原版本地依赖
templates/               新文章、图解与 MD 起点
scripts/build_site.py    生成首页、目录、集合入口、侧栏及外框
scripts/check_site.py    设计契约、登记、链接、JS 语法检查
```

存量图解样式从 HTML 中迁出，并限定在 `.lf-legacy[data-legacy="…"]` 内，通过 CSS layer 降低优先级。共享规则负责颜色、导航、阅读尺寸和组件；页面样式负责图形几何、数据配色、动画、演示控件与打印。不得复制旧页面整套 CSS 来创建新页面。

## 添加内容

### Markdown

1. 在对应集合目录创建 `.md`，以唯一 `# 标题` 开始，不嵌入全局样式或脚本。
2. 在索引 `entries` 中登记 source、url、section、topic、kind、collection、group，section 必须与集合一致；顺序决定侧栏和上一篇/下一篇顺序。同组条目放在一起。
3. `source` 是文件路径，例如 `cuda/lesson-02.md`；`url` 是阅读路由，例如 `cuda/#/lesson-02`。README 的 URL 使用 `cuda/#/`。
4. 图解链接增加 Docsify `:ignore` 标记。图片路径相对源文档，构建后检查实际显示。
5. 执行构建与检查。无需手工同步首页计数或侧栏。

### HTML

1. 复制 `templates/article.html` 或 `templates/interactive.html` 到合适目录。
2. 设置标题、description、唯一 data-page 和正文。保留三个 LF 生成区块以及正文 `id="lf-content"`。
3. 专属交互拆到独立 JS；专属样式使用唯一页面根类限定。引用资源时按当前 HTML 文件层级写相对路径；公共外框的路径由生成器处理。
4. 在索引登记。`source` 与 `url` 均为真实 HTML 路径。
5. 执行构建与检查，检查无 JavaScript 时文章基本内容仍然可读。

索引示例（示例路径不会自动创建文件）：

```json
{
  "id": "cuda-lesson-02",
  "title": "第二课：性能计时",
  "summary": "使用 CUDA Event 区分 kernel 时间和端到端耗时。",
  "section": "learning",
  "topic": "gpu",
  "kind": "学习笔记",
  "source": "cuda/lesson-02.md",
  "url": "cuda/#/lesson-02",
  "collection": "cuda",
  "group": "动手实验",
  "updated": null,
  "featured": false
}
```

不要自动生成假的日期、阅读时长、完成率或性能指标。日期未知为 null，正文中的数据应保留来源与适用条件。

### 扩展个人板块

先确认用户已有相应内容，再在 sections 登记板块、入口 URL 和真实描述；为内容设置 section，并按实际需要添加 topic 或 Markdown 集合。入口页面需复用公共外框，加入生成器或登记为普通 HTML。`templates/article.md` 只是技术文章示例，随笔等内容可以自由安排正文标题，不能强制套用「背景 / 实现 / 验证」。扩展后检查导航、首页入口、面包屑与站内链接；不要根据文件格式决定导航位置。

## 检查与发布边界

```sh
python scripts/build_site.py
python scripts/check_site.py
python -m http.server 8000 --bind 127.0.0.1
```

前两项需要 Python 3.10+ 和 Node.js（仅用于 JS 语法检查）；无 npm 依赖、无打包步骤。生成的静态结果可直接由 GitHub Pages 提供。请通过 HTTP 预览，Docsify 不支持依靠双击 file:// 完整加载集合。

自动检查覆盖：生成结果一致性、内容登记、重复 ID、公共外框与 CSS 接入、HTML 结构、静态资源、站内链接/锚点、Docsify 内容路由、JS 语法。CI 运行同一检查，不会自动发布。

浏览器手检覆盖：桌面与 360px、菜单和专题目录、深链接与刷新、正文搜索、代码复制、图解前进/回退/参数操作、简历编辑/下载/打印。宽图允许局部滚动，整个页面不能溢出；保持可见键盘焦点和浏览器缩放。

目录筛选只查标题与摘要；Docsify 搜索只查当前集合。全站全文搜索、统一深色模式及生成器迁移不属于当前实现，后续应作为明确需求单独设计。
