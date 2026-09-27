# LeetCode 刷题笔记

记录我的 LeetCode 刷题过程。每道题都会写下：

- **我的理解**：题意拆解、容易踩的坑
- **解题步骤**：从暴力思路到优化解法，一步步推导
- **代码实现**：C++ / Python 实现
- **复杂度与小结**：时间空间复杂度、易错点、可迁移的套路

## 题目列表

| 题号 | 题目 | 难度 | 标签 |
| ---- | ---- | ---- | ---- |
| 3 | [无重复字符的最长子串](solutions/0003-longest-substring-without-repeating-characters.md) | 中等 | 滑动窗口 · 哈希表 |
| 239 | [滑动窗口最大值](solutions/0239-sliding-window-maximum.md) | 困难 | 单调队列 · 滑动窗口 |

> 🎬 带交互图解的题目：[239. 滑动窗口最大值](demos/0239-sliding-window-maximum.html ':ignore :target=_blank') —— 逐步动画演示 `deque` 的每一次进出队。

> 持续更新中。左侧侧边栏按标签分类浏览，侧栏搜索框可搜索本专题正文中的题号或关键词。

## 如何添加一道新题

新增内容遵循仓库根目录的 `AGENTS.md` 和 `design/DESIGN_SYSTEM.md`，按下面步骤操作：

1. **新建题解文件**：在 `solutions/` 下创建 `题号-英文题名.md`
   （示例：`0001-two-sum.md`、`0146-lru-cache.md`）

2. **按固定格式写题解**：复制下面的模板填入内容

   ````markdown
   # 题号. 题目名

   > **难度**：简单/中等/困难　**标签**：xxx · xxx
   > **链接**：[LeetCode 题号](https://leetcode.cn/problems/xxx/)

   ## 题目

   （粘贴题目描述和示例）

   ## 我的理解

   （题意拆解、容易踩的坑）

   ## 解题步骤

   ### 思路一：暴力（想清楚最笨的办法）
   ### 思路二：优化（核心解法）
   ### 代码（C++）

   ```cpp
   // 代码实现
   ```

   ### 复杂度分析
   - 时间复杂度：...
   - 空间复杂度：...

   ## 小结

   （可迁移的套路、易错点）
   ````

3. **登记统一索引**：在根目录 `content-index.json` 增加条目，填写 `collection: "leetcode"`、对应 `group`、源文件和 `leetcode/#/solutions/xxx` 路由。执行 `python scripts/build_site.py` 自动更新侧边栏与全站目录，不手工修改 `_sidebar.md`。

4. **更新题目列表**：在上方表格里加一行

5. **（可选）挂交互图解**：如果这道题适合动画演示，复制统一的 `templates/interactive.html`，放进 `demos/0239-sliding-window-maximum.html` 这样的路径，在内容索引登记，再在题解顶部链接它：

   ```markdown
   > 🎬 **配套交互图解**：[题目名图解](demos/xxxx-题目名.html ':ignore :target=_blank')
   ```

   `:ignore` 不能省。docsify 默认会把同源链接当成站内路由接管，点进去会把 HTML 当 markdown 渲染，页面直接乱掉；`:ignore` 让它按普通链接打开。

6. **检查**：执行 `python scripts/check_site.py`，并在浏览器中检查题解、图解、侧栏入口以及手机布局。
