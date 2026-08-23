# 3. 无重复字符的最长子串

> **难度**：中等　**标签**：哈希表 · 字符串 · 滑动窗口
> **链接**：[LeetCode 3](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

## 题目

给定一个字符串 `s`，找出其中不含有重复字符的 **最长子串** 的长度。

| 输入 | 输出 | 说明 |
| ---- | ---- | ---- |
| `"abcabcbb"` | `3` | 最长无重复子串是 `"abc"` |
| `"bbbbb"` | `1` | 最长无重复子串是 `"b"` |
| `"pwwkew"` | `3` | 最长无重复子串是 `"wke"`（`"pwke"` 是子序列，不是子串） |

## 我的理解

核心是「**子串**」两个字——它必须是**连续**的一段，不能跳着取。

最容易踩的两个坑：

1. 把「子串」和「子序列」搞混。`"pwke"` 是子序列但不是子串，因为它跳过了第二个 `'w'`。
2. 遇到重复字符时，不是把窗口「清空重来」，而是把左边界「滑」到重复消除为止——中间可能还保留着有用的字符。

一句话概括：**用左右两个指针框住一个「当前无重复」的窗口，不断尝试扩大右边界；一旦右边的新字符和窗口内重复，就收缩左边界直到重复消失，全程记录窗口的最大长度。**

## 解题步骤

### 思路一：暴力（先把最笨的办法想清楚）

枚举所有子串，逐个判断是否无重复，取最长。

- 起点 `i` 从 `0` 到 `n-1`，终点 `j` 从 `i` 到 `n-1`，共 `O(n²)` 个子串；
- 每个子串再用一个集合判断是否重复，又花 `O(n)`；
- 总复杂度 `O(n³)`，`n` 稍大就超时。

这一步的价值在于：明确了「无重复」这个判定可以用**集合**来做，为下面的优化打底。

### 思路二：滑动窗口（核心解法）

关键观察：**当我们固定左边界 `left`、让右边界 `right` 一直向右扩张时，「窗口是否无重复」是单调的**——一旦某一步出现重复，继续往右只会更糟，此时唯一能做的是把 `left` 右移来「补救」。

于是用两个指针 + 一个哈希集合：

1. `right` 逐个向右移动，把字符加入窗口；
2. 如果新字符已经在窗口里，就把 `left` 右移、并删除移出的字符，直到重复消失；
3. 每一步窗口都是合法的，用 `right - left + 1` 更新最大长度。

这样 `left` 和 `right` 各自最多走 `n` 步，总复杂度降到 `O(n)`。

### 代码（C++）

```cpp
int lengthOfLongestSubstring(string s) {
    unordered_set<char> window;
    int left = 0, ans = 0;

    for (int right = 0; right < (int)s.size(); ++right) {
        // 新字符重复时，收缩左边界，直到窗口内不再有该字符
        while (window.count(s[right])) {
            window.erase(s[left]);
            ++left;
        }
        window.insert(s[right]);
        ans = max(ans, right - left + 1);
    }
    return ans;
}
```

### 代码（Python）

```python
def lengthOfLongestSubstring(s: str) -> int:
    window = set()
    left = ans = 0
    for right, ch in enumerate(s):
        while ch in window:
            window.remove(s[left])
            left += 1
        window.add(ch)
        ans = max(ans, right - left + 1)
    return ans
```

### 复杂度分析

- **时间复杂度**：`O(n)`。`left` 和 `right` 都只会向右移动，最多各 `n` 次。
- **空间复杂度**：`O(Σ)`，`Σ` 为字符集大小（ASCII 下最多 128）。

## 小结

- 看到「子串 + 最值」，先想滑动窗口：用 `right` 扩、用 `left` 收，保证窗口始终满足约束。
- 「收缩」这一步用 `while` 而不是 `if`，因为一次右移可能还不够消除重复（例如 `"abca..."` 中再遇 `'a'` 需要连移两次）。
- 进阶：字符集小（如纯小写字母）时，可用 `int[128]` 当桶，省去哈希开销。
