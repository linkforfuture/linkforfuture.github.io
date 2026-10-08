/* LeetCode 226 翻转二叉树：生成递归执行轨迹并逐步渲染树、调用栈与代码。 */

const IT_ROOT = 4;
const IT_IDS = [4, 2, 7, 1, 3, 6, 9];

const IT_CODE = [
  'class Solution {',
  'public:',
  '    TreeNode* invertTree(TreeNode* root) {',
  '        if (root == nullptr) {',
  '            return nullptr;',
  '        }',
  '        TreeNode* left = invertTree(root-&gt;left);',
  '        TreeNode* right = invertTree(root-&gt;right);',
  '        root-&gt;left = right;',
  '        root-&gt;right = left;',
  '        return root;',
  '    }',
  '};'
];

const IT_OP = {
  init: '准备', call: '下降 · 调用', null: '触底 · 返回',
  assign: '上升 · 接住', swap: '交换', done: '完成', fin: '复盘'
};

const IT_INIT_DESC = '这棵树的层序是 <code>[4,2,7,1,3,6,9]</code>，目标是变成它的镜像 <code>[4,7,2,9,6,3,1]</code>。'
  + '开始之前先把这份代码的<b>契约</b>记住：给它任何一棵子树，它把子树里<b>每个节点</b>的左右孩子互换，然后把根还给你。'
  + '接下来三十来步，就是看它怎么兑现这个契约。';

const IT_CALL_DESC = {
  4: '<code>root</code> 指向节点 4，不是空指针，第 4 行的刹车没有触发。接下来要执行第 7 行——'
    + '可它需要的是「左子树<b>翻转好之后</b>的样子」，这个结果只能再调一次自己才拿得到。于是这一帧<b>停在第 7 行</b>，把活儿派下去。',
  2: '新帧压栈。此刻的分工是：<b>4 那一帧冻结在第 7 行等结果</b>，2 这一帧负责把以 2 为根的子树整个翻好再交回去。'
    + '递归「下降」就是这样一帧压一帧——每一层都在等下一层，<b>谁还没有真正开始交换</b>。',
  1: '2 同样停在第 7 行，把任务继续往下派。图上高亮的 4 → 2 → 1 这条链，就是当前调用栈里活着的路径；'
    + '右边三帧叠着，只有最上面那帧在真正执行。',
  3: '左手已经拿稳，2 继续执行第 8 行，把右子树也派出去。注意它<b>不会先交换</b>——交换必须等两只手都拿满。',
  7: '左半边完工，右半棵树开始。接下来 7、6、9 会把刚才 2、1、3 的过程<b>原样重演一遍</b>——'
    + '这就是递归：同一份代码，换一批节点，逻辑一模一样。看懂了左半边，右半边自动就看懂了。',
  6: '7 停在第 7 行，把左子树 6 派出去。当前调用路径变成 4 → 7 → 6。',
  9: '7 的左手已就位，继续第 8 行，把右子树 9 派出去。'
};

const IT_NULL_DESC = {
  '1:left': '1 的左孩子是空指针。这次调用照样压了一帧进来，但一进来就满足 <code>root == nullptr</code>，'
    + '第 5 行直接 <code>return nullptr</code>，随即出栈——它在栈上只停留一瞬，什么也没做。'
    + '<b>这就是递归能停下来的唯一原因</b>：没有这个出口，调用会顺着不存在的孩子一路冲下去。返回后，1 这一帧的 <code>left = nullptr</code>。',
  '1:right': '右孩子同样是空，第 8 行的调用也立刻返回 <code>nullptr</code>。'
    + '现在 1 的两只手都拿到了东西（虽然都是空），轮到自己交换了。'
};

const IT_ASSIGN_DESC = {
  '1->2.left': '1 这一帧执行第 11 行 <code>return root</code>，然后出栈；节点 1 打上 ✓，含义是「以它为根的子树已翻转完毕」。'
    + '控制权回到 2 那一帧的第 7 行，局部变量 <code>left</code> 接住这棵<b>已经翻好</b>的子树。<b>递归从这里开始上升。</b>',
  '3->2.right': '3 出栈，✓ 到手。现在 2 这一帧的 <code>left</code> 和 <code>right</code> 都齐了：左手是翻好的 1，右手是翻好的 3。'
    + '<b>两棵都是成品</b>，交换的条件终于满足。',
  '2->4.left': '2 出栈，左半棵树（2、1、3）全部完工，✓ 已经打满。4 的 <code>left</code> 接住这棵成品子树，接着去处理右半边。',
  '6->7.left': '6 出栈，7 的 <code>left</code> 接住成品。和左半边的节奏完全一致。',
  '9->7.right': '9 出栈，7 的两只手都满了：<code>left</code> 是翻好的 6，<code>right</code> 是翻好的 9。',
  '7->4.right': '7 出栈，右半棵树完工。现在根节点 4 的两只手都拿满了：左手是翻好的 2 子树，右手是翻好的 7 子树。整棵树只差最后一下。'
};

const IT_SWAP_DESC = {
  1: '第 9、10 行执行。1 是叶子，交换的是两个空指针，画面上什么都没变。<b>但代码一行都没少跑</b>——'
    + '递归从不区分「叶子」和「非叶子」，它只区分「空」和「非空」。正是这份不特判，让几行代码能处理任意形状的树。',
  3: '交换两个 <code>nullptr</code>，同样没有可见变化。叶子节点走的是和非叶子完全相同的流程。',
  6: '交换两个 <code>nullptr</code>，流程与节点 1、3 完全一样。',
  9: '交换两个 <code>nullptr</code>。至此四个叶子都走完了自己的流程。',
  2: '<b>全题最关键的一步。</b>第 9 行 <code>root-&gt;left = right</code>、第 10 行 <code>root-&gt;right = left</code>：'
    + '1 和 3 在图上换了位置。注意交换的对象是<b>已经翻好的子树</b>——2 完全不需要知道 1、3 内部发生过什么，'
    + '它只负责把两个成品对调、挂回自己身上。到这里，以 2 为根的整棵子树翻转完毕。',
  7: '和节点 2 那一步完全对称：把翻好的 6、9 对调挂回。右半棵树也就此完工。',
  4: '根节点把左右两棵成品子树对调。<b>注意它为什么最后才发生</b>——交换的前提是「两个孩子都已翻好」，'
    + '而这个条件只能从叶子一层层往上满足。所以翻转是<b>自底向上</b>完成的：离根越近，交换越晚。'
};

const IT_DONE_DESC = '第 11 行把根节点交还给最初的调用者，栈清空，所有节点都打上了 ✓。'
  + '输入 <code>[4,2,7,1,3,6,9]</code>，输出 <code>[4,7,2,9,6,3,1]</code>，正是题目要求的镜像。';

const IT_FIN_DESC = '每个节点一生只做三件事：① 等左子树的成品；② 等右子树的成品；③ 把两个成品对调，再把自己交还上一层。'
  + '「翻转整棵树」听起来很大，摊到每个节点上只是两次指针赋值。<b>时间 O(n)</b>：每个节点恰好进栈一次、出栈一次。'
  + '<b>空间 O(h)</b>：同一时刻栈里只有「从根到当前节点」这一条路径，最长等于树高；这棵树最深时栈里有 4 帧（含触底的那一帧）。';

function itFreshTree() {
  return { 4: [2, 7], 2: [1, 3], 7: [6, 9], 1: [null, null], 3: [null, null], 6: [null, null], 9: [null, null] };
}

function buildItSteps() {
  const tree = itFreshTree();
  const steps = [];
  const stack = [];
  const done = [];

  function snap(extra) {
    const shot = {};
    IT_IDS.forEach(id => { shot[id] = [tree[id][0], tree[id][1]]; });
    steps.push(Object.assign({
      tree: shot,
      stack: stack.map(f => ({ id: f.id, left: f.left, right: f.right })),
      done: done.slice(),
      pulse: [],
      ghost: null,
      lines: [],
      op: 'init',
      title: '',
      desc: ''
    }, extra));
  }

  function invert(id, callerLine) {
    const frame = { id, left: 'pending', right: 'pending' };
    stack.push(frame);
    snap({
      op: 'call',
      lines: [3, 4],
      title: callerLine === null ? `进入 invertTree(${id})` : `第 ${callerLine} 行发起调用 → 进入 invertTree(${id})`,
      desc: IT_CALL_DESC[id]
    });

    function handleSide(slot, line, name) {
      const child = tree[id][slot];
      if (child === null) {
        frame[name] = 'calling';
        stack.push({ id: null, left: 'na', right: 'na' });
        snap({
          op: 'null',
          lines: [4, 5],
          ghost: { parent: id, side: name },
          title: `${name === 'left' ? '左' : '右'}孩子是空 → invertTree(nullptr) 立刻返回`,
          desc: IT_NULL_DESC[`${id}:${name}`]
            || `${name === 'left' ? '左' : '右'}孩子是空指针，调用照样压一帧进来，一进来就命中第 4 行的刹车，`
              + `第 5 行 <code>return nullptr</code> 后随即出栈。节点 ${id} 的 <code>${name}</code> 接住这个 <code>nullptr</code>。`
        });
        stack.pop();
        frame[name] = 'nullptr';
        return;
      }
      frame[name] = 'calling';
      invert(child, line);
      frame[name] = child;
      snap({
        op: 'assign',
        lines: [11, line],
        title: `return ${child} → 回到 ${id} 的第 ${line} 行，${name} 接住`,
        desc: IT_ASSIGN_DESC[`${child}->${id}.${name}`]
          || `${child} 这一帧执行第 11 行 <code>return root</code> 后出栈，节点 ${child} 打上 ✓。`
            + `控制权回到 ${id} 的第 ${line} 行，<code>${name}</code> 接住这棵已经翻好的子树。`
      });
    }

    handleSide(0, 7, 'left');
    handleSide(1, 8, 'right');

    const l = tree[id][0];
    const r = tree[id][1];
    tree[id] = [r, l];
    snap({
      op: 'swap',
      lines: [9, 10],
      pulse: [l, r].filter(x => x !== null),
      title: l === null && r === null
        ? `节点 ${id}：交换两个孩子（都是 nullptr）`
        : `节点 ${id}：交换两个孩子`,
      desc: IT_SWAP_DESC[id]
    });

    stack.pop();
    done.push(id);
  }

  snap({ op: 'init', lines: [], title: `准备：调用 invertTree(${IT_ROOT})`, desc: IT_INIT_DESC });
  invert(IT_ROOT, null);
  snap({ op: 'done', lines: [11], title: 'return root · 整棵树翻转完成', desc: IT_DONE_DESC });
  snap({ op: 'fin', lines: [], title: '复盘：每个节点一生只做三件事', desc: IT_FIN_DESC });
  return steps;
}

const IT_STEPS = buildItSteps();

const itTreeEl = document.getElementById('it-tree');
const itStackEl = document.getElementById('it-stack');
const itOpEl = document.getElementById('it-op');
const itTtEl = document.getElementById('it-tt');
const itDsEl = document.getElementById('it-ds');
const itCodeEl = document.getElementById('it-code');
const itCounterEl = document.getElementById('it-counter');
const itSlider = document.getElementById('it-slider');

itCodeEl.innerHTML = IT_CODE
  .map((line, i) => `<span class="ln" data-line="${i + 1}"><span class="num">${i + 1}</span>${line}</span>`)
  .join('');

function itLayout(tree) {
  const pos = {};
  let i = 0;
  (function walk(id, depth) {
    if (id === null) return;
    walk(tree[id][0], depth + 1);
    pos[id] = { x: 75 + i * 85, y: 46 + depth * 96 };
    i += 1;
    walk(tree[id][1], depth + 1);
  })(IT_ROOT, 0);
  return pos;
}

function itRenderTree(step) {
  const pos = itLayout(step.tree);
  const doneSet = new Set(step.done);
  const pathSet = new Set(step.stack.filter(f => f.id !== null).map(f => f.id));
  let activeId = null;
  for (let i = step.stack.length - 1; i >= 0; i -= 1) {
    if (step.stack[i].id !== null) { activeId = step.stack[i].id; break; }
  }

  let out = '<defs><marker id="it-mk-swap" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" '
    + 'markerHeight="5.5" orient="auto-start-reverse"><path class="it-mk-path" d="M0,0 L10,5 L0,10 z"></path></marker></defs>';

  IT_IDS.forEach(id => {
    step.tree[id].forEach(child => {
      if (child === null) return;
      const hot = step.pulse.indexOf(child) >= 0 ? ' it-edge-hot' : '';
      out += `<line class="it-edge${hot}" x1="${pos[id].x}" y1="${pos[id].y + 21}" x2="${pos[child].x}" y2="${pos[child].y - 21}"></line>`;
    });
  });

  if (step.ghost) {
    const p = pos[step.ghost.parent];
    const dx = step.ghost.side === 'left' ? -34 : 34;
    const gx = p.x + dx;
    const gy = p.y + 78;
    out += `<line class="it-ghost-line" x1="${p.x + dx * 0.45}" y1="${p.y + 19}" x2="${gx}" y2="${gy - 12}"></line>`;
    out += `<g class="it-ghost"><rect x="${gx - 27}" y="${gy - 12}" width="54" height="24" rx="12"></rect>`
      + `<text x="${gx}" y="${gy + 4}">nullptr</text></g>`;
  }

  if (step.pulse.length === 2) {
    const a = pos[step.pulse[0]];
    const b = pos[step.pulse[1]];
    const mx = (a.x + b.x) / 2;
    const dir = a.x < b.x ? 1 : -1;
    out += `<path class="it-swap-arc" d="M ${a.x + 14 * dir} ${a.y - 16} Q ${mx} ${a.y - 46} ${b.x - 14 * dir} ${b.y - 16}" `
      + 'marker-start="url(#it-mk-swap)" marker-end="url(#it-mk-swap)"></path>';
    out += `<text class="it-swap-label" x="${mx}" y="${a.y - 56}">互换</text>`;
  }

  IT_IDS.forEach(id => {
    const p = pos[id];
    const cls = ['it-node'];
    if (pathSet.has(id)) cls.push('is-path');
    if (id === activeId) cls.push('is-active');
    if (doneSet.has(id)) cls.push('is-done');
    if (step.pulse.indexOf(id) >= 0) cls.push('is-pulse');
    out += `<g class="${cls.join(' ')}"><circle class="it-circle" cx="${p.x}" cy="${p.y}" r="21"></circle>`
      + `<text class="it-val" x="${p.x}" y="${p.y + 5}">${id}</text>`;
    if (doneSet.has(id)) {
      out += `<g class="it-ok"><circle cx="${p.x + 15}" cy="${p.y - 15}" r="8.5"></circle>`
        + `<text x="${p.x + 15}" y="${p.y - 11}">✓</text></g>`;
    }
    out += '</g>';
  });

  itTreeEl.innerHTML = out;
  itTreeEl.setAttribute('aria-label', `二叉树当前状态：${step.title}`);
}

function itValCls(v) {
  if (v === 'pending') return 'it-v-none';
  if (v === 'calling') return 'it-v-call';
  if (v === 'nullptr') return 'it-v-null';
  return 'it-v-done';
}

function itValTxt(v) {
  if (v === 'pending') return '—（还没调用）';
  if (v === 'calling') return '调用中…';
  if (v === 'nullptr') return 'nullptr';
  return `${v} · 已翻转`;
}

function itFrameStatus(f) {
  if (f.left === 'pending') return '刚进入 · 第 4 行判空';
  if (f.left === 'calling') return '停在第 7 行 · 等左子树返回';
  if (f.right === 'calling') return '停在第 8 行 · 等右子树返回';
  if (f.right === 'pending') return '左手已就位 · 即将执行第 8 行';
  return '两手都就位 · 第 9–10 行交换';
}

function itRenderStack(step) {
  const frames = step.stack.slice().reverse();
  if (!frames.length) {
    itStackEl.innerHTML = '<div class="it-empty">[ 空 ] — 所有调用都已返回</div>';
    return;
  }
  itStackEl.innerHTML = frames.map((f, i) => {
    if (f.id === null) {
      return '<div class="it-frame it-frame-top it-frame-null">'
        + '<div class="it-frame-hd"><span>invertTree(nullptr)</span><span class="it-topmark">栈顶</span></div>'
        + '<div class="it-frame-st">第 4 行命中 → 第 5 行 return nullptr</div></div>';
    }
    const top = i === 0;
    return `<div class="it-frame${top ? ' it-frame-top' : ''}">`
      + `<div class="it-frame-hd"><span>invertTree(${f.id})</span>`
      + (top ? '<span class="it-topmark">栈顶 · 正在执行</span>' : '<span class="it-wait">挂起等待</span>')
      + '</div>'
      + `<div class="it-row"><span class="it-k">left</span><span class="${itValCls(f.left)}">${itValTxt(f.left)}</span></div>`
      + `<div class="it-row"><span class="it-k">right</span><span class="${itValCls(f.right)}">${itValTxt(f.right)}</span></div>`
      + `<div class="it-frame-st">${itFrameStatus(f)}</div></div>`;
  }).join('');
}

let itCur = 0;
let itTimer = null;
const itPlayBtn = document.getElementById('it-btn-play');

function itStopPlay() {
  if (itTimer) { clearInterval(itTimer); itTimer = null; }
  itPlayBtn.textContent = '自动播放';
}

function itRender(idx) {
  idx = Math.max(0, Math.min(IT_STEPS.length - 1, idx));
  itCur = idx;
  const step = IT_STEPS[idx];

  itRenderTree(step);
  itRenderStack(step);

  itOpEl.className = `it-op${step.op === 'init' || step.op === 'fin' ? '' : ` it-op-${step.op}`}`;
  itOpEl.textContent = IT_OP[step.op];
  itTtEl.textContent = step.title;
  itDsEl.innerHTML = step.desc;

  itCodeEl.querySelectorAll('.ln').forEach(el => {
    el.classList.toggle('hl', step.lines.indexOf(Number(el.dataset.line)) >= 0);
  });

  itSlider.value = idx;
  itCounterEl.textContent = `第 ${idx} / ${IT_STEPS.length - 1} 步`;
  document.getElementById('it-btn-prev').disabled = idx === 0;
  document.getElementById('it-btn-next').disabled = idx === IT_STEPS.length - 1;
}

itSlider.max = IT_STEPS.length - 1;
itSlider.addEventListener('input', e => { itStopPlay(); itRender(Number(e.target.value)); });
document.getElementById('it-btn-prev').addEventListener('click', () => { itStopPlay(); itRender(itCur - 1); });
document.getElementById('it-btn-next').addEventListener('click', () => { itStopPlay(); itRender(itCur + 1); });
document.getElementById('it-btn-reset').addEventListener('click', () => { itStopPlay(); itRender(0); });
itPlayBtn.addEventListener('click', () => {
  if (itTimer) { itStopPlay(); return; }
  if (itCur === IT_STEPS.length - 1) itRender(0);
  itPlayBtn.textContent = '暂停';
  itTimer = setInterval(() => {
    if (itCur >= IT_STEPS.length - 1) { itStopPlay(); return; }
    itRender(itCur + 1);
  }, 1500);
});

document.addEventListener('keydown', e => {
  if (e.target.closest('input,textarea,select,[contenteditable]')) return;
  if (e.key === 'ArrowRight') { itStopPlay(); itRender(itCur + 1); e.preventDefault(); }
  if (e.key === 'ArrowLeft') { itStopPlay(); itRender(itCur - 1); e.preventDefault(); }
});

itRender(0);
