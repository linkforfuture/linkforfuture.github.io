/* ================= 数据 ================= */
const NUMS = [7, 2, 4, 6, 3, 5, 1, 8];
const K = 3;

/* ================= 生成轨迹 ================= */
function buildTrace(nums, k) {
  const steps = [];
  const dq = [];
  const out = [];

  const snap = (o) => steps.push(Object.assign({
    i: null, dq: dq.slice(), out: out.slice(),
    phase: 'init', title: '', desc: '', lines: [], fresh: -1
  }, o));

  snap({
    phase: 'init', lines: [2, 3],
    title: '初始状态：队列空，结果空',
    desc: '接下来每次处理一个 <code>nums[i]</code>，都走同样四步：淘汰队尾 → 自己入队 → 淘汰过期队首 → 记录答案。'
  });

  for (let i = 0; i < nums.length; i++) {
    const winStart = Math.max(0, i - k + 1);
    const base = { i, winStart };

    // ---- 步骤① 队尾淘汰 ----
    const killed = [];
    while (dq.length && nums[dq[dq.length - 1]] <= nums[i]) {
      const j = dq.pop();
      killed.push(j);
      snap(Object.assign({}, base, {
        phase: 'popback', lines: [5, 6], killed: killed.slice(),
        title: `弹出队尾下标 ${j}（值 ${nums[j]}）`,
        desc: `因为 <code>nums[${j}] = ${nums[j]} &le; nums[${i}] = ${nums[i]}</code>：下标 ${j} 比 ${i} 旧，值还更小，`
            + `等 ${i} 都滑出窗口了，${j} 早就不在了。它永远没机会成为最大值，直接扔。`
      }));
    }

    // ---- 步骤② 入队 ----
    dq.push(i);
    snap(Object.assign({}, base, {
      phase: 'pushback', lines: [8],
      title: `下标 ${i}（值 ${nums[i]}）从队尾入队`,
      desc: (killed.length
              ? `刚才清掉了 ${killed.length} 个没前途的（下标 ${killed.join('、')}）。`
              : `队尾值 ${nums[dq[dq.length - 2]]} 比 ${nums[i]} 大，所以没淘汰任何人。`)
            + `现在队列是 <code>[${dq.join(', ')}]</code>，对应值 <code>[${dq.map(x => nums[x]).join(', ')}]</code>，从队首到队尾<b>递减</b>。`
    }));

    // ---- 步骤③ 队首过期 ----
    if (dq[0] <= i - k) {
      const j = dq.shift();
      snap(Object.assign({}, base, {
        phase: 'popfront', lines: [9, 10],
        title: `队首下标 ${j} 滑出窗口，弹出`,
        desc: `窗口是 <code>[${winStart}, ${i}]</code>，而下标 <code>${j} &le; i - k = ${i - k}</code>，`
            + `它已经跑到左边界外面了。它是队首（最大的），一直挡在前面，所以必须先把它踢掉。`
      }));
    }

    // ---- 步骤④ 记录答案 ----
    if (i >= k - 1) {
      out.push(nums[dq[0]]);
      snap(Object.assign({}, base, {
        phase: 'record', lines: [12, 13], fresh: out.length - 1,
        title: `窗口形成，记录最大值 ${nums[dq[0]]}`,
        desc: `窗口 <code>[${winStart}, ${i}]</code> = <code>[${nums.slice(winStart, i + 1).join(', ')}]</code>。`
            + `队首下标 ${dq[0]} 的值是 ${nums[dq[0]]}，因为队列递减，它就是这 k 个数里最大的。`
      }));
    }
  }

  snap({
    phase: 'done', lines: [16], dq: dq.slice(), out: out.slice(),
    title: '结束，返回 result',
    desc: `结果 <code>[${out.join(', ')}]</code>。整个过程中每个下标最多进出队各一次，所以总复杂度是 <b>O(n)</b>。`
  });

  return steps;
}

const STEPS = buildTrace(NUMS, K);

/* ================= 代码渲染 ================= */
const CODE = [
  ['vector&lt;int&gt; maxSlidingWindow(vector&lt;int&gt;&amp; nums, int k) {', ''],
  ['    std::deque&lt;int&gt; dq;            <span class="c">// 存下标，对应值严格递减</span>', ''],
  ['    std::vector&lt;int&gt; result;', ''],
  ['    for (int i = 0; i &lt; nums.size(); ++i) {', ''],
  ['        <span class="c">// 1. 队尾比 nums[i] 小的都没机会成为最大值，弹出</span>', ''],
  ['        while (!dq.empty() &amp;&amp; nums[dq.back()] &lt;= nums[i])', ''],
  ['            dq.pop_back();', ''],
  ['        dq.push_back(i);', ''],
  ['        <span class="c">// 2. 队首滑出窗口 [i-k+1, i]，弹出</span>', ''],
  ['        if (dq.front() &lt;= i - k)', ''],
  ['            dq.pop_front();', ''],
  ['        <span class="c">// 3. 窗口形成后，队首即最大值</span>', ''],
  ['        if (i &gt;= k - 1)', ''],
  ['            result.push_back(nums[dq.front()]);', ''],
  ['    }', ''],
  ['    return result;', ''],
  ['}', '']
];

const codeEl = document.getElementById('code');
codeEl.innerHTML = CODE.map((line, idx) =>
  `<span class="ln" data-line="${idx + 1}"><span class="num">${idx + 1}</span>${line[0] || '&nbsp;'}</span>`
).join('');

/* ================= 渲染 ================= */
const arrEl = document.getElementById('arr');
const dqEl = document.getElementById('dq');
const outEl = document.getElementById('out');
const opEl = document.getElementById('op');
const ttEl = document.getElementById('tt');
const dsEl = document.getElementById('ds');
const counterEl = document.getElementById('counter');
const slider = document.getElementById('slider');

const OPCLASS = {
  init: 'init', popback: 'popback', pushback: 'pushback',
  popfront: 'popfront', record: 'record', done: 'done'
};
const OPTEXT = {
  init: '准备', popback: 'pop_back', pushback: 'push_back',
  popfront: 'pop_front', record: '记录', done: '完成'
};

/** 每一格对应「进入这一步之前」的状态，用来展示被弹出的元素去哪了 */
function renderArray(step) {
  const dqSet = new Set(step.dq);
  const killedSet = new Set(step.killed || []);
  const frontIdx = step.dq.length ? step.dq[0] : -1;
  arrEl.innerHTML = NUMS.map((v, idx) => {
    const cls = ['cell'];
    if (step.winStart !== null && step.i !== null && idx >= step.winStart && idx <= step.i) cls.push('inwin');
    if (idx === step.i) cls.push('cur');
    if (dqSet.has(idx)) cls.push('indq');
    if (idx === frontIdx) cls.push('front');

    // 本步中被弹掉的元素，标一个「已淘汰」
    const flag = killedSet.has(idx) ? '<span class="flag gone">淘汰</span>' : '';

    return `<div class="${cls.join(' ')}">${flag}<div class="v">${v}</div><div class="i">${idx}</div></div>`;
  }).join('');
}

function renderDq(step) {
  if (!step.dq.length) {
    dqEl.innerHTML = '<span class="dqempty">[ 空 ]</span>';
    return;
  }
  dqEl.innerHTML = step.dq.map((idx, pos) =>
    `<div class="dqbox">${pos === 0 ? '<div class="head">队首·最大</div>' : ''}`
    + `<div class="v">${NUMS[idx]}</div><div class="i">idx ${idx}</div></div>`
  ).join('');
}

function renderOut(step) {
  if (!step.out.length) {
    outEl.innerHTML = '<span class="dqempty">[ 空 ]</span>';
    return;
  }
  outEl.innerHTML = step.out.map((v, idx) =>
    `<div class="outbox${idx === step.fresh ? ' fresh' : ''}">${v}</div>`
  ).join('');
}

let curStep = 0;

function render(idx) {
  idx = Math.max(0, Math.min(STEPS.length - 1, idx));
  curStep = idx;
  const step = STEPS[idx];

  renderArray(step);
  renderDq(step);
  renderOut(step);

  opEl.className = 'op ' + (OPCLASS[step.phase] || 'init');
  opEl.textContent = OPTEXT[step.phase] || '步骤';
  ttEl.textContent = (step.i !== null ? `i = ${step.i} · ` : '') + step.title;
  dsEl.innerHTML = step.desc;

  document.querySelectorAll('pre.code .ln').forEach(el => {
    el.classList.toggle('hl', step.lines.includes(Number(el.dataset.line)));
  });

  slider.value = idx;
  counterEl.textContent = `${idx} / ${STEPS.length - 1}`;
  document.getElementById('btnPrev').disabled = idx === 0;
  document.getElementById('btnNext').disabled = idx === STEPS.length - 1;
}

/* ================= 交互 ================= */
slider.max = STEPS.length - 1;
slider.addEventListener('input', e => { stopPlay(); render(Number(e.target.value)); });
document.getElementById('btnPrev').addEventListener('click', () => { stopPlay(); render(curStep - 1); });
document.getElementById('btnNext').addEventListener('click', () => { stopPlay(); render(curStep + 1); });
document.getElementById('btnReset').addEventListener('click', () => { stopPlay(); render(0); });

let timer = null;
const playBtn = document.getElementById('btnPlay');
function stopPlay() {
  if (timer) { clearInterval(timer); timer = null; }
  playBtn.textContent = '▶ 自动播放';
}
playBtn.addEventListener('click', () => {
  if (timer) { stopPlay(); return; }
  if (curStep === STEPS.length - 1) render(0);
  playBtn.textContent = '⏸ 暂停';
  timer = setInterval(() => {
    if (curStep >= STEPS.length - 1) { stopPlay(); return; }
    render(curStep + 1);
  }, 1100);
});

document.addEventListener('keydown', e => {
  if (e.target.closest('input,textarea,select,[contenteditable]')) return;
  if (e.key === 'ArrowRight') { stopPlay(); render(curStep + 1); e.preventDefault(); }
  if (e.key === 'ArrowLeft')  { stopPlay(); render(curStep - 1); e.preventDefault(); }
});

render(0);

/* ================= 题目分类筛选 ================= */
const filtersEl = document.getElementById('filters');
const probRows = [...document.querySelectorAll('#probTable tr[data-cat]')];
if (filtersEl) {
  filtersEl.addEventListener('click', e => {
    const btn = e.target.closest('button');
    if (!btn) return;
    [...filtersEl.querySelectorAll('button')].forEach(b => b.classList.toggle('on', b === btn));
    const f = btn.dataset.f;
    probRows.forEach(r => r.classList.toggle('hide', f !== 'all' && r.dataset.cat !== f));
  });
}
