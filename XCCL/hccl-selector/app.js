(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  let reduced = false;
  $('reduce-button').addEventListener('click', () => {
    reduced = !reduced;
    $('ranks').classList.toggle('reduced', reduced);
    document.querySelectorAll('.rank strong').forEach((node, i) => { node.textContent = reduced ? '10' : String(i + 1); });
    document.querySelectorAll('.rank-plus').forEach((node) => { node.textContent = reduced ? '=' : '+'; });
    $('rank-description').textContent = reduced ? '1 + 2 + 3 + 4 = 10，每张卡都获得同一个总和。' : '输入各不相同，目标是让4张卡都拿到10。';
    $('reduce-button').textContent = reduced ? '重置输入 ↺' : '执行一次求和 →';
  });

  const steps = [
    ['先确认允许的执行引擎', '部分PCIe混合场景会要求切换到AIV_ONLY，因此进入Run前的配置不一定是最终配置。'],
    ['初始化通信域的Tuner', '用独立的tuner_init上下文标记初始化状态。此图假设插件已加载；初始化完成后，同一通信域后续调用跳过这一步。'],
    ['按当前执行配置查询模型缓存', '模型标签由costmodel和执行配置组成。它标识一份模型，该模型可能包含多个候选引擎的算法。'],
    ['构建并保存可复用的模型', '遍历算法目录，检查拓扑，调用执行器计算系数，应用拓扑优先规则。保存模型后，再按候选引擎与算法配置过滤。'],
    ['取得CostModel', '拿到的是算法与系数数组，并不是这次调用最终使用的算法。模型命中缓存时，也仍需生成当前代价表。'],
    ['为当前调用生成代价表', '检查算子、数据类型、PROD、内存重叠与保序条件；代入当前数据量计算cost，再应用算子优先保留规则。'],
    ['取得CostTable', '表中存放当前候选的算法名和cost。通过调用条件只是取得比较资格，不表示已经被选中。'],
    ['可选地让Tuner调整cost', '表非空且插件已加载时，补充算法维度，再传给Tuner。插件可将候选cost设为负值禁用、0偏好或正值覆盖。'],
    ['选择最小有效cost，输出算法', '跳过名字为空或cost为负的候选。选择最小值，写入algName并更新opExecuteConfig；没有有效候选则返回不支持。']
  ];
  let mode = 'cold';
  let cursor = 0;
  function route() { return mode === 'cold' ? [0, 1, 2, 3, 4, 5, 6, 7, 8] : [0, 2, 4, 5, 6, 7, 8]; }
  function renderSequence() {
    const currentRoute = route();
    const current = currentRoute[cursor];
    document.querySelectorAll('[data-seq]').forEach((node) => {
      const step = Number(node.dataset.seq);
      node.classList.toggle('current', step === current);
      node.classList.toggle('done', currentRoute.indexOf(step) >= 0 && currentRoute.indexOf(step) < cursor);
      node.classList.toggle('skipped', !currentRoute.includes(step));
    });
    $('step-counter').textContent = `STEP ${String(cursor + 1).padStart(2, '0')} / ${String(currentRoute.length).padStart(2, '0')}`;
    $('step-title').textContent = steps[current][0];
    $('step-description').textContent = steps[current][1];
    $('seq-prev').disabled = cursor === 0;
    $('seq-next').textContent = cursor === currentRoute.length - 1 ? '重新开始 ↺' : '下一步 →';
    document.querySelectorAll('[data-cache]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.cache === mode)));
  }
  document.querySelectorAll('[data-cache]').forEach((button) => button.addEventListener('click', () => { mode = button.dataset.cache; cursor = 0; renderSequence(); }));
  $('seq-prev').addEventListener('click', () => { if (cursor > 0) cursor--; renderSequence(); });
  $('seq-next').addEventListener('click', () => { cursor = (cursor + 1) % route().length; renderSequence(); });

  const engineSets = {
    CCU_MS: ['CCU_MS', 'CCU_SCHED', 'AICPU_TS', 'HOSTCPU'],
    CCU_SCHED: ['CCU_SCHED', 'AICPU_TS', 'HOSTCPU'],
    AICPU_TS: ['AICPU_TS', 'HOSTCPU'],
    AIV: ['AIV', 'AICPU_TS', 'HOSTCPU'],
    AIV_ONLY: ['AIV'],
    HOSTCPU: ['HOSTCPU']
  };
  const candidates = [
    {name: '候选A', engine: 'CCU_MS', cost: 42, attrs: true},
    {name: '候选B', engine: 'AICPU_TS', cost: 28, attrs: true},
    {name: '候选C', engine: 'AIV', cost: 18, attrs: false}
  ];
  function renderCandidates() {
    const allowed = engineSets[$('engine-config').value];
    const tuner = $('tuner-mode').value;
    $('engine-allowed').textContent = `允许的引擎：${allowed.join(' · ')}`;
    const result = candidates.map((candidate) => {
      const engine = allowed.includes(candidate.engine);
      const inTable = engine && candidate.attrs;
      let cost = candidate.cost;
      if (inTable && tuner === 'prefer' && candidate.name === '候选A') cost = 0;
      if (inTable && tuner === 'disable') cost = -1;
      return {...candidate, inTable, engineAllowed: engine, finalCost: cost, valid: inTable && cost >= 0};
    });
    const valid = result.filter((candidate) => candidate.valid);
    const winner = valid.reduce((best, candidate) => !best || candidate.finalCost < best.finalCost ? candidate : best, null);
    $('candidate-list').replaceChildren();
    result.forEach((candidate) => {
      const row = document.createElement('div');
      row.className = `candidate-row ${!candidate.valid ? 'filtered' : ''} ${winner === candidate ? 'winner' : ''}`;
      const name = document.createElement('div');
      const strong = document.createElement('strong'); strong.textContent = candidate.name;
      const small = document.createElement('small'); small.textContent = candidate.engine;
      name.append(strong, small);
      const attrs = document.createElement('span'); attrs.textContent = candidate.attrs ? '通过' : '不通过';
      const cost = document.createElement('span'); cost.textContent = candidate.inTable ? (candidate.finalCost < 0 ? '-1（禁用）' : `${candidate.finalCost}μs`) : '未入表';
      const status = document.createElement('span'); status.className = 'candidate-status';
      status.textContent = !candidate.engineAllowed ? '引擎被排除' : !candidate.attrs ? '属性被排除' : !candidate.valid ? 'Tuner禁用' : winner === candidate ? '✓ 最终选中' : 'cost更大';
      row.append(name, attrs, cost, status); $('candidate-list').append(row);
    });
    $('selected-name').textContent = winner ? `${winner.name} · ${winner.engine}` : 'HCCL_E_NOT_SUPPORT';
    $('selected-reason').textContent = winner ? `最小有效cost为${winner.finalCost}μs，输出该候选，并将执行配置更新为${winner.engine}。` : '这组教学候选中没有有效算法。新Selector返回不支持，未进入旧选择器。';
    if (tuner === 'prefer' && !result[0].inTable) $('selected-reason').textContent += ' 候选A未进入代价表，Tuner无法通过修改cost把它恢复。';
    $('selected-name').parentElement.classList.toggle('error', !winner);
  }
  $('engine-config').addEventListener('change', renderCandidates);
  $('tuner-mode').addEventListener('change', renderCandidates);

  const meshTable = [[1,.7135],[2,.7758],[4,.8112],[8,.8301],[16,.84],[32,.8449],[64,.8475],[128,.8487],[256,.8494]];
  function sizeLabel(bytes) { return bytes >= 1048576 ? `${bytes / 1048576}MiB` : `${bytes / 1024}KiB`; }
  function updateCost() {
    const size = 4096 * (2 ** Number($('data-size').value));
    const transfer = size / 4;
    const util = (meshTable.find(([upper]) => transfer <= upper * 1048576) || meshTable[meshTable.length - 1])[1];
    const network = (1 / 56e9) / util * size * 1e6;
    const local = (1 / 750e9 + 3 / 483e9) * size * 1e6;
    const fixed = 2;
    const launch = 15;
    const segment = network + local + fixed;
    const total = Math.max(segment, launch);
    const maximum = total * 1.04;
    $('size-label').textContent = sizeLabel(size);
    $('element-count').textContent = (size / 4).toLocaleString('en-US');
    $('transfer-size').textContent = sizeLabel(transfer);
    $('util-value').textContent = util.toFixed(4);
    $('segment-value').textContent = `${segment.toFixed(3)}μs`;
    $('network-value').textContent = `${network.toFixed(3)}μs`;
    $('local-value').textContent = `${local.toFixed(3)}μs`;
    $('bar-network').style.width = `${network / maximum * 100}%`;
    $('bar-local').style.width = `${local / maximum * 100}%`;
    $('bar-fixed').style.width = `${fixed / maximum * 100}%`;
    $('bar-launch').style.width = `${launch / maximum * 100}%`;
    $('stacked-cost').setAttribute('aria-label', `跨卡${network.toFixed(3)}微秒，本地${local.toFixed(3)}微秒，固定2微秒`);
    $('cost-result').textContent = `${total.toFixed(3)}μs`;
    $('cost-dominant').textContent = segment >= launch ? '通信侧代价更大，最终取segCost。' : '展开侧代价更大，最终取D = 15μs。';
    $('data-size').setAttribute('aria-valuetext', sizeLabel(size));
  }
  $('data-size').addEventListener('input', updateCost);

  const snippets = [
    {file:'ins_v2_all_reduce_sole_executor.cc', path:'src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_sole_executor.cc', title:'把一个名字连接到一个可创建的执行器', text:'注册宏同时登记执行器工厂和算法目录。执行器的两个类型参数，分别是拓扑匹配器和通信模板。', code:`REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE,
    AicpuAllReduceSoleMeshOneShot,
    InsV2AllReduceSoleExecutor,
    TopoMatchOneLevel,
    InsTempAllReduceMesh1DOneShot);`},
    {file:'ins_v2_all_reduce_sole_executor.cc', path:'src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_sole_executor.cc', title:'声明算法能在哪些拓扑上运行', text:'这是属性注册的节选。最大层数为1、允许两种第0层形状，还设置了Mesh连通要求和自定义回调；未写的字段继续使用默认值。完整限制见原文第11节。', code:`// 属性注册节选，省略了后续自定义拓扑回调。
topo.maxTopoLevelNum = 1;
topo.supportLevel0Topos =
    LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
topo.isSupportLevel0PcieMix = true;
topo.requireAllMeshConnected = true;`},
    {file:'ins_temp_all_reduce_mesh_1D_one_shot.cc', path:'src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc', title:'从拓扑参数生成A、B、C、D', text:'真实模板先拒绝超过8个rank的模型输入，再计算跨卡、本地、固定时延与下发系数。执行器传dataRatio=1/rankSize，因此本例n=1。此处只摘录关键语句。', code:`if (param.rankSize > 8) {
    return {};
}
float n = param.dataRatio * param.rankSize;
// ……调用公共建模函数得到A、B1、B2、C、D。
B = B1 + (param.rankSize - 1) * B2;
params.push_back({A, B, C, D});`},
    {file:'selector_engine.cc', path:'src/ops/op_common/selector/selector_engine.cc', title:'输出名字，也更新执行配置', text:'这些语句来自SelectMinCost的不同位置，省略日志和并列记录。第一个有效候选通过minIdx==-1进入；只有遇到更小值才替换，因此并列保留表中第一个。', code:`if (name == nullptr || cost < 0.0f) {
    continue;
}
if (minIdx == -1 || cost < minCost) {
    minIdx = i;
    minCost = cost;
}
// ……循环完成并检查结果后：
algName = ct.costs[minIdx].algName;
param.opExecuteConfig = GetEngineByAlgName(algName);`}
  ];
  let codeIndex = 0;
  function setCode(index, focus = false) {
    codeIndex = index;
    const snippet = snippets[index];
    $('code-content').textContent = snippet.code;
    $('code-file').textContent = snippet.file;
    $('code-title').textContent = snippet.title;
    $('code-description').textContent = snippet.text;
    $('code-locator').textContent = `源码：${snippet.path}`;
    $('code-panel').setAttribute('aria-labelledby', `code-tab-${index}`);
    $('copy-code').textContent = '复制代码';
    document.querySelectorAll('[data-code]').forEach((button) => {
      const selected = Number(button.dataset.code) === index;
      button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
      if (selected && focus) button.focus();
    });
  }
  document.querySelectorAll('[data-code]').forEach((button) => {
    button.addEventListener('click', () => setCode(Number(button.dataset.code)));
    button.addEventListener('keydown', (event) => {
      if (!['ArrowRight','ArrowLeft','Home','End'].includes(event.key)) return;
      event.preventDefault();
      const index = event.key === 'Home' ? 0 : event.key === 'End' ? 3 : (codeIndex + (event.key === 'ArrowRight' ? 1 : 3)) % 4;
      setCode(index, true);
    });
  });
  $('copy-code').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(snippets[codeIndex].code); $('copy-code').textContent = '已复制 ✓'; }
    catch {
      const selection = window.getSelection();
      const range = document.createRange(); range.selectNodeContents($('code-content'));
      selection.removeAllRanges(); selection.addRange(range);
      $('copy-code').textContent = '已选中，请复制';
    }
  });
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        document.querySelectorAll('.sidebar nav a, .mobile-nav a').forEach((link) => {
          const active = link.hash === `#${entry.target.id}`;
          link.classList.toggle('active', active);
          if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
        });
      });
    }, {rootMargin: '-12% 0px -70% 0px'});
    document.querySelectorAll('.chapter').forEach((section) => observer.observe(section));
  }
  renderSequence(); renderCandidates(); updateCost();
})();
