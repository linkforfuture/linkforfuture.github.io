(function () {
  'use strict';

  var PL_X = 70, PL_W = 540, PL_Y = 30, PL_H = 210;

  var selM = document.getElementById('sel-m');
  var selN = document.getElementById('sel-n');
  var selAlg = document.getElementById('sel-alg');
  var rng = document.getElementById('rng-rho');
  var outRho = document.getElementById('out-rho');
  var lineMesh = document.getElementById('tf-mesh');
  var lineNhr = document.getElementById('tf-nhr');
  var cross = document.getElementById('tf-cross');
  var crossGd = document.getElementById('tf-cross-gd');
  var crossLb = document.getElementById('tf-cross-lb');
  var tk1 = document.getElementById('tf-t1'), tk2 = document.getElementById('tf-t2');
  var tk1Lb = document.getElementById('tf-t1-lb'), tk2Lb = document.getElementById('tf-t2-lb');
  var lbMesh = document.getElementById('lb-mesh'), lbNhr = document.getElementById('lb-nhr'), lbSum = document.getElementById('lb-sum');
  var valMesh = document.getElementById('val-mesh'), valNhr = document.getElementById('val-nhr'), valSum = document.getElementById('val-sum');
  var verdict = document.getElementById('verdict');
  var pill = document.getElementById('rs-pill');

  if (!selM || !rng) { return; }

  // ---- 流量模型（均匀分片，单位：RS 为 d 字节，AR 为 S 字节）----
  function vMeshRS(M, N, r) { return (M - 1) * (N * r + (1 - r)); }
  function vNhrRS(M, N, r) { return (N - 1) * (M * (1 - r) + r); }
  function vMeshAR(M, N, r) { return 2 * (M - 1) * (r / M + (1 - r) / (M * N)); }
  function vNhrAR(M, N, r) { return 2 * (N - 1) * ((1 - r) / N + r / (M * N)); }
  function vMesh(M, N, r, alg) { return alg === 'ar' ? vMeshAR(M, N, r) : vMeshRS(M, N, r); }
  function vNhr(M, N, r, alg) { return alg === 'ar' ? vNhrAR(M, N, r) : vNhrRS(M, N, r); }

  function xOf(r) { return PL_X + PL_W * r; }
  function yOf(frac) { return PL_Y + PL_H * (1 - frac); }
  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function curve(fn, M, N, alg, other) {
    var pts = [];
    for (var k = 0; k <= 100; k++) {
      var r = k / 100;
      var v = fn(M, N, r, alg);
      var tot = v + other(M, N, r, alg);
      var frac = tot > 0 ? v / tot : 0;
      pts.push(xOf(r).toFixed(1) + ',' + yOf(frac).toFixed(1));
    }
    return pts.join(' ');
  }

  function fmt(x) {
    if (!isFinite(x)) { return '—'; }
    if (x >= 1000) { return x.toFixed(0); }
    if (x >= 100) { return x.toFixed(1); }
    return x.toFixed(2);
  }

  function placeTick(el, elLb, r) {
    var x = xOf(clamp(r, 0, 1)).toFixed(1);
    el.setAttribute('x1', x); el.setAttribute('x2', x);
    elLb.setAttribute('x', x);
  }

  function render() {
    var M = parseInt(selM.value, 10);
    var N = parseInt(selN.value, 10);
    var alg = selAlg.value;
    var r = parseFloat(rng.value);
    var unit = alg === 'ar' ? '× S' : '× d';

    outRho.textContent = r.toFixed(2);
    pill.textContent = alg === 'ar'
      ? 'AllReduce · M=' + M + ', N=' + N + '，S = P·d'
      : 'ReduceScatter · M=' + M + ', N=' + N + '，d = 每 rank 输出';

    lineMesh.setAttribute('points', curve(vMesh, M, N, alg, vNhr));
    lineNhr.setAttribute('points', curve(vNhr, M, N, alg, vMesh));

    // 总量平衡点：两侧总量相等（三个算子公式一致）
    var rStar = (M * N - 2 * M + 1) / (2 * (M - 1) * (N - 1));
    rStar = clamp(rStar, 0, 1);
    var cx = xOf(rStar), cy = yOf(0.5);
    cross.setAttribute('cx', cx.toFixed(1));
    cross.setAttribute('cy', cy.toFixed(1));
    crossGd.setAttribute('x1', cx.toFixed(1));
    crossGd.setAttribute('x2', cx.toFixed(1));
    crossLb.setAttribute('x', clamp(cx, PL_X + 56, PL_X + PL_W - 56).toFixed(1));
    crossLb.setAttribute('y', (cy - 16 > PL_Y + 8 ? cy - 15 : cy + 24).toFixed(1));
    crossLb.textContent = '均衡点 ρ* = ' + rStar.toFixed(2);

    // 分阶段平衡点
    var r1 = M * (N - 1) / (N * (M - 1) + M * (N - 1));
    var r2 = (M - 1) / ((M - 1) + (N - 1));
    placeTick(tk1, tk1Lb, r1);
    placeTick(tk2, tk2Lb, r2);
    tk1Lb.textContent = 'ρ₁=' + r1.toFixed(2);
    tk2Lb.textContent = 'ρ₂=' + r2.toFixed(2);
    tk1Lb.setAttribute('y', '258');
    tk2Lb.setAttribute('y', '272');

    var vm = vMesh(M, N, r, alg), vn = vNhr(M, N, r, alg), vt = vm + vn;
    lbMesh.textContent = 'Mesh 发送量（' + unit + '）';
    lbNhr.textContent = 'NHR 发送量（' + unit + '）';
    lbSum.textContent = alg === 'ar' ? '总量 = 2(P−1)/P · S' : '总量 = (P−1) · d';
    valMesh.textContent = fmt(vm) + ' ' + unit;
    valNhr.textContent = fmt(vn) + ' ' + unit;
    valSum.textContent = fmt(vt) + ' ' + unit;

    var fMesh = vt > 0 ? vm / vt : 0;
    var who = Math.abs(fMesh - 0.5) < 0.02
      ? '两个维度几乎均分流量。'
      : (fMesh > 0.5
        ? 'Mesh 承担了 <b>' + (fMesh * 100).toFixed(0) + '%</b> 的流量，NHR 只占 ' + ((1 - fMesh) * 100).toFixed(0) + '%。'
        : 'NHR 承担了 <b>' + ((1 - fMesh) * 100).toFixed(0) + '%</b> 的流量，Mesh 只占 ' + (fMesh * 100).toFixed(0) + '%。');

    verdict.innerHTML = '当前 ρ = ' + r.toFixed(2) + '：' + who +
      ' 总量始终是 <b>' + fmt(vt) + ' ' + unit + '</b>，与 ρ 无关 —— ' +
      'ρ 只是在两个维度之间重新分配。总量平衡点在 ρ* = ' + rStar.toFixed(2) +
      '，而两个阶段各自的配平点落在 ρ₁ = ' + r1.toFixed(2) + ' 与 ρ₂ = ' + r2.toFixed(2) +
      '（M = N 时三者才会重合到 0.5）。';
  }

  selM.addEventListener('change', render);
  selN.addEventListener('change', render);
  selAlg.addEventListener('change', render);
  rng.addEventListener('input', render);
  render();
})();

(function () {
  'use strict';
  var links = Array.prototype.slice.call(document.querySelectorAll('.sidebar nav a, .mobile-nav a'));
  var targets = [];
  links.forEach(function (a) {
    var id = (a.getAttribute('href') || '').replace('#', '');
    var el = id && document.getElementById(id);
    if (el && targets.indexOf(el) === -1) { targets.push(el); }
  });
  if (!targets.length || !('IntersectionObserver' in window)) { return; }

  function mark(id) {
    links.forEach(function (a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + id);
    });
  }
  var seen = {};
  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { seen[e.target.id] = e.isIntersecting ? e.intersectionRatio : 0; });
    var best = null, bestV = 0;
    targets.forEach(function (t) {
      var v = seen[t.id] || 0;
      if (v > bestV) { bestV = v; best = t.id; }
    });
    if (best) { mark(best); }
  }, { rootMargin: '-90px 0px -55% 0px', threshold: [0, 0.15, 0.4, 0.75, 1] });

  targets.forEach(function (t) { obs.observe(t); });
})();
