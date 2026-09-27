(function () {
  var selM = document.getElementById('tf-m');
  var selN = document.getElementById('tf-n');
  var rho  = document.getElementById('tf-rho');
  if (!selM || !selN || !rho) { return; }

  var rhoOut   = document.getElementById('tf-rho-out');
  var barMesh  = document.getElementById('tf-bar-mesh');
  var barNhr   = document.getElementById('tf-bar-nhr');
  var txtMesh  = document.getElementById('tf-mesh');
  var txtNhr   = document.getElementById('tf-nhr');
  var numP     = document.getElementById('tf-p');
  var numMesh  = document.getElementById('tf-mesh-num');
  var numNhr   = document.getElementById('tf-nhr-num');
  var lineMesh = document.getElementById('tf-line-mesh');
  var lineNhr  = document.getElementById('tf-line-nhr');
  var cross    = document.getElementById('tf-cross');
  var crossG   = document.getElementById('tf-cross-g');
  var crossGd  = document.getElementById('tf-cross-guide');
  var crossLb  = document.getElementById('tf-cross-label');
  var verdict  = document.getElementById('tf-verdict');

  // 堆叠条几何
  var BAR_X = 120, BAR_W = 480;
  // 折线图几何
  var PL_X = 70, PL_Y = 30, PL_W = 550, PL_H = 160;

  function vMesh(M, N, r) { return (M - 1) * (1 + (N - 1) * r); }
  function vNhr(M, N, r)  { return (N - 1) * (M - (M - 1) * r); }

  function fmt(v) { return v.toFixed(2) + ' d'; }

  function draw() {
    var M = parseInt(selM.value, 10);
    var N = parseInt(selN.value, 10);
    var r = parseFloat(rho.value);
    var total = M * N - 1;

    var vm = vMesh(M, N, r);
    var vn = vNhr(M, N, r);

    // 数值与堆叠条
    rhoOut.textContent = r.toFixed(2);
    numP.textContent = String(M * N);
    numMesh.textContent = fmt(vm);
    numNhr.textContent = fmt(vn);
    txtMesh.textContent = fmt(vm);
    txtNhr.textContent = fmt(vn);

    var wm = Math.max(0, Math.round(BAR_W * vm / total));
    var wn = Math.max(0, Math.round(BAR_W * vn / total));
    barMesh.setAttribute('width', wm);
    barNhr.setAttribute('width', wn);

    // 两条曲线
    var ptsM = [], ptsN = [], i, rr, x, y;
    for (i = 0; i <= 40; i++) {
      rr = i / 40;
      x = PL_X + PL_W * rr;
      y = PL_Y + PL_H * (1 - vMesh(M, N, rr) / total);
      ptsM.push(x.toFixed(1) + ',' + y.toFixed(1));
      y = PL_Y + PL_H * (1 - vNhr(M, N, rr) / total);
      ptsN.push(x.toFixed(1) + ',' + y.toFixed(1));
    }
    lineMesh.setAttribute('points', ptsM.join(' '));
    lineNhr.setAttribute('points', ptsN.join(' '));

    // 均衡点 rho* = (MN - 2M + 1) / (2(M-1)(N-1))
    var rStar = (M * N - 2 * M + 1) / (2 * (M - 1) * (N - 1));
    if (rStar < 0) { rStar = 0; }
    if (rStar > 1) { rStar = 1; }
    var cx = PL_X + PL_W * rStar;
    var cy = PL_Y + PL_H * (1 - vMesh(M, N, rStar) / total);
    cross.setAttribute('cx', cx.toFixed(1));
    cross.setAttribute('cy', cy.toFixed(1));
    crossGd.setAttribute('d', 'M' + cx.toFixed(1) + ' ' + cy.toFixed(1) + ' V' + (PL_Y + PL_H));
    crossLb.setAttribute('x', Math.max(PL_X + 62, Math.min(cx, PL_X + PL_W - 62)).toFixed(1));
    crossLb.setAttribute('y', (cy - 16 > PL_Y + 12 ? cy - 15 : cy + 25).toFixed(1));
    crossLb.textContent = '均衡点 ρ* = ' + rStar.toFixed(2);

    // 文字结论
    var diff = vm - vn;
    var html;
    if (Math.abs(diff) < 0.005) {
      html = '<b>ρ = ' + r.toFixed(2) + '：</b>两条链路的发送量几乎相等，正好落在均衡点上。';
    } else if (diff > 0) {
      html = '<b>ρ = ' + r.toFixed(2) + '：</b>Mesh 比 NHR 多背 ' + diff.toFixed(2) +
             ' d。把 ρ 调小可以让流量往 NHR 那边挪。';
    } else {
      html = '<b>ρ = ' + r.toFixed(2) + '：</b>NHR 比 Mesh 多背 ' + (-diff).toFixed(2) +
             ' d。把 ρ 调大可以让流量往 Mesh 那边挪。';
    }
    html += ' 两条合计恒为 <b>(P − 1)·d = ' + total + ' d</b>。';
    verdict.innerHTML = html;
  }

  selM.addEventListener('change', draw);
  selN.addEventListener('change', draw);
  rho.addEventListener('input', draw);
  draw();
})();

(function () {
  // 侧栏与移动端导航的当前章节高亮
  var links = Array.prototype.slice.call(document.querySelectorAll('.sidebar nav a, .mobile-nav a'));
  var sections = links.map(function (a) {
    return document.querySelector(a.getAttribute('href'));
  });
  if (!sections.length || !('IntersectionObserver' in window)) { return; }

  function activate(id) {
    links.forEach(function (a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + id);
    });
  }

  var io = new IntersectionObserver(function (entries) {
    var best = null;
    entries.forEach(function (e) {
      if (e.isIntersecting && (!best || e.intersectionRatio > best.intersectionRatio)) { best = e; }
    });
    if (best) { activate(best.target.id); }
  }, { rootMargin: '-100px 0px -70% 0px', threshold: [0, 0.2, 0.5] });

  sections.forEach(function (s) { if (s) { io.observe(s); } });
})();
