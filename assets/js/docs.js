/* Shared Docsify configuration. Loaded before docsify.min.js on every collection. */
(() => {
  'use strict';
  const cid = document.body.dataset.collection;
  const data = window.LF_CONTENT;
  const collection = data.collections.find(item => item.id === cid);
  const entries = data.entries.filter(item => item.collection === cid);
  window.$docsify = {
    name: collection.title, nameLink: '#/', loadSidebar: true,
    alias: { '/.*/_sidebar.md': '/_sidebar.md' }, subMaxLevel: 0,
    auto2top: true, coverpage: false, notFoundPage: '_404.md',
    search: { placeholder: '搜索本专题正文', noData: '本专题内没有找到结果', paths: 'auto', depth: 4, namespace: 'lf-' + cid },
    plugins: [function(hook, vm) {
      hook.doneEach(() => {
        const article = document.querySelector('.markdown-section');
        if (!article) return;
        const search = document.querySelector('.search input');
        if (search) search.setAttribute('aria-label','搜索本专题正文');
        const route = decodeURIComponent(vm.route.path).replace(/^\//,'').replace(/\.md$/,'');
        const entry = entries.find(item => item.source === cid + '/' + (route || 'README') + '.md');
        const h1 = article.querySelector('h1');
        const title = h1 ? h1.textContent : collection.title;
        document.title = title + ' · Linkforfuture';
        const crumb = document.createElement('div');
        crumb.className = 'lf-breadcrumb';
        const home = document.createElement('a'); home.href = '../topics.html'; home.textContent = '专题';
        const start = document.createElement('a'); start.href = '#/'; start.textContent = collection.title;
        crumb.append(home, document.createTextNode(' / '), start);
        if (route) crumb.append(document.createTextNode(' / ' + title));
        article.prepend(crumb);
        if (entry && h1) {
          const meta = document.createElement('div'); meta.className = 'lf-doc-meta';
          const label = document.createElement('span'); label.className = 'lf-tag'; label.textContent = entry.kind;
          const source = document.createElement('a'); source.href = entry.source.slice(cid.length+1); source.textContent = '查看 Markdown 源文件'; source.target = '_blank'; source.rel = 'noopener';
          meta.append(label, source);
          if (entry.updated) meta.append(document.createTextNode('更新于 ' + entry.updated));
          h1.after(meta);
        }
        const headings = Array.from(article.querySelectorAll('h2[id]'));
        if (headings.length > 2) {
          const toc = document.createElement('details'); toc.className = 'lf-doc-toc';
          const summary = document.createElement('summary'); summary.textContent = '本页目录 · ' + headings.length + ' 节';
          const list = document.createElement('ul');
          headings.forEach(heading => {
            const item = document.createElement('li'); const link = document.createElement('a');
            link.href = '#' + vm.route.path + '?id=' + encodeURIComponent(heading.id);
            link.textContent = heading.textContent; item.append(link); list.append(item);
          });
          toc.append(summary,list);
          const meta = article.querySelector('.lf-doc-meta'); (meta || h1 || crumb).after(toc);
        }
        article.querySelectorAll('pre').forEach(pre => {
          const code = pre.querySelector('code'); if (!code) return;
          const button = document.createElement('button'); button.type = 'button'; button.className = 'lf-copy-code'; button.textContent = '复制代码';
          button.addEventListener('click', async () => {
            try { await navigator.clipboard.writeText(code.textContent); button.textContent = '已复制'; }
            catch (_) { button.textContent = '请选中代码复制'; }
            setTimeout(() => { button.textContent = '复制代码'; }, 2000);
          }); pre.append(button);
        });
        const index = entries.indexOf(entry);
        if (index >= 0 && entries.length > 1) {
          const pager = document.createElement('nav'); pager.className = 'lf-doc-pager'; pager.setAttribute('aria-label','相邻文档');
          [[index-1,'← 上一篇：'],[index+1,'下一篇：']].forEach(([i,label]) => {
            if (!entries[i]) return;
            const link = document.createElement('a'); link.href = entries[i].url.slice(cid.length+1); link.textContent = label + entries[i].title; pager.append(link);
          }); article.append(pager);
        }
        if (matchMedia('(max-width:768px)').matches) document.body.classList.remove('close');
      });
      hook.mounted(() => {
        const toggle = document.querySelector('.sidebar-toggle');
        const sidebar = document.querySelector('.sidebar');
        if (!toggle || !sidebar) return;
        sidebar.id = 'lf-collection-nav'; toggle.setAttribute('aria-controls',sidebar.id);
        toggle.setAttribute('aria-label','展开或收起专题目录'); toggle.title = '专题目录';
        const sync = () => {
          const mobile = matchMedia('(max-width:768px)').matches;
          const shown = mobile ? document.body.classList.contains('close') : !document.body.classList.contains('close');
          toggle.setAttribute('aria-expanded',String(shown)); sidebar.inert = !shown;
        };
        new MutationObserver(sync).observe(document.body,{attributes:true,attributeFilter:['class']});
        window.addEventListener('resize',sync); sync();
        document.addEventListener('keydown', event => {
          if (event.key === 'Escape' && matchMedia('(max-width:768px)').matches && document.body.classList.contains('close')) {
            document.body.classList.remove('close'); toggle.focus();
          }
        });
      });
    }]
  };
})();
