/* Shared navigation and metadata-only directory filtering. No framework required. */
(() => {
  'use strict';
  if (document.body.classList.contains('lf-docs')) {
    document.querySelector('.lf-skip').addEventListener('click', event => {
      event.preventDefault();
      const article = document.getElementById('main') || document.getElementById('lf-content');
      article.tabIndex = -1; article.focus(); article.scrollIntoView();
    });
  }
  document.querySelectorAll('.lf-menu').forEach(menu => {
    menu.addEventListener('keydown', event => {
      if (event.key === 'Escape') { menu.open = false; menu.querySelector('summary').focus(); }
    });
    menu.querySelectorAll('a').forEach(link => link.addEventListener('click', () => { menu.open = false; }));
    document.addEventListener('click', event => { if (!menu.contains(event.target)) menu.open = false; });
  });
  const filters = document.querySelector('[data-lf-filters]');
  if (!filters) return;
  const query = filters.querySelector('input');
  const topic = filters.querySelector('select');
  const rows = Array.from(document.querySelectorAll('[data-lf-entry]'));
  const count = document.getElementById('lf-result-count');
  const empty = document.getElementById('lf-empty');
  function update() {
    const term = query.value.trim().toLocaleLowerCase();
    let shown = 0;
    rows.forEach(row => {
      row.hidden = !(row.textContent.toLocaleLowerCase().includes(term) && (!topic.value || row.dataset.topic === topic.value));
      if (!row.hidden) shown++;
    });
    document.querySelectorAll('[data-lf-group]').forEach(group => {
      group.hidden = !Array.from(group.querySelectorAll('[data-lf-entry]')).some(row => !row.hidden);
    });
    count.textContent = `显示 ${shown} / ${rows.length} 篇内容`;
    empty.hidden = shown !== 0;
  }
  filters.hidden = false;
  query.addEventListener('input', update);
  topic.addEventListener('change', update);
  if (location.hash) {
    const name = decodeURIComponent(location.hash.slice(1));
    if (Array.from(topic.options).some(option => option.value === name)) topic.value = name;
  }
  update();
})();
