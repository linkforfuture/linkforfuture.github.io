(() => {
      const editButton = document.getElementById('edit-button');
      const status = document.getElementById('status');
      const fields = Array.from(document.querySelectorAll('[data-editable]'));
      let editing = false;
      let unsaved = false;
      const tasks = Array.from(document.querySelectorAll('[data-task]'));
      const printRoadmap = document.getElementById('print-roadmap');
      function updateProgress() {
        const completed = tasks.filter(task => task.checked).length;
        const progress = document.getElementById('learning-progress');
        progress.max = tasks.length;
        progress.value = completed;
        document.getElementById('progress-text').textContent = `${completed} / ${tasks.length}项完成 · ${
          completed === 0 ? '尚未开始' : completed === tasks.length ? '计划已完成' : '进行中'}`;
      }
      tasks.forEach(task => task.addEventListener('change', () => {
        updateProgress();
        unsaved = true;
        status.textContent = '学习进度已更新；请下载HTML保存当前进度。';
      }));
      printRoadmap.addEventListener('change', () => {
        document.body.classList.toggle('print-roadmap', printRoadmap.checked);
        unsaved = true;
      });
      document.body.classList.toggle('print-roadmap', printRoadmap.checked);
      updateProgress();

      editButton.addEventListener('click', () => {
        editing = !editing;
        document.body.classList.toggle('is-editing', editing);
        editButton.setAttribute('aria-pressed', String(editing));
        editButton.textContent = editing ? '完成编辑' : '编辑内容';
        fields.forEach(field => {
          if (editing) field.setAttribute('contenteditable', 'plaintext-only');
          else field.removeAttribute('contenteditable');
        });
        status.textContent = editing
          ? '点击正文即可修改；编辑仅保留在当前页面，完成后请下载HTML保存。'
          : '可继续编辑，或下载HTML保存修改；打印按钮可导出PDF。';
        if (editing) fields[0].focus();
      });

      fields.forEach(field => field.addEventListener('input', () => {
        unsaved = true;
        if (field.hasAttribute('data-placeholder') && !/[\[\]]/.test(field.textContent)) {
          field.removeAttribute('data-placeholder');
        }
      }));

      document.getElementById('download-button').addEventListener('click', async () => {
        const copy = document.documentElement.cloneNode(true);
        // Inline same-origin assets so the downloaded resume remains standalone.
        try {
          for (const link of Array.from(copy.querySelectorAll('link[rel="stylesheet"]'))) {
            const response = await fetch(new URL(link.getAttribute('href'), document.baseURI));
            if (!response.ok) throw new Error('stylesheet');
            const style = document.createElement('style');
            style.textContent = await response.text();
            link.replaceWith(style);
          }
          for (const script of Array.from(copy.querySelectorAll('script[src]'))) {
            const response = await fetch(new URL(script.getAttribute('src'), document.baseURI));
            if (!response.ok) throw new Error('script');
            script.textContent = (await response.text()).replace(/<\/script/gi, '<\\/script');
            script.removeAttribute('src');
            script.removeAttribute('defer');
            // Inline scripts do not honor defer. Move head scripts after the DOM.
            if (script.parentElement === copy.querySelector('head')) copy.querySelector('body').append(script);
          }
          copy.querySelectorAll('a[href]').forEach(a => {
            if (!a.getAttribute('href').startsWith('#')) a.href = new URL(a.getAttribute('href'), document.baseURI).href;
          });
        } catch (_) {
          status.textContent = '下载失败：无法读取页面资源。请通过网站或本地 HTTP 预览重试，修改仍保留在当前页。';
          return;
        }
        const originalCheckboxes = document.querySelectorAll('input[type="checkbox"]');
        copy.querySelectorAll('input[type="checkbox"]').forEach((checkbox, index) => {
          checkbox.toggleAttribute('checked', originalCheckboxes[index].checked);
        });
        copy.querySelector('body').classList.remove('is-editing');
        copy.querySelectorAll('[contenteditable]').forEach(field => field.removeAttribute('contenteditable'));
        const copiedButton = copy.querySelector('#edit-button');
        copiedButton.textContent = '编辑内容';
        copiedButton.setAttribute('aria-pressed', 'false');
        copy.querySelector('#status').textContent = '已加载保存的简历。点击“编辑内容”可继续修改。';
        const blob = new Blob(['<!DOCTYPE html>\n', copy.outerHTML], { type: 'text/html;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'HCCL-集合通信研发-个人简历.html';
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        unsaved = false;
        status.textContent = '已发起HTML下载，请确认文件已保存。修改后的文件可继续编辑和打印。';
      });

      document.getElementById('print-button').addEventListener('click', () => window.print());
      window.addEventListener('beforeunload', event => {
        if (!unsaved) return;
        event.preventDefault();
        event.returnValue = '';
      });
    })();
