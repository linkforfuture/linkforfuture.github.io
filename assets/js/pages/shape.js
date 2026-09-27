const fig = document.getElementById('fig');
  document.querySelectorAll('.legend [data-dim]').forEach(chip => {
    const dim = chip.dataset.dim;
    const on  = () => fig.classList.add('hl-' + dim);
    const off = () => fig.classList.remove('hl-' + dim);
    chip.addEventListener('mouseenter', on);
    chip.addEventListener('mouseleave', off);
    chip.addEventListener('focus', on);
    chip.addEventListener('blur', off);
  });
