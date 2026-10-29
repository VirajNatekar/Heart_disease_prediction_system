(() => {
  const form = document.querySelector('.assessment-form');
  document.querySelectorAll('.score-fill[data-percent]').forEach((bar) => {
    const percent = Number(bar.dataset.percent);
    if (Number.isFinite(percent)) bar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  });
  if (!form) return;

  const resetButton = document.querySelector('#reset-form');
  form.addEventListener('submit', (event) => {
    // Let the server remain the source of truth, but provide fast, accessible feedback.
    const invalid = [...form.querySelectorAll('input[required], select[required]')]
      .find((field) => !field.value || !field.checkValidity());
    if (invalid) {
      event.preventDefault();
      invalid.focus();
      invalid.reportValidity();
      return;
    }

    const submitButton = form.querySelector('button[type="submit"]');
    if (submitButton) {
      submitButton.disabled = true;
      submitButton.classList.add('is-loading');
      submitButton.innerHTML = 'Evaluating input <span class="button-spinner" aria-hidden="true"></span>';
    }
  });

  if (resetButton) {
    resetButton.addEventListener('click', () => {
      window.setTimeout(() => {
        form.querySelectorAll('select').forEach((select) => { select.selectedIndex = 0; });
        form.querySelectorAll('input').forEach((input) => { input.value = ''; });
      }, 0);
    });
  }

  // Avoid displaying browser-saved autofill details in a second use of the form.
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
      const submitButton = form.querySelector('button[type="submit"]');
      if (submitButton) {
        submitButton.disabled = false;
        submitButton.classList.remove('is-loading');
      }
    }
  });
})();
