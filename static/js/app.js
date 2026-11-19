(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* Sidebar (mobile) */
  const side = $('#side'), scrim = $('#scrim'), menuBtn = $('#menuBtn');
  const setSide = (open) => {
    if (!side) return;
    side.classList.toggle('open', open);
    scrim && scrim.classList.toggle('show', open);
    menuBtn && menuBtn.setAttribute('aria-expanded', open);
  };
  menuBtn && menuBtn.addEventListener('click', () => setSide(!side.classList.contains('open')));
  scrim && scrim.addEventListener('click', () => setSide(false));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setSide(false); });

  /* Upload: drag and drop, preview, validation, loading state */
  const drop = $('#dropzone'), input = $('#imageInput');
  if (drop && input) {
    const meta = $('#fileMeta'), img = $('#previewImg'), btn = $('#analyzeBtn'), form = drop.closest('form');
    const MAX = 5 * 1024 * 1024, OK = ['image/jpeg', 'image/png'];
    btn.disabled = !input.files.length; // only disabled once JS is running
    const reset = (text, isError) => {
      drop.classList.remove('has-file');
      btn.disabled = true;
      meta.textContent = text;
      meta.classList.toggle('err', !!isError);
    };
    const show = (f) => {
      if (!f) return reset('No file selected');
      if (!OK.includes(f.type)) { input.value = ''; return reset('Unsupported format. Choose a JPG or PNG image.', true); }
      if (f.size > MAX) { input.value = ''; return reset('That file is over 5 MB. Choose a smaller photo.', true); }
      img.src = URL.createObjectURL(f);
      drop.classList.add('has-file');
      meta.classList.remove('err');
      meta.textContent = `${f.name} · ${(f.size / 1048576).toFixed(1)} MB`;
      btn.disabled = false;
    };
    input.addEventListener('change', () => show(input.files[0]));
    ['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('is-over'); }));
    ['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, () => drop.classList.remove('is-over')));
    drop.addEventListener('drop', (e) => {
      e.preventDefault();
      if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; show(input.files[0]); }
    });
    form.addEventListener('submit', () => {
      drop.classList.add('is-loading');
      btn.disabled = true;
      btn.firstElementChild.textContent = 'Analyzing…';
    });
    window.addEventListener('pageshow', (e) => {
      if (e.persisted) { drop.classList.remove('is-loading'); btn.firstElementChild.textContent = 'Analyze disease'; show(input.files[0]); }
    });
  }

  /* Bring results into view */
  const results = $('#results');
  if (results) results.scrollIntoView({ behavior: calm ? 'auto' : 'smooth', block: 'start' });

  /* Recommendation text -> bullet rows (text is set with textContent, never as HTML) */
  $$('.rec-body').forEach((el) => {
    const text = el.textContent.trim();
    let parts = text.split(/\n+/);
    if (parts.length < 2) parts = text.split(/\s+[-•]\s+/);
    parts = parts.map((s) => s.replace(/^\s*(?:[-•*]|\d+[.)])\s*/, '').trim()).filter(Boolean);
    if (parts.length < 2) return;
    const ul = document.createElement('ul');
    ul.className = 'rec-list';
    parts.forEach((p) => { const li = document.createElement('li'); li.textContent = p; ul.appendChild(li); });
    el.replaceChildren(ul);
  });

  /* Tabs */
  $$('[role="tablist"]').forEach((list) => {
    const tabs = $$('[role="tab"]', list);
    const activate = (t) => tabs.forEach((x) => {
      const on = x === t;
      x.setAttribute('aria-selected', on);
      x.tabIndex = on ? 0 : -1;
      document.getElementById(x.getAttribute('aria-controls')).hidden = !on;
    });
    tabs.forEach((t, i) => {
      t.addEventListener('click', () => activate(t));
      t.addEventListener('keydown', (e) => {
        if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
        const n = tabs[(i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length];
        n.focus(); activate(n);
      });
    });
  });

  /* History filters */
  $$('[data-autosubmit]').forEach((el) => el.addEventListener('change', () => el.form.submit()));

  /* Contact: opens the visitor's mail app with the message filled in */
  const cf = $('#contactForm');
  cf && cf.addEventListener('submit', (e) => {
    e.preventDefault();
    const d = new FormData(cf);
    const body = `${d.get('message')}\n\n${d.get('name')} (${d.get('email')})`;
    location.href = `mailto:vermatushar1117@gmail.com?subject=${encodeURIComponent(d.get('subject'))}&body=${encodeURIComponent(body)}`;
  });
})();