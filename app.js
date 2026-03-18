// Daily Paper Reading — app.js
//
// papers.json schema:
// {
//   "id":          "my_paper",          // file = {id}.md  (or {id}_{lang}.md when langs is set)
//   "title":       "Paper Title",
//   "report_date": "2026-03-15",        // date we published this note  ← shown in main list
//   "date":        "2025-01",           // original paper date           ← shown as secondary info
//   "venue":       "NeurIPS 2025",      // optional
//   "tags":        ["LLM"],             // optional
//   "summary":     "One-line summary.", // optional
//   "langs":       ["zh", "en"]        // optional: enables lang switcher; files = {id}_zh.md etc.
// }
//
// Likes and language preferences are stored in localStorage (browser-local only).

(function () {
  'use strict';

  const content = document.getElementById('content');
  const nav     = document.getElementById('nav');
  const main    = document.getElementById('main');

  // ── KaTeX / marked ────────────────────────────────────────────────────────
  function renderKatex(math, display) {
    if (typeof katex === 'undefined') return null;
    try {
      return katex.renderToString(math, { displayMode: display, throwOnError: false });
    } catch (e) { return null; }
  }

  marked.use({
    extensions: [
      {
        name: 'displayMath',
        level: 'block',
        start(src) { return src.indexOf('$$'); },
        tokenizer(src) {
          const m = /^\$\$([\s\S]+?)\$\$/.exec(src);
          if (m) return { type: 'displayMath', raw: m[0], math: m[1].trim() };
        },
        renderer(token) {
          return renderKatex(token.math, true) ??
            '<div class="math-display"><em>Math error</em></div>\n';
        },
      },
      {
        name: 'inlineMath',
        level: 'inline',
        start(src) { return src.indexOf('$'); },
        tokenizer(src) {
          const m = /^\$(?!\$)((?:[^$\\]|\\[\s\S])+?)\$/.exec(src);
          if (m) return { type: 'inlineMath', raw: m[0], math: m[1] };
        },
        renderer(token) {
          return renderKatex(token.math, false) ??
            '<span class="math-inline"><em>Math error</em></span>';
        },
      },
    ],
  });
  marked.setOptions({ breaks: false, gfm: true });

  // ── localStorage helpers ──────────────────────────────────────────────────
  const isLiked     = id      => localStorage.getItem('like_' + id) === '1';
  const toggleLike  = id      => { const v = !isLiked(id); localStorage.setItem('like_' + id, v ? '1' : '0'); return v; };
  const getPrefLang = id      => localStorage.getItem('lang_' + id) || null;
  const setPrefLang = (id, l) => localStorage.setItem('lang_' + id, l);

  // ── TOC (floating widget inside content, not sidebar) ─────────────────────
  let _scrollCleanup = null;

  function buildToc(bodyEl) {
    if (_scrollCleanup) { _scrollCleanup(); _scrollCleanup = null; }

    const floatEl = document.getElementById('toc-float');
    if (!floatEl) return;

    const sections = [...bodyEl.querySelectorAll('h2, h3, h4')];
    sections.forEach((el, i) => { el.id = 'toc-' + i; });

    if (!sections.length) {
      floatEl.hidden = true;
      floatEl.innerHTML = '';
      return;
    }
    floatEl.hidden = false;

    floatEl.innerHTML =
      '<div class="toc-float-title">Contents</div>' +
      sections.map((el, i) => {
        const indent = (parseInt(el.tagName[1]) - 2) * 10; // h2→0, h3→10, h4→20
        return `<a class="toc-item" href="#toc-${i}" style="padding-left:${12 + indent}px">${escHtml(el.textContent)}</a>`;
      }).join('');

    // Click: scroll without triggering a hash-change navigation
    floatEl.addEventListener('click', e => {
      const item = e.target.closest('.toc-item');
      if (!item) return;
      e.preventDefault();
      document.getElementById(item.getAttribute('href').slice(1))
        ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });

    // Highlight active section while scrolling
    const tocLinks = [...floatEl.querySelectorAll('.toc-item')];
    const onScroll = () => {
      const containerTop = main.getBoundingClientRect().top;
      let active = 0;
      sections.forEach((el, i) => {
        if (el.getBoundingClientRect().top - containerTop < 120) active = i;
      });
      tocLinks.forEach((a, i) => a.classList.toggle('active', i === active));
    };
    main.addEventListener('scroll', onScroll, { passive: true });
    _scrollCleanup = () => main.removeEventListener('scroll', onScroll);
    onScroll();
  }

  // ── sidebar: always shows paper list ──────────────────────────────────────
  function renderNav(papers, activeId) {
    nav.innerHTML =
      (papers.length ? '<div class="nav-section">Papers</div>' : '') +
      papers.map(p => {
        const meta = p.report_date || p.date || '';
        return `
          <a href="#${p.id}" class="nav-item ${p.id === activeId ? 'active' : ''}">
            <div class="nav-item-title">${escHtml(p.title)}</div>
            ${meta ? `<div class="nav-item-meta">${escHtml(meta)}</div>` : ''}
          </a>`;
      }).join('');
  }

  // ── home page ─────────────────────────────────────────────────────────────
  function renderHome(papers) {
    const floatEl = document.getElementById('toc-float');
    if (floatEl) { floatEl.hidden = true; floatEl.innerHTML = ''; }
    if (!papers.length) {
      content.innerHTML = `
        <div class="empty-state">
          <strong>No papers yet.</strong>
          <p>Add entries to <code>papers.json</code> to get started.</p>
        </div>`;
      return;
    }
    content.innerHTML = `
      <h1 class="home-title">Daily Paper Reading</h1>
      <p class="home-subtitle">${papers.length} note${papers.length !== 1 ? 's' : ''}</p>
      <div class="papers-list">
        ${papers.map(p => {
          const reportDate = p.report_date || '';
          const paperMeta  = [p.date, p.venue].filter(Boolean).join(' · ');
          return `
            <a href="#${p.id}" class="paper-card">
              <div class="paper-card-date">${escHtml(reportDate)}</div>
              <div class="paper-card-body">
                <div class="paper-card-title-row">
                  <div class="paper-card-title">${escHtml(p.title)}</div>
                  <button class="like-btn ${isLiked(p.id) ? 'liked' : ''}"
                          data-id="${escHtml(p.id)}" title="Like">${isLiked(p.id) ? '♥' : '♡'}</button>
                </div>
                ${paperMeta  ? `<div class="paper-card-meta">${escHtml(paperMeta)}</div>` : ''}
                ${p.summary  ? `<div class="paper-card-summary">${escHtml(p.summary)}</div>` : ''}
                ${p.tags?.length ? `<div class="tags">${p.tags.map(t => `<span class="tag">${escHtml(t)}</span>`).join('')}</div>` : ''}
              </div>
            </a>`;
        }).join('')}
      </div>`;

    content.querySelectorAll('.like-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        e.preventDefault(); e.stopPropagation();
        const liked = toggleLike(btn.dataset.id);
        btn.classList.toggle('liked', liked);
        btn.textContent = liked ? '♥' : '♡';
      });
    });
  }

  // ── paper page ────────────────────────────────────────────────────────────
  async function renderPaper(paperId, papers, lang) {
    const paper     = papers.find(p => p.id === paperId);
    const multiLang = (paper?.langs?.length ?? 0) > 1;
    const hasLangs  = (paper?.langs?.length ?? 0) > 0;

    if (hasLangs) {
      if (!lang || !paper.langs.includes(lang))
        lang = getPrefLang(paperId) || paper.langs[0];
    } else {
      lang = null;
    }

    const filename = hasLangs ? `src/${paperId}_${lang}.md` : `src/${paperId}.md`;
    content.innerHTML = '<p style="color:var(--text-muted);padding:40px 0">Loading…</p>';

    try {
      const res = await fetch(filename);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const html = marked.parse(await res.text());

      // Language switcher (top-right, only when multiple langs available)
      const langSwitcher = multiLang
        ? `<div class="lang-switcher">${paper.langs.map(l =>
            `<button class="lang-btn${l === lang ? ' active' : ''}" data-lang="${l}">${escHtml(l.toUpperCase())}</button>`
          ).join('')}</div>`
        : '';

      const liked = isLiked(paperId);
      content.innerHTML = `
        <div class="paper-topbar">
          <a href="#" class="back-btn">← Back</a>
          ${langSwitcher}
        </div>
        <div class="paper-body">${html}</div>
        <div class="paper-footer">
          <button class="like-btn-paper${liked ? ' liked' : ''}" id="paper-like-btn">
            <span class="like-icon">${liked ? '♥' : '♡'}</span> Like this note
          </button>
        </div>`;

      buildToc(content.querySelector('.paper-body'));

      content.querySelectorAll('.lang-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          setPrefLang(paperId, btn.dataset.lang);
          renderPaper(paperId, papers, btn.dataset.lang);
        });
      });

      document.getElementById('paper-like-btn')?.addEventListener('click', () => {
        const liked = toggleLike(paperId);
        const btn = document.getElementById('paper-like-btn');
        btn?.classList.toggle('liked', liked);
        if (btn) btn.querySelector('.like-icon').textContent = liked ? '♥' : '♡';
      });

      main.scrollTo({ top: 0, behavior: 'instant' });

    } catch (err) {
      content.innerHTML = `
        <div class="paper-topbar"><a href="#" class="back-btn">← Back</a></div>
        <p style="color:#ef4444;margin-top:16px">
          Could not load <code>${escHtml(filename)}</code>: ${escHtml(err.message)}</p>`;
    }
  }

  // ── router ────────────────────────────────────────────────────────────────
  let papers = [];

  async function route() {
    const id = decodeURIComponent(location.hash.slice(1));
    if (_scrollCleanup && !id) { _scrollCleanup(); _scrollCleanup = null; }
    renderNav(papers, id);
    if (id) {
      await renderPaper(id, papers, null);
    } else {
      renderHome(papers);
    }
  }

  // ── mobile nav ────────────────────────────────────────────────────────────
  document.getElementById('menu-toggle').addEventListener('click', () => {
    nav.classList.toggle('open');
  });
  nav.addEventListener('click', e => {
    if (e.target.closest('.nav-item')) nav.classList.remove('open');
  });

  // ── init ──────────────────────────────────────────────────────────────────
  async function init() {
    try {
      const res = await fetch('papers.json');
      papers = res.ok ? await res.json() : [];
    } catch (e) {
      papers = [];
      console.warn('papers.json:', e.message);
    }
    window.addEventListener('hashchange', route);
    await route();
  }

  // ── util ──────────────────────────────────────────────────────────────────
  function escHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  init();
})();
