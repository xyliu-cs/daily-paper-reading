// Paper Hub — app.js
//
// papers.json schema (add one entry per paper):
// {
//   "id":      "filename_without_md_extension",   // e.g. "my_paper_report"
//   "title":   "Paper Title",
//   "date":    "2025-06-01",
//   "venue":   "NeurIPS 2025",                    // optional
//   "tags":    ["LLM", "Efficiency"],             // optional
//   "summary": "One-sentence summary."            // optional
// }

(function () {
  'use strict';

  const content = document.getElementById('content');
  const nav     = document.getElementById('nav');
  const main    = document.getElementById('main');

  // ── marked config ──────────────────────────────────────────────────────────
  marked.setOptions({ breaks: false, gfm: true });

  // ── data ───────────────────────────────────────────────────────────────────
  async function loadPapers() {
    const res = await fetch('papers.json');
    if (!res.ok) throw new Error('Failed to load papers.json');
    return await res.json();
  }

  // ── render helpers ────────────────────────────────────────────────────────
  function renderNav(papers, activeId) {
    if (!papers.length) { nav.innerHTML = ''; return; }
    nav.innerHTML =
      '<div class="nav-section">Papers</div>' +
      papers.map(p => `
        <a href="#${p.id}" class="nav-item ${p.id === activeId ? 'active' : ''}">
          <div class="nav-item-title">${escHtml(p.title)}</div>
          ${p.date ? `<div class="nav-item-meta">${escHtml(p.date)}</div>` : ''}
        </a>
      `).join('');
  }

  function renderHome(papers) {
    if (!papers.length) {
      content.innerHTML = `
        <div class="empty-state">
          <strong>No papers yet.</strong>
          <p>Add entries to <code>papers.json</code> to get started.</p>
        </div>`;
      return;
    }
    content.innerHTML = `
      <h1 class="home-title">Paper Hub</h1>
      <p class="home-subtitle">${papers.length} paper note${papers.length !== 1 ? 's' : ''}</p>
      <div class="papers-list">
        ${papers.map(p => `
          <a href="#${p.id}" class="paper-card">
            <div class="paper-card-title">${escHtml(p.title)}</div>
            <div class="paper-card-meta">
              ${[p.date, p.venue].filter(Boolean).map(escHtml).join(' · ')}
            </div>
            ${p.summary ? `<div class="paper-card-summary">${escHtml(p.summary)}</div>` : ''}
            ${p.tags && p.tags.length ? `
              <div class="tags">
                ${p.tags.map(t => `<span class="tag">${escHtml(t)}</span>`).join('')}
              </div>` : ''}
          </a>
        `).join('')}
      </div>`;
  }

  async function renderPaper(id) {
    content.innerHTML = '<p style="color:var(--text-muted);padding:40px 0">Loading…</p>';
    try {
      const res = await fetch(`${id}.md`);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const md   = await res.text();
      const html = marked.parse(md);

      content.innerHTML = `
        <a href="#" class="back-btn">← All papers</a>
        <div class="paper-body">${html}</div>`;

      // render math if KaTeX is available
      if (typeof renderMathInElement !== 'undefined') {
        renderMathInElement(content.querySelector('.paper-body'), {
          delimiters: [
            { left: '$$', right: '$$', display: true  },
            { left: '$',  right: '$',  display: false },
          ],
          throwOnError: false,
        });
      }

      main.scrollTo(0, 0);
    } catch (err) {
      content.innerHTML = `
        <a href="#" class="back-btn">← All papers</a>
        <p style="color:#ef4444;margin-top:16px">Could not load <code>${escHtml(id)}.md</code>: ${escHtml(err.message)}</p>`;
    }
  }

  // ── router ─────────────────────────────────────────────────────────────────
  let papers = [];

  async function route() {
    const id = decodeURIComponent(location.hash.slice(1));
    renderNav(papers, id);
    if (id) {
      await renderPaper(id);
    } else {
      renderHome(papers);
    }
  }

  // ── mobile nav toggle ──────────────────────────────────────────────────────
  document.getElementById('menu-toggle').addEventListener('click', () => {
    nav.classList.toggle('open');
  });

  // close nav on item click (mobile)
  nav.addEventListener('click', e => {
    if (e.target.closest('.nav-item')) nav.classList.remove('open');
  });

  // ── init ───────────────────────────────────────────────────────────────────
  async function init() {
    try {
      papers = await loadPapers();
    } catch (e) {
      papers = [];
      console.warn('papers.json not found or invalid:', e.message);
    }
    window.addEventListener('hashchange', route);
    await route();
  }

  // ── util ───────────────────────────────────────────────────────────────────
  function escHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  init();
})();
