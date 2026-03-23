'use strict';

/* ── Code formatter ──────────────────────────────────────────────────────────── */
function formatCode(raw, language) {
  if (!raw) return '';
  const lang = (language || '').toUpperCase();
  if (lang === 'PANDAS' || lang === 'PYTHON') {
    let out = raw.trim().replace(/\s+/g, ' ');
    out = out.replace(/\.([A-Za-z_]\w*)\(/g, '\n  .$1(');
    out = out.replace(/,\s*([A-Za-z_"'\[{])/g, ',\n    $1');
    return out.trim();
  }
  const CLAUSES = ['UNION ALL','UNION','INTERSECT','EXCEPT','GROUP BY','ORDER BY',
                   'WITH','SELECT','FROM','WHERE','HAVING','LIMIT','OFFSET'];
  const JOINS   = ['INNER JOIN','LEFT OUTER JOIN','RIGHT OUTER JOIN','FULL OUTER JOIN',
                   'LEFT JOIN','RIGHT JOIN','FULL JOIN','CROSS JOIN','JOIN'];
  let out = raw.trim().replace(/\s+/g, ' ');
  CLAUSES.forEach(kw => {
    out = out.replace(new RegExp('\\b(' + kw.replace(/ /g,'\\s+') + ')\\b','gi'), '\n$1');
  });
  JOINS.forEach(kw => {
    out = out.replace(new RegExp('\\b(' + kw.replace(/ /g,'\\s+') + ')\\b','gi'), '\n  $1');
  });
  out = out.replace(/\b(AND|OR)\b/gi, '\n  $1');
  out = out.replace(/\)\s*\n\s*(SELECT)/gi, ')\n$1');
  return out.split('\n').map(l => l.trimEnd())
    .filter((l,i) => i === 0 || l.trim() !== '').join('\n').trim();
}

/* ── State ──────────────────────────────────────────────────────────────────── */
let allSources   = [];
let allQueries   = [];
let activeSource = null;

/* ── Boot ───────────────────────────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', async () => {
  await loadSources();
  updateTopbarMeta();
});

/* ── Utilities ──────────────────────────────────────────────────────────────── */
function ts() {
  return new Date().toLocaleTimeString('en-GB',
    { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
function esc(s) {
  return String(s ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function log(msg, cls = 'log-info') {
  const box  = document.getElementById('logBox');
  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `<span class="log-ts">${ts()}</span><span class="${cls}">${msg}</span>`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}
function isNumeric(v) { return v !== null && v !== '' && !isNaN(Number(v)); }

function activeKind() {
  return allSources.find(s => s.id === activeSource)?.kind ?? 'PANDAS';
}

/* ── Sources ─────────────────────────────────────────────────────────────────── */
async function loadSources() {
  const grid = document.getElementById('sourceGrid');
  grid.innerHTML = '<span style="font-size:12px;color:var(--text-muted)">Loading sources…</span>';
  try {
    const res = await fetch('/api/sources');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    allSources = await res.json();
    renderSourceGrid();
  } catch (e) {
    grid.innerHTML = `<span style="color:var(--err);font-size:12px">Failed: ${esc(e.message)}</span>`;
  }
}

function renderSourceGrid() {
  const grid = document.getElementById('sourceGrid');
  grid.innerHTML = '';
  allSources.forEach(src => {
    const card = document.createElement('div');
    card.className = 'source-card' + (activeSource === src.id ? ' active' : '');
    card.id = `src-${src.id}`;
    card.onclick = () => selectSource(src.id);
    const qCount = src.statements?.query_count ?? 0;
    const qLine  = qCount > 0 ? `<div class="source-q-count">↳ ${qCount} queries</div>` : '';
    card.innerHTML = `
      <div class="source-name">${esc(src.label)}</div>
      <div class="source-type">${esc(src.type)}</div>
      ${badgeHtml(src)}
      ${qLine}`;
    grid.appendChild(card);
  });
}

function badgeHtml(src) {
  if (src.error) return `<span class="source-badge badge-miss"><span class="dot"></span>Config error</span>`;
  const missing = (src.files || []).filter(f => !f.exists).length;
  if (missing === 0)              return `<span class="source-badge badge-ready"><span class="dot"></span>Ready</span>`;
  if (missing < src.files.length) return `<span class="source-badge badge-warn"><span class="dot"></span>${missing} missing</span>`;
  return `<span class="source-badge badge-miss"><span class="dot"></span>Not configured</span>`;
}

async function selectSource(id) {
  activeSource = id;
  renderSourceGrid();

  const src = allSources.find(s => s.id === id);
  document.getElementById('workArea').style.display = '';
  renderReadiness(src);
  updateRunButton(src);
  updateKindIndicator(src);
  updateTopbarMeta();

  document.getElementById('genPanel').style.display       = 'none';
  document.getElementById('doneFlash').style.display      = 'none';
  document.getElementById('queriesSection').style.display = 'none';
  allQueries = [];

  const stmts  = src.statements ?? { exists: false, query_count: 0 };
  const gen    = src.generation  ?? {};
  const banner = document.getElementById('existsBanner');
  document.getElementById('runRow').style.display = '';

  const isInterrupted  = (gen.done_count ?? 0) > 0 && !gen.is_complete;
  const isComplete     = !!gen.is_complete;
  // Queries exist for OTHER kinds but none for this kind yet
  const isOtherKindOnly = !isComplete && (gen.done_count ?? 0) === 0 && stmts.exists;

  try {
    if (isInterrupted)    { setBanner('interrupted', gen, stmts);  banner.style.display = ''; }
    else if (isComplete)  { setBanner('complete',    gen, stmts);  banner.style.display = ''; }
    else if (isOtherKindOnly) { setBanner('other_kind', gen, stmts); banner.style.display = ''; }
    else                  { banner.style.display = 'none'; }
  } catch (e) {
    console.error('Banner error:', e);
    banner.style.display = 'none';
  }

  await loadQueries();
}

/* ── Kind indicator ──────────────────────────────────────────────────────────── */
function updateKindIndicator(src) {
  const el   = document.getElementById('kindIndicator');
  const pill = document.getElementById('kindPill');
  if (!src || src.error) { el.style.display = 'none'; return; }

  el.style.display = '';
  if (src.kind) {
    // Fully configured — normal green pill
    pill.textContent = src.kind;
    pill.className   = 'kind-pill';
  } else {
    // Kind missing from config — warn the user
    pill.textContent = 'not configured';
    pill.className   = 'kind-pill kind-pill-warn';
  }
}

/* ── Readiness ───────────────────────────────────────────────────────────────── */
function renderReadiness(src) {
  const panel = document.getElementById('readinessPanel');
  if (src.error) {
    panel.innerHTML = `<div class="readiness-row">
      <span class="r-icon">⚠</span><span class="r-name">Config error</span>
      <span class="r-path">${esc(src.error)}</span>
      <span class="r-status r-miss">Error</span></div>`;
    return;
  }
  panel.innerHTML = (src.files || []).map(f => `
    <div class="readiness-row">
      <span class="r-icon">${f.exists ? '✓' : '✗'}</span>
      <span class="r-name">${esc(f.name)}</span>
      <span class="r-path" title="${esc(f.path)}">${esc(f.path)}</span>
      <span class="r-status ${f.exists ? 'r-ok' : 'r-miss'}">${f.exists ? 'Found' : 'Missing'}</span>
    </div>`).join('');
}

function updateRunButton(src) {
  const btn  = document.getElementById('runBtn');
  const hint = document.getElementById('runHint');
  const kind       = src.kind ?? null;   // null = genuinely not configured
  const filesReady = !src.error && (src.files || []).every(f => f.exists);
  const ready      = filesReady && !!kind;
  btn.disabled = !ready;
  const icon = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2.5 1.5l8 4.5-8 4.5V1.5z" fill="currentColor"/></svg>`;
  const gen   = src.generation ?? {};
  const label = kind
    ? (gen.is_complete ? `Re-run ${kind}` : `Run ${kind}`)
    : 'Run';
  btn.innerHTML = `${icon} ${label}`;

  // Resume button — visible only when generation started but isn't complete
  let resumeBtn = document.getElementById('resumeBtn');
  const isInterrupted = (gen.done_count ?? 0) > 0 && !gen.is_complete;
  if (isInterrupted && ready) {
    if (!resumeBtn) {
      resumeBtn = document.createElement('button');
      resumeBtn.className = 'btn btn-resume';
      resumeBtn.id        = 'resumeBtn';
      resumeBtn.onclick   = () => resumeStatements();
      btn.insertAdjacentElement('afterend', resumeBtn);
    }
    resumeBtn.innerHTML     = `↩ Resume ${kind} <span class="resume-progress">${gen.done_count}/${gen.total}</span>`;
    resumeBtn.style.display = '';
  } else if (resumeBtn) {
    resumeBtn.style.display = 'none';
  }

  if (src.error) {
    hint.textContent = 'Config could not be loaded for this source.';
  } else if (!kind) {
    hint.textContent = 'cfg.statement_generation.kind is not set — generation disabled.';
  } else if (!filesReady) {
    hint.textContent = `Missing: ${(src.files||[]).filter(f=>!f.exists).map(f=>f.name).join(', ')}`;
  } else {
    hint.textContent = `${src.files.length} required files verified.`;
  }
}

function updateTopbarMeta() {
  const el  = document.getElementById('topbarMeta');
  if (!activeSource) { el.textContent = '—'; return; }
  const src = allSources.find(s => s.id === activeSource);
  el.textContent = src
    ? `${src.label} · ${src.type}${src.kind ? ' · ' + src.kind : ''}`
    : '—';
}

/* ── Run click ───────────────────────────────────────────────────────────────── */
function onRunClick() {
  const gen = allSources.find(s => s.id === activeSource)?.generation ?? {};
  if ((gen.done_count ?? 0) > 0) {
    // Current kind has (at least partial) results — confirm before overwriting
    askRegenerate();
  } else {
    // No results for this kind yet — start fresh immediately
    startStatements();
  }
}

/* ── Banner ──────────────────────────────────────────────────────────────────── */
const _ICON_OK = `<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
  <circle cx="10" cy="10" r="9" stroke="var(--ok)" stroke-width="1.5"/>
  <path d="M6 10l3 3 5-5" stroke="var(--ok)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>`;
const _ICON_WARN = `<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
  <circle cx="10" cy="10" r="9" stroke="var(--warn)" stroke-width="1.5"/>
  <path d="M10 6v5" stroke="var(--warn)" stroke-width="1.5" stroke-linecap="round"/>
  <circle cx="10" cy="14.5" r="1" fill="var(--warn)"/>
</svg>`;

function setBanner(state, gen, stmts) {
  const banner  = document.getElementById('existsBanner');
  const iconEl  = document.getElementById('existsIcon');
  const titleEl = document.getElementById('existsTitle');
  const subEl   = document.getElementById('existsSub');
  const actEl   = document.getElementById('existsActions');
  if (!banner || !actEl) return;
  banner.dataset.state = state;
  actEl.innerHTML = '';
  const kind = gen.kind ?? activeKind();

  if (state === 'interrupted') {
    iconEl.innerHTML    = _ICON_WARN;
    titleEl.textContent = `${kind} generation incomplete`;
    subEl.textContent   = `${gen.done_count} of ${gen.total} processed (last: #${gen.last_idx})`;
    if (stmts.exists) {
      const viewBtn = document.createElement('button');
      viewBtn.className   = 'btn btn-outline';
      viewBtn.textContent = 'Scroll to queries ↓';
      viewBtn.onclick     = scrollToQueries;
      actEl.appendChild(viewBtn);
    }
  } else if (state === 'complete') {
    const qCount = gen.query_count ?? 0;
    iconEl.innerHTML    = _ICON_OK;
    titleEl.textContent = `${qCount} ${kind} quer${qCount === 1 ? 'y' : 'ies'} generated`;
    subEl.textContent   = `Use Re-run to regenerate from scratch, or scroll down to view.`;
    const viewBtn = document.createElement('button');
    viewBtn.className   = 'btn btn-outline';
    viewBtn.textContent = 'Scroll to queries ↓';
    viewBtn.onclick     = scrollToQueries;
    actEl.appendChild(viewBtn);
  } else if (state === 'other_kind') {
    const otherKind = kind;
    iconEl.innerHTML    = _ICON_WARN;
    titleEl.textContent = `No ${otherKind} queries yet`;
    subEl.textContent   = `${stmts.query_count} quer${stmts.query_count === 1 ? 'y' : 'ies'} exist in other language(s). Click Run ${otherKind} to generate for this source.`;
    const viewBtn = document.createElement('button');
    viewBtn.className   = 'btn btn-outline';
    viewBtn.textContent = 'View other queries ↓';
    viewBtn.onclick     = scrollToQueries;
    actEl.appendChild(viewBtn);
  } else {
    iconEl.innerHTML    = _ICON_OK;
    titleEl.textContent = `${stmts.query_count} quer${stmts.query_count === 1 ? 'y' : 'ies'} generated`;
    subEl.textContent   = `Use Run to re-generate ${kind} queries, or scroll down to view.`;
    const viewBtn = document.createElement('button');
    viewBtn.className   = 'btn btn-outline';
    viewBtn.textContent = 'Scroll to queries ↓';
    viewBtn.onclick     = scrollToQueries;
    actEl.appendChild(viewBtn);
  }
}

function scrollToQueries() {
  const qs = document.getElementById('queriesSection');
  if (qs && qs.style.display !== 'none') {
    qs.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } else {
    loadQueries().then(() =>
      document.getElementById('queriesSection')
        .scrollIntoView({ behavior: 'smooth', block: 'start' })
    );
  }
}

/* ── Confirm ─────────────────────────────────────────────────────────────────── */
function askRegenerate() {
  const kind = activeKind();
  document.getElementById('confirmBody').textContent =
    `This will overwrite all existing ${kind} results and restart from match #0.`;
  document.getElementById('confirmOkBtn').onclick = () => { closeConfirm(); startStatements(); };
  document.getElementById('confirmBackdrop').style.display = '';
  document.getElementById('confirmDialog').style.display   = '';
}
function closeConfirm() {
  document.getElementById('confirmBackdrop').style.display = 'none';
  document.getElementById('confirmDialog').style.display   = 'none';
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closeConfirm(); closeModal(); }
});

/* ── Generation ──────────────────────────────────────────────────────────────── */
function _resetGenPanel(statusText) {
  document.getElementById('genPanel').style.display   = '';
  document.getElementById('doneFlash').style.display  = 'none';
  document.getElementById('logBox').innerHTML         = '';
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('progressTxt').textContent  = '0 / 0 matches';
  document.getElementById('succCount').textContent    = '0 ✓';
  document.getElementById('failCount').textContent    = '0 ✗';
  document.getElementById('genStatus').textContent    = statusText;
}

function _setBtnState(label, running = false) {
  const btn  = document.getElementById('runBtn');
  const icon = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2.5 1.5l8 4.5-8 4.5V1.5z" fill="currentColor"/></svg>`;
  if (!btn) return;
  btn.disabled  = running;
  btn.innerHTML = running ? `<span class="spinner-sm"></span> ${label}` : `${icon} ${label}`;
}

function resumeStatements() {
  const kind = activeKind();
  const gen  = allSources.find(s => s.id === activeSource)?.generation ?? {};
  _setBtnState('Resuming…', true);
  _resetGenPanel('resuming…');
  log(`↩ Resuming ${kind} from match #${(gen.last_idx ?? -1) + 1} of ${gen.total}`, 'log-info');
  fetchSSE(`/api/resume/statements/${activeSource}`);
}

function startStatements() {
  const kind = activeKind();
  _setBtnState('Running…', true);
  _resetGenPanel('initialising…');
  log(`▶ Starting ${kind} generation…`, 'log-info');
  fetchSSE(`/api/run/statements/${activeSource}`);
}

async function fetchSSE(url) {
  const kind = activeKind();
  try {
    const resp = await fetch(url, { method: 'POST' });
    if (!resp.ok) {
      const txt = await resp.text().catch(() => resp.statusText);
      throw new Error(`HTTP ${resp.status}: ${txt}`);
    }
    const reader = resp.body.getReader();
    const dec    = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const part of parts) {
        const lines = part.trim().split('\n');
        let event = 'message', data = '';
        for (const l of lines) {
          if (l.startsWith('event:')) event = l.slice(6).trim();
          if (l.startsWith('data:'))  data  = l.slice(5).trim();
        }
        if (!data) continue;
        try { handleSSE(event, JSON.parse(data)); } catch (_) {}
      }
    }
  } catch (err) {
    log(`✗ Error: ${err.message}`, 'log-fail');
    document.getElementById('genStatus').textContent = 'error';
    _setBtnState(`Run ${kind}`);
  }
}

function handleSSE(event, data) {
  const kind = activeKind();
  if (event === 'progress') {
    const pct = data.total > 0 ? Math.round((data.idx / data.total) * 100) : 0;
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('progressTxt').textContent  = `${data.idx} / ${data.total} matches`;
    document.getElementById('succCount').textContent    = `${data.successes} ✓`;
    document.getElementById('failCount').textContent    = `${data.failures} ✗`;
    document.getElementById('genStatus').textContent    = `${pct}%`;
    log(
      `[${data.idx}/${data.total}] ${data.status.toUpperCase()} — ${(data.aliases||[]).join(' × ')} (${data.query_count} queries)`,
      data.status === 'success' ? 'log-ok' : 'log-fail'
    );
  }
  if (event === 'done') {
    document.getElementById('genStatus').textContent    = 'complete';
    document.getElementById('progressFill').style.width = '100%';
    log(`✓ Done — ${data.successes} succeeded, ${data.failures} failed`, 'log-ok');
    const flash = document.getElementById('doneFlash');
    flash.textContent   = `✓ ${kind} generation complete — ${data.successes}/${data.total} matches produced queries.`;
    flash.style.display = '';
    _setBtnState(`Re-run ${kind}`);
    loadQueries();
  }
  if (event === 'error') {
    log(`✗ Server error: ${data.message}`, 'log-fail');
    document.getElementById('genStatus').textContent = 'error';
    _setBtnState(`Run ${kind}`);
  }
}

/* ── Queries — all languages ─────────────────────────────────────────────────── */
async function loadQueries() {
  if (!activeSource) return;
  try {
    const res = await fetch(`/api/queries/${activeSource}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    allQueries = Array.isArray(data) ? data : [];
    const section = document.getElementById('queriesSection');
    if (allQueries.length > 0) {
      section.style.display = '';
      populateLangFilter();
      renderQueries();
    } else {
      section.style.display = 'none';
    }
  } catch (e) {
    console.error('loadQueries failed:', e);
    log(`Failed to load queries: ${e.message}`, 'log-fail');
  }
}

function populateLangFilter() {
  const sel     = document.getElementById('langFilter');
  const current = sel.value;
  const langs   = [...new Set(allQueries.map(q => q.language).filter(Boolean))].sort();
  sel.innerHTML = '<option value="">All languages</option>';
  langs.forEach(l => {
    const opt = document.createElement('option');
    opt.value = l; opt.textContent = l;
    if (l === current) opt.selected = true;
    sel.appendChild(opt);
  });
}

function renderQueries() {
  const search = document.getElementById('searchBox').value.toLowerCase();
  const lang   = document.getElementById('langFilter').value;
  const diff   = document.getElementById('diffFilter').value.toLowerCase();
  const respF  = document.getElementById('respFilter').value;

  const filtered = allQueries.filter(q => {
    if (search && !q.question.toLowerCase().includes(search)) return false;
    if (lang   && q.language !== lang) return false;
    if (diff   && (q.difficulty || '').toLowerCase() !== diff) return false;
    if (respF === 'done'    && !q.response) return false;
    if (respF === 'pending' && q.response)  return false;
    return true;
  });

  document.getElementById('qCount').textContent = `${filtered.length} of ${allQueries.length}`;
  const list = document.getElementById('queriesList');
  list.innerHTML = '';

  if (filtered.length === 0) {
    list.innerHTML = '<div class="banner">No queries match your filters.</div>';
    return;
  }

  filtered.forEach(q => {
    const hasResp = !!q.response;
    const hasErr  = !!q.execution_error;
    const d       = (q.difficulty || '').toLowerCase();
    const diffCls = d === 'easy' ? 'tag-easy' : d === 'medium' ? 'tag-medium' : d === 'hard' ? 'tag-hard' : 'tag-plain';
    const tableTags = Object.values(q.tables || {}).map(t => `<span class="tag tag-plain">${esc(t)}</span>`).join('');
    const langTag   = q.language ? `<span class="tag tag-lang">${esc(q.language)}</span>` : '';
    const statusTag = hasResp ? '<span class="tag tag-done">✓ response</span>'
                    : hasErr  ? '<span class="tag tag-error">✗ exec error</span>' : '';
    const playIcon  = hasResp
      ? `<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 7l3 3 5-5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`
      : `<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2.5 1.5l8 4.5-8 4.5V1.5z" fill="currentColor"/></svg>`;

    const div = document.createElement('div');
    div.className = 'qcard';
    div.id = `card-${q.entry_key}-${q.query_id}`;
    div.innerHTML = `
      <div class="qcard-hdr">
        <span class="qcard-num">#${q.entry_key}.${q.query_id}</span>
        <div class="qcard-main">
          <div class="qcard-q">${esc(q.question)}</div>
          <div class="qcard-tags">
            <span class="tag ${diffCls}">${q.difficulty || '—'}</span>
            ${langTag}
            ${tableTags}
            ${statusTag}
          </div>
        </div>
        <button class="btn-play ${hasResp ? 'done-play' : ''}"
          id="playBtn-${q.entry_key}-${q.query_id}"
          title="${hasResp ? 'View result' : 'Run query & generate response'}"
          onclick="runQuery('${q.entry_key}', ${q.query_id})">
          ${playIcon}
        </button>
      </div>`;
    list.appendChild(div);
  });
}

/* ── Single query execution ──────────────────────────────────────────────────── */
async function runQuery(entryKey, queryId) {
  const q           = allQueries.find(x => x.entry_key == entryKey && x.query_id == queryId);
  const hasResponse = !!(q?.response);
  const btn = document.getElementById(`playBtn-${entryKey}-${queryId}`);
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner-sm"></span>`;
  openModalLoading(q, hasResponse ? 'Running query…' : 'Executing query & generating response…');

  const endpoint = hasResponse
    ? `/api/execute/${activeSource}/${entryKey}/${queryId}`
    : `/api/response/${activeSource}/${entryKey}/${queryId}`;

  try {
    const res  = await fetch(endpoint, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (q) {
      if (data.response) q.response = data.response;
      q.execution_error = data.execution_error;
      q.query_code      = data.query_code   || q.query_code;
      q.language        = data.language     || q.language;
      q.query_tables    = data.query_tables || q.query_tables;
    }
    if (hasResponse && !data.response) data.response = q.response;
    if (data.response) {
      btn.classList.add('done-play');
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 7l3 3 5-5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
      const tags = document.getElementById(`card-${entryKey}-${queryId}`)?.querySelector('.qcard-tags');
      if (tags && !tags.querySelector('.tag-done'))
        tags.insertAdjacentHTML('beforeend', '<span class="tag tag-done">✓ response</span>');
    }
    btn.disabled = false;
    openModal(q, data);
  } catch (err) {
    btn.disabled = false;
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2.5 1.5l8 4.5-8 4.5V1.5z" fill="currentColor"/></svg>`;
    document.getElementById('modalBody').innerHTML =
      `<div class="exec-error">Request failed: ${esc(err.message)}</div>`;
  }
}

/* ── Modal ───────────────────────────────────────────────────────────────────── */
function openModalLoading(q, message) {
  document.getElementById('modalBackdrop').style.display = '';
  document.getElementById('resultModal').style.display   = '';
  document.getElementById('modalEyebrow').textContent    = `#${q?.entry_key}.${q?.query_id} · Running`;
  document.getElementById('modalQuestion').textContent   = q?.question ?? '';
  document.getElementById('modalBody').innerHTML =
    `<div class="modal-loading"><div class="spinner"></div><div>${message}</div></div>`;
}

function openModal(q, data) {
  document.getElementById('modalBackdrop').style.display = '';
  document.getElementById('resultModal').style.display   = '';
  const src      = data || q;
  const language = src.language ?? q?.language ?? '';
  const code     = src.query_code ?? q?.query_code ?? '';
  const nlText   = src.response   ?? q?.response   ?? null;
  document.getElementById('modalEyebrow').textContent  =
    `#${src.entry_key ?? q?.entry_key}.${src.query_id ?? q?.query_id}`;
  document.getElementById('modalQuestion').textContent =
    src.question ?? q?.question ?? '';

  let html = '';
  html += `<div class="result-section">
    <div class="result-section-label">Query <span class="lang-badge">${esc(language)}</span></div>`;
  html += code
    ? `<div class="code-block"><pre><code>${esc(formatCode(code, language))}</code></pre></div>`
    : `<div style="font-size:12px;color:var(--text-muted)">Query code not available.</div>`;
  html += `</div>`;

  if (src.execution_error) {
    html += `<div class="result-section">
      <div class="result-section-label">Execution Error</div>
      <div class="exec-error">${esc(src.execution_error)}</div></div>`;
  }

  const cols = src.df_columns ?? [];
  const rows = src.df_rows    ?? [];
  html += `<div class="result-section"><div class="result-section-label">Result DataFrame</div>`;
  if (cols.length > 0) {
    const trunc = (src.df_total_rows ?? 0) > rows.length
      ? `<div class="df-truncated">Showing ${rows.length} of ${src.df_total_rows} rows</div>` : '';
    html += `<div class="df-wrap"><table class="df-table">
      <thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead>
      <tbody>${rows.map(row=>
        `<tr>${row.map(v=>`<td${isNumeric(v)?' class="num"':''}>${esc(v)}</td>`).join('')}</tr>`
      ).join('')}</tbody></table></div>${trunc}`;
  } else if (!src.execution_error) {
    html += `<div style="font-size:13px;color:var(--text-muted)">The query returned no rows.</div>`;
  } else {
    html += `<div style="font-size:13px;color:var(--text-muted)">Could not execute — see error above.</div>`;
  }
  html += `</div>`;

  html += `<div class="result-section"><div class="result-section-label">Natural Language Response</div>`;
  html += nlText
    ? `<div class="nl-response md-body">${marked.parse(nlText)}</div>`
    : `<div style="font-size:13px;color:var(--text-muted)">No response yet — click ▶ to run.</div>`;
  html += `</div>`;

  const queryTables = src.query_tables ?? q?.query_tables ?? [];
  if (queryTables.length > 0) {
    html += `<div class="result-section"><div class="result-section-label">Tables Used</div>
      <div class="table-reasons">`;
    queryTables.forEach(t => {
      const tc = (t.columns_involved || []).join(', ');
      html += `<div class="table-reason-row">
        <div class="table-reason-name">${esc(t.name)}</div>
        <div class="table-reason-body">
          <div class="table-reason-text">${esc(t.reason || '')}</div>
          ${tc ? `<div class="table-reason-cols">${esc(tc)}</div>` : ''}
        </div></div>`;
    });
    html += `</div></div>`;
  }

  document.getElementById('modalBody').innerHTML = html;
}

function closeModal() {
  document.getElementById('modalBackdrop').style.display = 'none';
  document.getElementById('resultModal').style.display   = 'none';
}