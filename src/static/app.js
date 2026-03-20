'use strict';

function formatCode(raw, language) {
  if (!raw) return '';
  const lang = (language || '').toUpperCase();

  // ── PANDAS / Python ──────────────────────────────────────────────────────
  if (lang === 'PANDAS' || lang === 'PYTHON') {
    let out = raw.trim().replace(/\s+/g, ' ');
    // break before chained .method( calls
    out = out.replace(/\.([A-Za-z_]\w*)\(/g, '\n  .$1(');
    // break long argument lists at commas before values
    out = out.replace(/,\s*([A-Za-z_"'\[{])/g, ',\n    $1');
    return out.trim();
  }

  // ── SQL (default) ────────────────────────────────────────────────────────
  const CLAUSES = [
    'UNION ALL','UNION','INTERSECT','EXCEPT',
    'GROUP BY','ORDER BY',
    'WITH','SELECT','FROM','WHERE',
    'HAVING','LIMIT','OFFSET',
  ];
  const JOINS = [
    'INNER JOIN','LEFT OUTER JOIN','RIGHT OUTER JOIN','FULL OUTER JOIN',
    'LEFT JOIN','RIGHT JOIN','FULL JOIN','CROSS JOIN','JOIN',
  ];

  let out = raw.trim().replace(/\s+/g, ' ');

  // Break before clause keywords
  CLAUSES.forEach(function(kw) {
    var pat = kw.replace(/ /g, '\\s+');
    out = out.replace(new RegExp('\\b(' + pat + ')\\b', 'gi'), '\n$1');
  });

  // Break before join keywords and indent them
  JOINS.forEach(function(kw) {
    var pat = kw.replace(/ /g, '\\s+');
    out = out.replace(new RegExp('\\b(' + pat + ')\\b', 'gi'), '\n  $1');
  });

  // Indent AND / OR conditions
  out = out.replace(/\b(AND|OR)\b/gi, '\n  $1');

  // CTE closing paren then SELECT
  out = out.replace(/\)\s*\n\s*(SELECT)/gi, ')\n$1');

  // Clean up empty lines and trailing whitespace
  return out.split('\n')
    .map(function(l) { return l.trimEnd(); })
    .filter(function(l, i) { return i === 0 || l.trim() !== ''; })
    .join('\n')
    .trim();
}

// Backward-compat alias
function formatSQL(raw) { return formatCode(raw, 'SQL'); }

'use strict';

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
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
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

/* ── SQL / code formatter ────────────────────────────────────────────────────
   Inserts line-breaks and indentation before major SQL keywords so the code
   block wraps naturally instead of requiring horizontal scrolling.           */
/* ── Sources ─────────────────────────────────────────────────────────────────── */
async function loadSources() {
  const grid = document.getElementById('sourceGrid');
  grid.innerHTML = '<span style="font-size:12px;color:var(--text-muted)">Loading sources…</span>';
  try {
    const res = await fetch('/api/sources');
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
    const qLine  = qCount > 0
      ? `<div class="source-q-count">↳ ${qCount} queries generated</div>`
      : '';

    card.innerHTML = `
      <div class="source-name">${esc(src.label)}</div>
      <div class="source-type">${esc(src.type)}</div>
      ${badgeHtml(src)}
      ${qLine}
    `;
    grid.appendChild(card);
  });
}

function badgeHtml(src) {
  if (src.error) return `<span class="source-badge badge-miss"><span class="dot"></span>Config error</span>`;
  const missing = (src.files || []).filter(f => !f.exists).length;
  if (missing === 0)              return `<span class="source-badge badge-ready"><span class="dot"></span>Ready</span>`;
  if (missing < src.files.length) return `<span class="source-badge badge-warn"><span class="dot"></span>${missing} file${missing>1?'s':''} missing</span>`;
  return `<span class="source-badge badge-miss"><span class="dot"></span>Not configured</span>`;
}

async function selectSource(id) {
  activeSource = id;
  renderSourceGrid();

  const src = allSources.find(s => s.id === id);
  document.getElementById('workArea').style.display = '';
  renderReadiness(src);
  updateRunButton(src);
  updateTopbarMeta();

  // Reset panels
  document.getElementById('genPanel').style.display  = 'none';
  document.getElementById('doneFlash').style.display = 'none';
  document.getElementById('queriesSection').style.display = 'none';
  allQueries = [];

  // ── Banner logic ─────────────────────────────────────────────────────────
  const stmts      = src.statements ?? { exists: false, query_count: 0 };
  const generation = src.generation ?? {};
  const banner     = document.getElementById('existsBanner');
  const runRow     = document.getElementById('runRow');

  // Languages with some matches done but not all
  const interrupted = Object.entries(generation)
    .filter(([, g]) => g.done_count > 0 && !g.is_complete)
    .sort(([, a], [, b]) => b.done_count - a.done_count);

  try {
    if (interrupted.length > 0) {
      setBanner('interrupted', interrupted, stmts);
      banner.style.display = '';
      runRow.style.display = 'none';
    } else if (stmts.exists) {
      setBanner('complete', [], stmts);
      banner.style.display = '';
      runRow.style.display = 'none';
    } else {
      banner.style.display = 'none';
      runRow.style.display = '';
    }
  } catch (e) {
    console.error('Banner setup error:', e);
    banner.style.display = 'none';
    runRow.style.display = '';
  }

  // Always load queries — runs regardless of banner errors
  await loadQueries();
}

/* ── Readiness ───────────────────────────────────────────────────────────────── */
function renderReadiness(src) {
  const panel = document.getElementById('readinessPanel');
  if (src.error) {
    panel.innerHTML = `<div class="readiness-row">
      <span class="r-icon">⚠</span>
      <span class="r-name">Config error</span>
      <span class="r-path">${esc(src.error)}</span>
      <span class="r-status r-miss">Error</span>
    </div>`;
    return;
  }
  panel.innerHTML = (src.files || []).map(f => `
    <div class="readiness-row">
      <span class="r-icon">${f.exists ? '✓' : '✗'}</span>
      <span class="r-name">${esc(f.name)}</span>
      <span class="r-path" title="${esc(f.path)}">${esc(f.path)}</span>
      <span class="r-status ${f.exists ? 'r-ok' : 'r-miss'}">${f.exists ? 'Found' : 'Missing'}</span>
    </div>
  `).join('');
}

function updateRunButton(src) {
  const btn  = document.getElementById('runBtn');
  const hint = document.getElementById('runHint');
  const ready = !src.error && (src.files || []).every(f => f.exists);
  btn.disabled = !ready;
  if (src.error) {
    hint.textContent = 'Config could not be loaded for this source.';
  } else if (!ready) {
    const missing = (src.files || []).filter(f => !f.exists).map(f => f.name).join(', ');
    hint.textContent = `Missing: ${missing}`;
  } else {
    hint.textContent = `${src.files.length} required files verified — ready to run.`;
  }
}

function updateTopbarMeta() {
  const el = document.getElementById('topbarMeta');
  if (!activeSource) { el.textContent = '—'; return; }
  const src = allSources.find(s => s.id === activeSource);
  el.textContent = src ? `${src.label} · ${src.type}` : '—';
}

/* ── Banner rendering ────────────────────────────────────────────────────────── */
const _ICON_OK = `<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
  <circle cx="10" cy="10" r="9" stroke="var(--ok)" stroke-width="1.5"/>
  <path d="M6 10l3 3 5-5" stroke="var(--ok)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>`;

const _ICON_WARN = `<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
  <circle cx="10" cy="10" r="9" stroke="var(--warn)" stroke-width="1.5"/>
  <path d="M10 6v5" stroke="var(--warn)" stroke-width="1.5" stroke-linecap="round"/>
  <circle cx="10" cy="14.5" r="1" fill="var(--warn)"/>
</svg>`;

function setBanner(state, interrupted, stmts) {
  const banner  = document.getElementById('existsBanner');
  const iconEl  = document.getElementById('existsIcon');
  const titleEl = document.getElementById('existsTitle');
  const subEl   = document.getElementById('existsSub');
  const actEl   = document.getElementById('existsActions');

  if (!banner || !actEl) return;   // guard against stale/old HTML

  banner.dataset.state = state;
  actEl.innerHTML = '';

  if (state === 'interrupted') {
    iconEl.innerHTML  = _ICON_WARN;
    titleEl.textContent = 'Generation incomplete — matches still pending';

    const lines = interrupted.map(([kind, g]) =>
      `${kind}: ${g.done_count} of ${g.total} processed (last index: #${g.last_idx})`
    );
    subEl.textContent = lines.join(' · ');

    // One resume button per interrupted language
    interrupted.forEach(([kind, g]) => {
      const btn = document.createElement('button');
      btn.className   = 'btn btn-resume';
      btn.title       = `Pick up at match #${g.last_idx + 1}`;
      btn.innerHTML   = `↩ Resume ${kind}`;
      btn.onclick     = () => resumeStatements(kind);
      actEl.appendChild(btn);
    });

    // Re-generate with confirm dialog
    const regenBtn = document.createElement('button');
    regenBtn.className = 'btn btn-ghost';
    regenBtn.textContent = 'Re-generate from scratch';
    regenBtn.onclick = () => askRegenerate(interrupted.map(([k]) => k));
    actEl.appendChild(regenBtn);

    if (stmts.exists) {
      const viewBtn = document.createElement('button');
      viewBtn.className = 'btn btn-outline';
      viewBtn.textContent = 'View existing queries';
      viewBtn.onclick = viewExisting;
      actEl.appendChild(viewBtn);
    }

  } else {
    // complete
    iconEl.innerHTML  = _ICON_OK;
    titleEl.textContent = `${stmts.query_count} quer${stmts.query_count === 1 ? 'y' : 'ies'} generated`;
    subEl.textContent   = 'All matches processed. View queries or overwrite by re-generating.';

    const viewBtn = document.createElement('button');
    viewBtn.className = 'btn btn-outline';
    viewBtn.textContent = 'View queries';
    viewBtn.onclick = viewExisting;
    actEl.appendChild(viewBtn);

    const regenBtn = document.createElement('button');
    regenBtn.className = 'btn btn-ghost';
    regenBtn.textContent = 'Re-generate from scratch';
    regenBtn.onclick = () => askRegenerate([]);
    actEl.appendChild(regenBtn);
  }
}

/* ── Confirm dialog ──────────────────────────────────────────────────────────── */
function askRegenerate(affectedKinds) {
  const body = document.getElementById('confirmBody');
  body.textContent = affectedKinds.length > 0
    ? `This will overwrite all existing results for ${affectedKinds.join(', ')} and restart from match #0. Any generated queries will be lost.`
    : 'This will overwrite all existing results and restart from match #0. Any generated queries will be lost.';
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

/* ── Existing-statements actions ─────────────────────────────────────────────── */
function viewExisting() {
  document.getElementById('existsBanner').style.display = 'none';
  const qs = document.getElementById('queriesSection');
  if (qs.style.display === 'none') {
    loadQueries();
  } else {
    qs.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function showGenPanel() {
  document.getElementById('existsBanner').style.display = 'none';
  document.getElementById('runRow').style.display = '';
  document.getElementById('runRow').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/* ── Statement generation via SSE ────────────────────────────────────────────── */
function resumeStatements(kind) {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;

  document.getElementById('existsBanner').style.display = 'none';
  document.getElementById('runRow').style.display = '';
  document.getElementById('genPanel').style.display = '';
  document.getElementById('doneFlash').style.display = 'none';
  document.getElementById('logBox').innerHTML = '';
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('progressTxt').textContent = 'Resuming…';
  document.getElementById('succCount').textContent = '0 ✓';
  document.getElementById('failCount').textContent = '0 ✗';
  document.getElementById('genStatus').textContent = 'resuming…';

  const src = allSources.find(s => s.id === activeSource);
  const g   = src?.generation?.[kind] ?? {};
  log(`↩ Resuming ${kind} from match #${g.last_idx + 1} of ${g.total}`, 'log-info');

  fetchSSE(`/api/resume/statements/${activeSource}?kind=${encodeURIComponent(kind)}`);
}

function startStatements() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner-sm"></span> Running…`;

  document.getElementById('genPanel').style.display = '';
  document.getElementById('doneFlash').style.display = 'none';
  document.getElementById('logBox').innerHTML = '';
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('progressTxt').textContent = '0 / 0 matches';
  document.getElementById('succCount').textContent = '0 ✓';
  document.getElementById('failCount').textContent = '0 ✗';
  document.getElementById('genStatus').textContent = 'initialising…';

  fetchSSE(`/api/run/statements/${activeSource}?kind=PANDAS`);
}

async function fetchSSE(url) {
  const resp = await fetch(url, { method: 'POST' });
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
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
}

function handleSSE(event, data) {
  if (event === 'progress') {
    const pct = Math.round((data.idx / data.total) * 100);
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('progressTxt').textContent  = `${data.idx} / ${data.total} matches`;
    document.getElementById('succCount').textContent    = `${data.successes} ✓`;
    document.getElementById('failCount').textContent    = `${data.failures} ✗`;
    document.getElementById('genStatus').textContent    = `${pct}%`;
    const cls = data.status === 'success' ? 'log-ok' : 'log-fail';
    log(`[${data.idx}/${data.total}] ${data.status.toUpperCase()} — ${(data.aliases||[]).join(' × ')} (${data.query_count} queries)`, cls);
  }

  if (event === 'done') {
    document.getElementById('genStatus').textContent = 'complete';
    document.getElementById('progressFill').style.width = '100%';
    log(`✓ Done — ${data.successes} succeeded, ${data.failures} failed`, 'log-ok');
    const flash = document.getElementById('doneFlash');
    flash.textContent = `✓ Statement generation complete — ${data.successes}/${data.total} matches produced queries.`;
    flash.style.display = '';
    const btn = document.getElementById('runBtn');
    btn.disabled = false;
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2.5 1.5l8 4.5-8 4.5V1.5z" fill="currentColor"/></svg> Re-run`;
    loadQueries();
  }

  if (event === 'error') {
    log(`✗ Error: ${data.message}`, 'log-fail');
    document.getElementById('genStatus').textContent = 'error';
    const btn = document.getElementById('runBtn');
    btn.disabled = false;
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2.5 1.5l8 4.5-8 4.5V1.5z" fill="currentColor"/></svg> Run Statements`;
  }
}

/* ── Queries ─────────────────────────────────────────────────────────────────── */
async function loadQueries() {
  if (!activeSource) return;
  try {
    const res = await fetch(`/api/queries/${activeSource}`);
    allQueries = await res.json();
    if (allQueries.length > 0) {
      document.getElementById('queriesSection').style.display = '';
      populateLangFilter();
      renderQueries();
    }
  } catch (_) {}
}

function populateLangFilter() {
  const sel    = document.getElementById('langFilter');
  const current = sel.value;
  const langs  = [...new Set(allQueries.map(q => q.language).filter(Boolean))].sort();
  // Rebuild options, preserving the "All languages" default
  sel.innerHTML = '<option value="">All languages</option>';
  langs.forEach(l => {
    const opt = document.createElement('option');
    opt.value = l;
    opt.textContent = l;
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
    const diff    = (q.difficulty || '').toLowerCase();
    const diffCls = diff === 'easy' ? 'tag-easy' : diff === 'medium' ? 'tag-medium' : diff === 'hard' ? 'tag-hard' : 'tag-plain';

    const tableTags  = Object.values(q.tables || {}).map(t => `<span class="tag tag-plain">${esc(t)}</span>`).join('');
    const statusTag  = hasResp ? '<span class="tag tag-done">✓ response</span>'
                     : hasErr  ? '<span class="tag tag-error">✗ exec error</span>' : '';

    // Play button SVG (checkmark if done, play if not)
    const playIcon = hasResp
      ? `<svg width="14" height="14" viewBox="0 0 14 14" fill="none">
           <path d="M3 7l3 3 5-5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
         </svg>`
      : `<svg width="12" height="12" viewBox="0 0 12 12" fill="none">
           <path d="M2.5 1.5l8 4.5-8 4.5V1.5z" fill="currentColor"/>
         </svg>`;

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
            ${tableTags}
            ${statusTag}
          </div>
        </div>
        <button
          class="btn-play ${hasResp ? 'done-play' : ''}"
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
  const hasResponse = !!(q?.response);   // already generated → execute-only

  const btn = document.getElementById(`playBtn-${entryKey}-${queryId}`);
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner-sm"></span>`;

  // Show modal immediately with appropriate loading label
  openModalLoading(q, hasResponse ? 'Running query…' : 'Executing query & generating response…');

  const endpoint = hasResponse
    ? `/api/execute/${activeSource}/${entryKey}/${queryId}`   // df only, no LLM
    : `/api/response/${activeSource}/${entryKey}/${queryId}`; // df + NL

  try {
    const res  = await fetch(endpoint, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Update local query cache
    if (q) {
      if (data.response) q.response = data.response;
      q.execution_error = data.execution_error;
      q.query_code      = data.query_code   || q.query_code;
      q.language        = data.language     || q.language;
      q.query_tables    = data.query_tables || q.query_tables;
    }
    // Execute-only: inject the cached NL response so the modal still shows it
    if (hasResponse && !data.response) data.response = q.response;

    // Update play button
    if (data.response) {
      btn.classList.add('done-play');
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <path d="M3 7l3 3 5-5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>`;
      // Update tags
      const card = document.getElementById(`card-${entryKey}-${queryId}`);
      const tags = card?.querySelector('.qcard-tags');
      if (tags && !tags.querySelector('.tag-done')) {
        tags.insertAdjacentHTML('beforeend', '<span class="tag tag-done">✓ response</span>');
      }
    }

    btn.disabled = false;

    // Populate and show modal with result
    openModal(q, data);

  } catch (err) {
    btn.disabled = false;
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none">
      <path d="M2.5 1.5l8 4.5-8 4.5V1.5z" fill="currentColor"/>
    </svg>`;
    document.getElementById('modalBody').innerHTML =
      `<div class="exec-error">Request failed: ${esc(err.message)}</div>`;
  }
}

/* ── Modal ───────────────────────────────────────────────────────────────────── */
function openModalLoading(q, message = 'Executing query and generating response…') {
  document.getElementById('modalBackdrop').style.display = '';
  document.getElementById('resultModal').style.display   = '';
  document.getElementById('modalEyebrow').textContent   = `#${q?.entry_key}.${q?.query_id} · Running`;
  document.getElementById('modalQuestion').textContent  = q?.question ?? '';
  document.getElementById('modalBody').innerHTML = `
    <div class="modal-loading">
      <div class="spinner"></div>
      <div>${message}</div>
    </div>`;
}

function openModal(q, data) {
  document.getElementById('modalBackdrop').style.display = '';
  document.getElementById('resultModal').style.display   = '';

  // data = fresh API result; q = cached query list item
  const src      = data || q;
  const language = src.language ?? q?.language ?? 'PANDAS';
  const code     = src.query_code ?? q?.query_code ?? '';
  const nlText   = src.response ?? q?.response ?? null;

  document.getElementById('modalEyebrow').textContent  =
    `#${src.entry_key ?? q?.entry_key}.${src.query_id ?? q?.query_id}`;
  document.getElementById('modalQuestion').textContent =
    src.question ?? q?.question ?? '';

  let html = '';

  // ── Language + query code ──────────────────────────────────────────────────
  html += `<div class="result-section">
    <div class="result-section-label">
      Query
      <span class="lang-badge">${esc(language)}</span>
    </div>`;

  if (code) {
    const formatted = formatCode(code, language);
    html += `<div class="code-block"><pre><code>${esc(formatted)}</code></pre></div>`;
  } else {
    html += `<div style="font-size:12px;color:var(--text-muted)">Query code not available.</div>`;
  }
  html += `</div>`;

  // ── Execution error ────────────────────────────────────────────────────────
  if (src.execution_error) {
    html += `<div class="result-section">
      <div class="result-section-label">Execution Error</div>
      <div class="exec-error">${esc(src.execution_error)}</div>
    </div>`;
  }

  // ── DataFrame ──────────────────────────────────────────────────────────────
  const cols = src.df_columns ?? [];
  const rows = src.df_rows    ?? [];

  html += `<div class="result-section"><div class="result-section-label">Result DataFrame</div>`;
  if (cols.length > 0) {
    const truncNote = (src.df_total_rows ?? 0) > rows.length
      ? `<div class="df-truncated">Showing ${rows.length} of ${src.df_total_rows} rows</div>` : '';
    const headerCells = cols.map(c => `<th>${esc(c)}</th>`).join('');
    const bodyRows    = rows.map(row =>
      `<tr>${row.map(v => `<td${isNumeric(v) ? ' class="num"' : ''}>${esc(v)}</td>`).join('')}</tr>`
    ).join('');
    html += `<div class="df-wrap">
      <table class="df-table">
        <thead><tr>${headerCells}</tr></thead>
        <tbody>${bodyRows}</tbody>
      </table>
    </div>${truncNote}`;
  } else if (!src.execution_error) {
    html += `<div style="font-size:13px;color:var(--text-muted)">The query returned no rows.</div>`;
  } else {
    html += `<div style="font-size:13px;color:var(--text-muted)">Could not execute — see error above.</div>`;
  }
  html += `</div>`;

  // ── NL Response ────────────────────────────────────────────────────────────
  html += `<div class="result-section"><div class="result-section-label">Natural Language Response</div>`;
  if (nlText) {
    html += `<div class="nl-response md-body">${marked.parse(nlText)}</div>`;
  } else {
    html += `<div style="font-size:13px;color:var(--text-muted)">
      No response generated yet — click ▶ to run this query.
    </div>`;
  }
  html += `</div>`;

  // ── Table reasons ──────────────────────────────────────────────────────────
  const queryTables = src.query_tables ?? q?.query_tables ?? [];
  if (queryTables.length > 0) {
    html += `<div class="result-section"><div class="result-section-label">Tables Used</div>
      <div class="table-reasons">`;
    queryTables.forEach(t => {
      const cols = (t.columns_involved || []).join(', ');
      html += `<div class="table-reason-row">
        <div class="table-reason-name">${esc(t.name)}</div>
        <div class="table-reason-body">
          <div class="table-reason-text">${esc(t.reason || '')}</div>
          ${cols ? `<div class="table-reason-cols">${esc(cols)}</div>` : ''}
        </div>
      </div>`;
    });
    html += `</div></div>`;
  }

  document.getElementById('modalBody').innerHTML = html;
}

function closeModal() {
  document.getElementById('modalBackdrop').style.display = 'none';
  document.getElementById('resultModal').style.display   = 'none';
}