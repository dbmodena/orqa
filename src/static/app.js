'use strict';

/* ═══════════════════════════════════════════════════════════════════════════
   OrQA Query Browser & Executor
   Reads generated queries (with their full plan/code judge histories) and
   re-executes them on demand. No generation controls — read + run only.
   ═══════════════════════════════════════════════════════════════════════════ */

/* ── Code formatter ─────────────────────────────────────────────────────── */
function formatCode(raw, kind) {
  if (!raw) return '';
  const k = (kind || '').toUpperCase();
  if (k === 'PANDAS' || k === 'PYTHON') return raw.trim();
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

/* ── State ──────────────────────────────────────────────────────────────── */
let allSources   = [];
let allQueries   = [];
let activeSource = null;

/* ── Boot ───────────────────────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', async () => {
  await loadSources();
  updateTopbarMeta();
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

/* ── Utilities ──────────────────────────────────────────────────────────── */
function esc(s) {
  return String(s ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function isNumeric(v) { return v !== null && v !== '' && !isNaN(Number(v)); }
function judgeName(raw) { return String(raw || 'judge').replace(/^oci\//, ''); }

/* ── Icons ──────────────────────────────────────────────────────────────── */
const ICON_CHECK = `<svg viewBox="0 0 14 14" fill="none" width="12" height="12">
  <path d="M2.6 7.4l3 3 5.8-6" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const ICON_CROSS = `<svg viewBox="0 0 14 14" fill="none" width="12" height="12">
  <path d="M3.6 3.6l6.8 6.8M10.4 3.6l-6.8 6.8" stroke="currentColor" stroke-width="2.1" stroke-linecap="round"/></svg>`;
const ICON_PLAY = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none">
  <path d="M2.5 1.5l8 4.5-8 4.5V1.5z" fill="currentColor"/></svg>`;
const ICON_BOLT = `<svg viewBox="0 0 14 14" fill="none" width="12" height="12">
  <path d="M7.8 1L3 8h3l-.8 5L10 6H7l.8-5z" fill="currentColor"/></svg>`;

function robotSvg() {
  return `<svg class="robot-svg" viewBox="0 0 40 40" fill="none">
    <line x1="20" y1="4" x2="20" y2="9" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
    <circle cx="20" cy="3.2" r="2" fill="currentColor"/>
    <rect x="7" y="9" width="26" height="20" rx="6" stroke="currentColor" stroke-width="2.2" fill="none"/>
    <circle cx="14.5" cy="18" r="2.6" fill="currentColor"/>
    <circle cx="25.5" cy="18" r="2.6" fill="currentColor"/>
    <path d="M14.5 24.5c1.6 1.6 4 2.4 5.5 2.4s3.9-.8 5.5-2.4" stroke="currentColor" stroke-width="2" stroke-linecap="round" fill="none"/>
    <line x1="3.5" y1="16" x2="7" y2="16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <line x1="33" y1="16" x2="36.5" y2="16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <line x1="3.5" y1="16" x2="3.5" y2="22" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <line x1="36.5" y1="16" x2="36.5" y2="22" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <rect x="13" y="32" width="14" height="4.5" rx="2.2" stroke="currentColor" stroke-width="1.8" fill="none"/>
  </svg>`;
}
function robotSadSvg() {
  return `<svg class="robot-svg" viewBox="0 0 40 40" fill="none">
    <line x1="20" y1="4" x2="20" y2="9" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
    <circle cx="20" cy="3.2" r="2" fill="currentColor"/>
    <rect x="7" y="9" width="26" height="20" rx="6" stroke="currentColor" stroke-width="2.2" fill="none"/>
    <path d="M12 16.5l5 2M28 16.5l-5 2" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
    <path d="M14.5 26.9c1.6-1.6 4-2.4 5.5-2.4s3.9 .8 5.5 2.4" stroke="currentColor" stroke-width="2" stroke-linecap="round" fill="none"/>
    <line x1="3.5" y1="16" x2="7" y2="16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <line x1="33" y1="16" x2="36.5" y2="16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <line x1="3.5" y1="16" x2="3.5" y2="22" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <line x1="36.5" y1="16" x2="36.5" y2="22" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <rect x="13" y="32" width="14" height="4.5" rx="2.2" stroke="currentColor" stroke-width="1.8" fill="none"/>
  </svg>`;
}

/* ── Sources ────────────────────────────────────────────────────────────── */
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
    card.onclick = () => selectSource(src.id);
    const qCount = src.statements?.query_count ?? 0;
    const badge = src.error
      ? `<span class="source-badge badge-miss"><span class="dot"></span>Config error</span>`
      : qCount > 0
        ? `<span class="source-badge badge-ready"><span class="dot"></span>${qCount} queries</span>`
        : `<span class="source-badge badge-warn"><span class="dot"></span>No queries</span>`;
    card.innerHTML = `
      <div class="source-name">${esc(src.label)}</div>
      <div class="source-type">${esc(src.type)}</div>
      ${badge}`;
    grid.appendChild(card);
  });
}

async function selectSource(id) {
  activeSource = id;
  renderSourceGrid();
  updateTopbarMeta();
  document.getElementById('workArea').style.display = '';
  await loadQueries();
}

function updateTopbarMeta() {
  const el = document.getElementById('topbarMeta');
  if (!activeSource) { el.textContent = '—'; return; }
  const src = allSources.find(s => s.id === activeSource);
  el.textContent = src ? `${src.label} · ${src.type}` : '—';
}

/* ── Queries ────────────────────────────────────────────────────────────── */
async function loadQueries() {
  if (!activeSource) return;
  const section = document.getElementById('queriesSection');
  const empty   = document.getElementById('emptyState');
  try {
    const res = await fetch(`/api/queries/${activeSource}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    allQueries = await res.json();
    if (allQueries.length > 0) {
      section.style.display = '';
      empty.style.display   = 'none';
      populateKindFilter();
      renderQueries();
    } else {
      section.style.display = 'none';
      empty.style.display   = '';
    }
  } catch (e) {
    section.style.display = 'none';
    empty.style.display   = '';
    empty.textContent     = `Failed to load queries: ${e.message}`;
  }
}

function populateKindFilter() {
  const sel     = document.getElementById('kindFilter');
  const current = sel.value;
  const kinds   = [...new Set(allQueries.map(q => q.kind).filter(Boolean))].sort();
  sel.innerHTML = '<option value="">All kinds</option>';
  kinds.forEach(k => {
    const opt = document.createElement('option');
    opt.value = k; opt.textContent = k;
    if (k === current) opt.selected = true;
    sel.appendChild(opt);
  });
}

function queryKey(q) { return `${q.kind}|${q.entry_key}|${q.qnum}`; }

function renderQueries() {
  const search = document.getElementById('searchBox').value.toLowerCase();
  const kind   = document.getElementById('kindFilter').value;
  const skill  = document.getElementById('skillFilter').value;
  const diff   = document.getElementById('diffFilter').value.toLowerCase();

  const filtered = allQueries.filter(q => {
    if (search && !q.question.toLowerCase().includes(search)) return false;
    if (kind && q.kind !== kind) return false;
    if (skill === '__plain__' && (q.task_types || []).length > 0) return false;
    if (skill && skill !== '__plain__' && !(q.task_types || []).includes(skill)) return false;
    if (diff && (q.difficulty || '').toLowerCase() !== diff) return false;
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
    const d       = (q.difficulty || '').toLowerCase();
    const diffCls = d === 'easy' ? 'tag-easy' : d === 'medium' ? 'tag-medium' : d === 'hard' ? 'tag-hard' : 'tag-plain';
    const skillTags = (q.task_types || [])
      .map(t => `<span class="tag tag-skill">${ICON_BOLT} ${esc(t)}</span>`).join('');
    const resultTag = q.expected_result_type
      ? `<span class="tag tag-restype">→ ${esc(q.expected_result_type)}</span>` : '';
    const approved  = q.status === 'approved';
    const statusTag = q.status
      ? `<span class="tag ${approved ? 'tag-done' : 'tag-error'}">${approved ? '✓' : '✗'} ${esc(q.status)}</span>`
      : '';

    const div = document.createElement('div');
    div.className = 'qcard';
    div.onclick = () => openDetail(q);
    div.innerHTML = `
      <div class="qcard-hdr">
        <span class="qcard-num">#${esc(q.entry_key)}.${esc(q.qnum)}</span>
        <div class="qcard-main">
          <div class="qcard-q">${esc(q.question)}</div>
          <div class="qcard-tags">
            <span class="tag tag-lang">${esc(q.kind)}</span>
            <span class="tag ${diffCls}">${esc(q.difficulty || '—')}</span>
            ${skillTags}
            ${resultTag}
            ${statusTag}
          </div>
        </div>
        <span class="qcard-open">›</span>
      </div>`;
    list.appendChild(div);
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   Detail modal
   ═══════════════════════════════════════════════════════════════════════════ */

function voteChip(label, ok) {
  if (ok === null || ok === undefined) return '';
  return `<span class="vote-chip ${ok ? 'vote-ok' : 'vote-bad'}">
    ${ok ? ICON_CHECK : ICON_CROSS}<span>${esc(label)}</span></span>`;
}

const PLAN_LAYERS = [
  ['question_approval',        'question'],
  ['plan_approval',            'plan'],
  ['table_usage_approval',     'tables'],
  ['skill_approval',           'skill'],
  ['predictor_approval',       'predictors'],
  ['expected_result_approval', 'result-type'],
];

function robotCard(vote, layers) {
  const ok = vote.approved !== false && !vote.error;
  const chips = (layers || [])
    .filter(([field]) => field in vote)
    .map(([field, label]) => voteChip(label, vote[field]))
    .join('');
  const violated = (vote.violated_criteria || [])
    .map(c => `<span class="vote-chip vote-bad">${ICON_CROSS}<span>${esc(c)}</span></span>`).join('');
  const fb  = vote.error ? `⚡ ${vote.error}` : (vote.feedback || '');
  const sug = vote.suggestions ? `<div class="robot-suggestion">💡 ${esc(vote.suggestions)}</div>` : '';
  return `
    <div class="robot-card ${ok ? 'robot-ok' : 'robot-bad'}">
      <div class="robot-head">
        <span class="robot-avatar">${ok ? robotSvg() : robotSadSvg()}</span>
        <div class="robot-id">
          <span class="robot-name">${esc(judgeName(vote.judge))}</span>
          <span class="robot-verdict ${ok ? 'v-ok' : 'v-bad'}">${ok ? ICON_CHECK : ICON_CROSS} ${ok ? 'approved' : 'rejected'}</span>
        </div>
      </div>
      ${chips || violated ? `<div class="robot-votes">${chips}${violated}</div>` : ''}
      ${fb ? `<div class="robot-feedback">${esc(fb)}</div>` : ''}
      ${sug}
    </div>`;
}

function panelBlock(panel, layers) {
  if (!panel || !(panel.votes || []).length) return '';
  const tally = `
    <span class="panel-tally">
      <span class="t-ok">${panel.approve_votes ?? 0} ${ICON_CHECK}</span>
      <span class="t-bad">${panel.reject_votes ?? 0} ${ICON_CROSS}</span>
      ${panel.failed_votes ? `<span class="t-fail">${panel.failed_votes} ⚡</span>` : ''}
    </span>`;
  const robots = panel.votes.map(v => robotCard(v, layers)).join('');
  return `
    <div class="panel-block">
      <div class="panel-block-hdr">${robotSvg()} Judge panel ${tally}</div>
      <div class="robot-grid">${robots}</div>
    </div>`;
}

function attemptSummaryBadge(ok) {
  return `<span class="attempt-badge ${ok ? 'ab-ok' : 'ab-bad'}">${ok ? ICON_CHECK : ICON_CROSS} ${ok ? 'approved' : 'rejected'}</span>`;
}

/* Natural-language responses: each code judge's own business insight for
   the (approved) query, shown per model with its robot avatar. Taken from
   the LAST attempt whose panel carries any non-empty response. */
function judgeResponsesHtml(q) {
  const history = q.attempt_history || [];
  let responses = [];
  for (let i = history.length - 1; i >= 0 && !responses.length; i--) {
    responses = ((history[i].panel || {}).votes || [])
      .filter(v => (v.response || '').trim())
      .map(v => ({ judge: judgeName(v.judge), text: v.response }));
  }
  if (!responses.length) return '';
  const cards = responses.map(r => `
    <div class="resp-card">
      <div class="resp-head">
        <span class="resp-avatar">${robotSvg()}</span>
        <span class="resp-model">${esc(r.judge)}</span>
      </div>
      <div class="resp-body md-body">${marked.parse(String(r.text))}</div>
    </div>`).join('');
  return `
    <div class="result-section">
      <div class="result-section-label loop-label">
        ${robotSvg()} Judge Responses
        <span class="loop-count">${responses.length} model${responses.length === 1 ? '' : 's'}</span>
      </div>
      <div class="resp-grid">${cards}</div>
    </div>`;
}

/* Plan-judge loop: one collapsible per attempt */
function planLoopHtml(q) {
  const attempts = q.plan_attempts || [];
  if (!attempts.length) return '';
  const items = attempts.map((att, i) => {
    const ok = !!att.approved;
    const open = i === attempts.length - 1 ? ' open' : '';
    const layerChips = PLAN_LAYERS
      .filter(([f]) => f in att)
      .map(([f, label]) => voteChip(label, att[f])).join('');
    return `
      <details class="collapse attempt${open}">
        <summary>
          <span class="collapse-arrow">▸</span>
          <span class="attempt-title">Attempt ${esc(att.attempt ?? i + 1)}</span>
          ${attemptSummaryBadge(ok)}
          <span class="attempt-summary">${esc(att.summary || '')}</span>
        </summary>
        <div class="collapse-body">
          ${att.question && att.question !== q.question
            ? `<div class="attempt-q"><span class="attempt-q-label">question at this round</span>${esc(att.question)}</div>` : ''}
          ${layerChips ? `<div class="attempt-layers">${layerChips}</div>` : ''}
          ${panelBlock(att.panel, PLAN_LAYERS)}
          ${att.feedback ? `<div class="attempt-feedback"><span class="fb-label">panel feedback</span>${esc(att.feedback)}</div>` : ''}
          ${att.suggestions ? `<div class="robot-suggestion">💡 ${esc(att.suggestions)}</div>` : ''}
        </div>
      </details>`;
  }).join('');
  return `
    <div class="result-section">
      <div class="result-section-label loop-label">
        ${robotSvg()} Plan Judge Loop
        <span class="loop-count">${attempts.length} attempt${attempts.length === 1 ? '' : 's'}</span>
      </div>
      <div class="loop-track">${items}</div>
    </div>`;
}

/* Code loop (validation + code judges): one collapsible per attempt */
function codeLoopHtml(q) {
  const history = q.attempt_history || [];
  if (!history.length) return '';
  const items = history.map((att, i) => {
    const outcome = String(att.outcome || '').toLowerCase();
    const ok = outcome === 'approved';
    const open = i === history.length - 1 ? ' open' : '';
    const stage = att.stage ? `<span class="stage-badge stage-${esc(att.stage)}">${esc(att.stage)}</span>` : '';
    return `
      <details class="collapse attempt${open}">
        <summary>
          <span class="collapse-arrow">▸</span>
          <span class="attempt-title">Attempt ${esc(att.attempt ?? i + 1)}</span>
          ${stage}
          ${attemptSummaryBadge(ok)}
          <span class="attempt-summary">${esc(att.summary || att.detail || '')}</span>
        </summary>
        <div class="collapse-body">
          ${att.detail && att.summary ? `<div class="attempt-feedback"><span class="fb-label">detail</span>${esc(att.detail)}</div>` : ''}
          ${panelBlock(att.panel, [])}
          ${att.proposed_code ? `
            <details class="collapse code-collapse">
              <summary><span class="collapse-arrow">▸</span>code at this attempt</summary>
              <div class="collapse-body"><div class="code-block"><pre><code>${esc(att.proposed_code)}</code></pre></div></div>
            </details>` : ''}
        </div>
      </details>`;
  }).join('');
  return `
    <div class="result-section">
      <div class="result-section-label loop-label">
        ${robotSvg()} Code Judge Loop
        <span class="loop-count">${history.length} attempt${history.length === 1 ? '' : 's'}</span>
      </div>
      <div class="loop-track">${items}</div>
    </div>`;
}

function openDetail(q) {
  document.getElementById('modalBackdrop').style.display = '';
  document.getElementById('resultModal').style.display   = '';
  document.getElementById('modalEyebrow').textContent =
    `${q.kind} · #${q.entry_key}.${q.qnum}${q.model ? ' · ' + q.model : ''}`;
  document.getElementById('modalQuestion').textContent = q.question || '';

  let html = '';

  // Meta chips
  const skillTags = (q.task_types || [])
    .map(t => `<span class="tag tag-skill">${ICON_BOLT} ${esc(t)}</span>`).join('');
  html += `<div class="detail-chips">
    <span class="tag tag-lang">${esc(q.kind)}</span>
    <span class="tag tag-plain">${esc(q.difficulty || '—')}</span>
    ${skillTags}
    ${Object.values(q.tables_map || {}).map(t => `<span class="tag tag-plain">${esc(t)}</span>`).join('')}
  </div>`;

  // Expected result contract
  if (q.expected_result_type || q.expected_result_description) {
    html += `<div class="contract-box">
      <span class="contract-type">→ ${esc(q.expected_result_type || 'table')}</span>
      <span class="contract-desc">${esc(q.expected_result_description || '')}</span>
    </div>`;
  }

  // Code + run
  html += `<div class="result-section">
    <div class="result-section-label code-label">
      Code
      <button class="btn btn-run btn-exec" id="execBtn" onclick='execQuery(${JSON.stringify(queryKey(q))})'>
        ${ICON_PLAY} Execute
      </button>
    </div>
    <div class="code-block"><pre><code>${esc(formatCode(q.code, q.kind))}</code></pre></div>
  </div>`;

  // Result area (filled on execute)
  html += `<div class="result-section" id="execSection" style="display:none">
    <div class="result-section-label">Execution Result</div>
    <div id="execResult"></div>
  </div>`;

  // Per-model natural-language responses, right under the result
  html += judgeResponsesHtml(q);

  // Story / topic
  if (q.story) {
    html += `<div class="result-section">
      <div class="result-section-label">Story</div>
      <div class="nl-response md-body">${marked.parse(String(q.story))}</div>
    </div>`;
  }

  // Judge loops
  html += planLoopHtml(q);
  html += codeLoopHtml(q);

  // Tables used
  const tables = q.tables || [];
  if (tables.length > 0) {
    html += `<div class="result-section"><div class="result-section-label">Tables Used</div>
      <div class="table-reasons">`;
    tables.forEach(t => {
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
  document.getElementById('resultModal').scrollTop = 0;
}

function closeModal() {
  document.getElementById('modalBackdrop').style.display = 'none';
  document.getElementById('resultModal').style.display   = 'none';
}

/* ── Execution ──────────────────────────────────────────────────────────── */
async function execQuery(key) {
  const [kind, entryKey, qnum] = key.split('|');
  const btn     = document.getElementById('execBtn');
  const section = document.getElementById('execSection');
  const out     = document.getElementById('execResult');
  btn.disabled  = true;
  btn.innerHTML = `<span class="spinner-sm"></span> Running…`;
  section.style.display = '';
  out.innerHTML = `<div class="modal-loading"><div class="spinner"></div><div>Executing against the real datasets…</div></div>`;
  section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  try {
    const res  = await fetch(
      `/api/execute/${activeSource}/${encodeURIComponent(kind)}/${encodeURIComponent(entryKey)}/${encodeURIComponent(qnum)}`,
      { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
    if (!data.ok) {
      out.innerHTML = `<div class="exec-error"><strong>Execution failed</strong><br>${esc(data.error || 'unknown error')}</div>`;
    } else {
      out.innerHTML = renderResultTable(data.result);
    }
  } catch (err) {
    out.innerHTML = `<div class="exec-error">Request failed: ${esc(err.message)}</div>`;
  } finally {
    btn.disabled  = false;
    btn.innerHTML = `${ICON_PLAY} Execute`;
  }
}

function renderResultTable(result) {
  const cols = result?.columns ?? [];
  const rows = result?.rows ?? [];
  if (!cols.length) return `<div style="font-size:13px;color:var(--text-muted)">The query returned no rows.</div>`;
  const trunc = (result.total_rows ?? 0) > rows.length
    ? `<div class="df-truncated">Showing ${rows.length} of ${result.total_rows} rows</div>` : '';
  return `<div class="df-wrap"><table class="df-table">
    <thead><tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead>
    <tbody>${rows.map(row =>
      `<tr>${row.map(v => `<td${isNumeric(v) ? ' class="num"' : ''}>${esc(v ?? '')}</td>`).join('')}</tr>`
    ).join('')}</tbody></table></div>${trunc}`;
}
