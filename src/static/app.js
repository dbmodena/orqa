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
let activePage   = 'browse';   // 'browse' | 'stats'
let currentPage  = 1;
const PAGE_SIZE  = 30;
let statsCache   = {};          // source id -> {stats, walks, clusters} once fetched

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
function isNumeric(v) { return typeof v !== 'boolean' && v !== null && v !== '' && !isNaN(Number(v)); }
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
/* Reverse-index / search glyph for the Retrieval Gate badge. */
const ICON_INDEX = `<svg viewBox="0 0 14 14" fill="none" width="12" height="12">
  <circle cx="6" cy="6" r="4" stroke="currentColor" stroke-width="1.8"/>
  <path d="M9.3 9.3l3 3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`;

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
  currentPage  = 1;
  renderSourceGrid();
  updateTopbarMeta();
  document.getElementById('workArea').style.display = '';
  document.getElementById('topNav').style.display   = '';
  showPage('browse');
  await loadQueries();
}

function updateTopbarMeta() {
  const el = document.getElementById('topbarMeta');
  if (!activeSource) { el.textContent = '—'; return; }
  const src = allSources.find(s => s.id === activeSource);
  el.textContent = src ? `${src.label} · ${src.type}` : '—';
}

/* ── Page nav ───────────────────────────────────────────────────────────── */
function showPage(page) {
  activePage = page;
  document.getElementById('pageBrowse').style.display = page === 'browse' ? '' : 'none';
  document.getElementById('pageStats').style.display  = page === 'stats'  ? '' : 'none';
  document.getElementById('navBrowse').classList.toggle('active', page === 'browse');
  document.getElementById('navStats').classList.toggle('active', page === 'stats');
  if (page === 'stats') loadStatsPage();
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

/* Filter changes always jump back to page 1 — otherwise a narrower filter
   can silently strand the user on a now-empty page. */
function onFilterChange() {
  currentPage = 1;
  renderQueries();
}

function filteredQueries() {
  const search = document.getElementById('searchBox').value.toLowerCase();
  const kind   = document.getElementById('kindFilter').value;
  const skill  = document.getElementById('skillFilter').value;
  const diff   = document.getElementById('diffFilter').value.toLowerCase();

  return allQueries.filter(q => {
    if (search && !q.question.toLowerCase().includes(search)) return false;
    if (kind && q.kind !== kind) return false;
    if (skill === '__plain__' && (q.task_types || []).length > 0) return false;
    if (skill && skill !== '__plain__' && !(q.task_types || []).includes(skill)) return false;
    if (diff && (q.difficulty || '').toLowerCase() !== diff) return false;
    return true;
  });
}

function renderQueries() {
  const filtered  = filteredQueries();
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  currentPage     = Math.min(Math.max(1, currentPage), pageCount);
  const start     = (currentPage - 1) * PAGE_SIZE;
  const pageItems = filtered.slice(start, start + PAGE_SIZE);

  document.getElementById('qCount').textContent = filtered.length
    ? `${filtered.length} of ${allQueries.length} · page ${currentPage}/${pageCount}`
    : `0 of ${allQueries.length}`;
  const list = document.getElementById('queriesList');
  list.innerHTML = '';

  if (filtered.length === 0) {
    list.innerHTML = '<div class="banner">No queries match your filters.</div>';
    renderPagination(0, 1);
    return;
  }

  pageItems.forEach(q => {
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

  renderPagination(filtered.length, pageCount);
}

function goToPage(p) {
  currentPage = p;
  renderQueries();
  document.getElementById('queriesSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* Compact page-number list: first, last, current ±1, with ellipses for gaps
   — plain Prev/Next + 1..N would be unusable once a filter still leaves
   dozens of pages. */
function renderPagination(total, pageCount) {
  const el = document.getElementById('pagination');
  if (!el) return;
  if (total === 0 || pageCount <= 1) { el.innerHTML = ''; return; }

  const pages = new Set([1, pageCount, currentPage, currentPage - 1, currentPage + 1]);
  const sorted = [...pages].filter(p => p >= 1 && p <= pageCount).sort((a, b) => a - b);

  let html = `<button class="pg-btn" ${currentPage === 1 ? 'disabled' : ''} onclick="goToPage(${currentPage - 1})">‹ Prev</button>`;
  let prev = 0;
  sorted.forEach(p => {
    if (p - prev > 1) html += `<span class="pg-ellipsis">…</span>`;
    html += `<button class="pg-btn ${p === currentPage ? 'pg-active' : ''}" onclick="goToPage(${p})">${p}</button>`;
    prev = p;
  });
  html += `<button class="pg-btn" ${currentPage === pageCount ? 'disabled' : ''} onclick="goToPage(${currentPage + 1})">Next ›</button>`;
  el.innerHTML = html;
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
  ['expected_result_approval', 'result-type'],
  ['difficulty_approval',      'difficulty'],
  ['convergence_approval',     'convergence'],
  ['metric_combination_approval', 'combination'],
  ['topic_linkage_approval',   'topic-linkage'],
];

const CODE_LAYERS = [
  ['plan_compliance_approval', 'compliance'],
  ['present_result_approval',  'result'],
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

/* Retrieval Gate card: same shape as robotCard (avatar/name/verdict/
   feedback) so it sits INSIDE the judge panel grid as one more member of
   the panel, but for the deterministic keyword-searchability check (see
   orqa.agent.utility.keyword_searchability) rather than an LLM vote — an
   index/magnifying-glass icon stands in for the robot avatar, and the
   feedback shown is the EXACT constructed text this gate sent (or would
   send) to the query planner (agent.py's kw_planner_feedback), not a
   UI-only paraphrase. Absent entirely on older runs that predate this
   check (the field simply isn't on the attempt). */
function retrievalGateCard(att) {
  if (att.keyword_searchability_approval === null || att.keyword_searchability_approval === undefined) return '';
  const ok = !!att.keyword_searchability_approval;
  const fb = ok
    ? 'Every table this plan uses was retrieved by these keywords.'
    : (att.keyword_searchability_feedback
        || `Not retrieved: ${(att.keyword_searchability_missing_tables || []).join(', ')}`);
  return `
    <div class="robot-card gate-card ${ok ? 'robot-ok' : 'robot-bad'}">
      <div class="robot-head">
        <span class="robot-avatar gate-avatar">${ICON_INDEX}</span>
        <div class="robot-id">
          <span class="robot-name">Retrieval Gate</span>
          <span class="robot-verdict ${ok ? 'v-ok' : 'v-bad'}">${ok ? ICON_CHECK : ICON_CROSS} ${ok ? 'approved' : 'rejected'}</span>
        </div>
      </div>
      ${fb ? `<div class="robot-feedback">${esc(fb)}</div>` : ''}
    </div>`;
}

/* `gate`: the plan attempt dict (carries keyword_searchability_* fields) to
   fold a Retrieval Gate card into this panel's grid, alongside the LLM
   judges — omitted (or lacking the field) for panels with no such gate,
   e.g. the code judge loop's panelBlock(att.panel, []) call. */
function panelBlock(panel, layers, gate) {
  if (!panel || !(panel.votes || []).length) return '';
  const tally = `
    <span class="panel-tally">
      <span class="t-ok">${panel.approve_votes ?? 0} ${ICON_CHECK}</span>
      <span class="t-bad">${panel.reject_votes ?? 0} ${ICON_CROSS}</span>
      ${panel.failed_votes ? `<span class="t-fail">${panel.failed_votes} ⚡</span>` : ''}
    </span>`;
  const robots = panel.votes.map(v => robotCard(v, layers)).join('') + (gate ? retrievalGateCard(gate) : '');
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
          ${panelBlock(att.panel, PLAN_LAYERS, att)}
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
          ${panelBlock(att.panel, CODE_LAYERS)}
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

  // Keyword retrieval — re-run the plan judge's keyword-searchability
  // check by hand: search the portal's real reverse index with this
  // query's own question_keywords and see which tables actually surface.
  const qKeywords = q.question_keywords || [];
  if (qKeywords.length > 0) {
    html += `<div class="result-section">
      <div class="result-section-label">
        Keyword Retrieval
        <button class="btn btn-run btn-exec" id="kwSearchBtn" onclick='runKeywordSearch(${JSON.stringify(queryKey(q))})'>
          ${ICON_PLAY} Search Index
        </button>
      </div>
      <div class="kw-tags">${qKeywords.map(k => `<span class="tag tag-plain">${esc(k)}</span>`).join('')}</div>
      <div id="kwSearchResult"></div>
    </div>`;
  }

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

/* ── Keyword-search retrieval check ────────────────────────────────────── */
async function runKeywordSearch(key) {
  const q = allQueries.find(x => queryKey(x) === key);
  const btn = document.getElementById('kwSearchBtn');
  const out = document.getElementById('kwSearchResult');
  if (!q || !btn || !out) return;

  btn.disabled  = true;
  btn.innerHTML = `<span class="spinner-sm"></span> Searching…`;
  out.innerHTML = `<div class="modal-loading"><div class="spinner"></div><div>Searching the reverse index…</div></div>`;

  try {
    const params = new URLSearchParams();
    (q.question_keywords || []).forEach(k => params.append('keywords', k));
    // Same adaptive top-K the plan judge's gate used for this plan:
    // round(num_tables * keyword_search_top_k_coefficient), computed
    // server-side from the source's own workflow config — see
    // web_app.py's keyword_search().
    params.append('num_tables', (q.tables || []).length || 1);
    const res  = await fetch(`/api/keyword-search/${activeSource}?${params.toString()}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
    out.innerHTML = renderKeywordSearchResults(data, q);
  } catch (err) {
    out.innerHTML = `<div class="exec-error">Request failed: ${esc(err.message)}</div>`;
  } finally {
    btn.disabled  = false;
    btn.innerHTML = `${ICON_PLAY} Search Index`;
  }
}

function renderKeywordSearchResults(data, q) {
  if (!data.available) {
    return `<div class="banner">No reverse index available for this source.</div>`;
  }
  if (!data.results.length) {
    return `<div class="banner">No matches for these keywords.</div>`;
  }
  // This query's own tables, by the resource_id the index actually keys
  // on (q.tables carries the ALIAS in .name — q.tables_map resolves it).
  const ownIds = new Set(
    (q.tables || []).map(t => (q.tables_map || {})[t.name] || t.name)
  );
  const retrievedIds = new Set(data.results.map(r => r.resource_id));
  const missing = [...ownIds].filter(id => !retrievedIds.has(id));

  // The actual K requested (adaptive, from the server — see keyword_search()
  // in web_app.py), not data.results.length: the index can legitimately
  // return fewer hits than K, and reporting that smaller count as "the top
  // N" would understate how wide a net was actually cast.
  const topK = data.top_k || data.results.length;
  const summary = missing.length
    ? `<div class="exec-error">${missing.length} of this query's table(s) did NOT surface in the top ${topK}: ${missing.map(esc).join(', ')}</div>`
    : `<div class="banner banner-ok">${ICON_CHECK} Every table this query uses was retrieved.</div>`;

  const rows = data.results.map(r => {
    const hit = ownIds.has(r.resource_id);
    return `<div class="table-reason-row ${hit ? 'kw-hit' : ''}">
      <div class="table-reason-name">${hit ? ICON_CHECK + ' ' : ''}${esc(r.title || r.resource_id)}</div>
      <div class="table-reason-body">
        <div class="table-reason-text">score ${r.score.toFixed(2)}${(r.matched_terms || []).length ? ' · matched: ' + r.matched_terms.map(esc).join(', ') : ''}</div>
      </div>
    </div>`;
  }).join('');

  return `${summary}<div class="table-reasons" style="margin-top:8px">${rows}</div>`;
}

/* One result cell -> its <td>. Missing values (JSON null — the backend
   collapses both None and NaN/NaT/inf into it, see web_app.py's
   _json_safe) get an explicit, visibly-marked placeholder instead of
   silently rendering as an empty cell indistinguishable from a real empty
   string. Array/object cells (e.g. a `list`-typed plan's per-group
   aggregation, which puts an actual list in a DataFrame cell) are shown as
   compact JSON rather than JS's default comma-joined Array.toString(). */
function renderCell(v) {
  if (v === null || v === undefined) return `<td class="cell-na">NaN</td>`;
  if (typeof v === 'boolean') return `<td>${v}</td>`;
  if (Array.isArray(v) || (typeof v === 'object')) {
    return `<td>${esc(JSON.stringify(v))}</td>`;
  }
  return `<td${isNumeric(v) ? ' class="num"' : ''}>${esc(v)}</td>`;
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
      `<tr>${row.map(renderCell).join('')}</tr>`
    ).join('')}</tbody></table></div>${trunc}`;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Stats page
   ═══════════════════════════════════════════════════════════════════════════ */

function clusterColor(clusterId) {
  const hue = (Number(clusterId) * 47) % 360;   // hash-spread, distinct-enough without a hardcoded palette
  return `hsl(${hue}, 62%, 52%)`;
}

async function loadStatsPage() {
  if (!activeSource) return;
  if (!statsCache[activeSource]) {
    document.getElementById('statOverview').innerHTML       = `<div class="modal-loading"><div class="spinner"></div><div>Loading stats…</div></div>`;
    document.getElementById('failureRateGrid').innerHTML    = '';
    document.getElementById('planRejectionGrid').innerHTML  = '';
    document.getElementById('codeRejectionGrid').innerHTML  = '';
    document.getElementById('clusterSection').innerHTML     = '';
    document.getElementById('randomWalksSection').innerHTML = '';

    const [stats, walks, clusters] = await Promise.all([
      fetch(`/api/stats/${activeSource}`).then(r => r.json()).catch(() => null),
      fetch(`/api/random-walks/${activeSource}`).then(r => r.json()).catch(() => null),
      fetch(`/api/clusters/${activeSource}`).then(r => r.json()).catch(() => null),
    ]);
    statsCache[activeSource] = { stats, walks, clusters };
  }
  const { stats, walks, clusters } = statsCache[activeSource];
  renderOverview(stats);
  renderRateGrid('failureRateGrid', stats?.failure_rate, ['by_skill', 'by_difficulty', 'by_kind'],
    ['Skill', 'Difficulty', 'Language'], 'rejected', 'not approved');
  renderRateGrid('planRejectionGrid', stats?.plan_rejection, ['by_model', 'by_difficulty', 'by_skill', 'by_kind'],
    ['Judge Model', 'Difficulty', 'Skill', 'Language'], 'rejected', 'rejected');
  renderRateGrid('codeRejectionGrid', stats?.code_rejection, ['by_model', 'by_difficulty', 'by_skill', 'by_kind'],
    ['Judge Model', 'Difficulty', 'Skill', 'Language'], 'rejected', 'rejected');
  renderClusterMap(clusters);
  renderRandomWalks(walks);
}

function renderOverview(stats) {
  const el = document.getElementById('statOverview');
  if (!stats) { el.innerHTML = `<div class="banner">Stats unavailable.</div>`; return; }
  const total = stats.totals?.queries ?? 0;
  const approved = Object.values(stats.failure_rate?.by_kind || {}).reduce((s, v) => s + (v.total - v.rejected), 0);
  const rate = total ? Math.round((approved / total) * 100) : 0;
  el.innerHTML = `
    <div class="stat-card">
      <div class="stat-num">${total}</div>
      <div class="stat-label">Total queries</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" style="color:var(--ok)">${rate}%</div>
      <div class="stat-label">Final approval rate</div>
    </div>`;
}

/* One breakdown group -> a set of small rate tables, one per axis. */
function renderRateGrid(containerId, group, axes, axisLabels, _rejField, rejectedWord) {
  const el = document.getElementById(containerId);
  if (!group) { el.innerHTML = `<div class="banner">Not available.</div>`; return; }
  el.innerHTML = axes.map((axisKey, i) => rateTable(axisLabels[i], group[axisKey] || {}, rejectedWord)).join('');
}

function rateTable(label, data, rejectedWord) {
  const rows = Object.entries(data).sort((a, b) => b[1].rate - a[1].rate);
  if (!rows.length) return `<div class="rate-card"><div class="rate-card-hdr">${esc(label)}</div><div class="rate-empty">No data.</div></div>`;
  return `
    <div class="rate-card">
      <div class="rate-card-hdr">${esc(label)}</div>
      <table class="rate-table">
        <thead><tr><th>${esc(label)}</th><th>${esc(rejectedWord)}</th><th>total</th><th>rate</th></tr></thead>
        <tbody>
          ${rows.map(([key, v]) => `
            <tr>
              <td class="rate-key">${esc(key)}</td>
              <td class="num">${v.rejected}</td>
              <td class="num">${v.total}</td>
              <td class="rate-cell">
                <span class="rate-bar-wrap"><span class="rate-bar" style="width:${Math.round(v.rate * 100)}%"></span></span>
                <span class="rate-pct">${(v.rate * 100).toFixed(1)}%</span>
              </td>
            </tr>`).join('')}
        </tbody>
      </table>
    </div>`;
}

/* ── Random walks ───────────────────────────────────────────────────────── */
function renderRandomWalks(data) {
  const el = document.getElementById('randomWalksSection');
  if (!data || !data.available) {
    el.innerHTML = `<div class="banner">No random-walk file persisted for this source
      ${data?.path ? `<br><span class="path-hint">(expected at <code>${esc(data.path)}</code>)</span>` : ''}.
      Re-run the <code>generate-query-candidates</code> step to produce one.</div>`;
    return;
  }
  const shown = data.walks || [];
  el.innerHTML = `
    <div class="walks-meta">${data.total} walk${data.total === 1 ? '' : 's'} fetched
      ${data.total > shown.length ? `— showing first ${shown.length}` : ''}</div>
    <div class="walks-list">
      ${shown.map(w => `
        <div class="walk-row">
          <span class="walk-seed">${esc(w.seed || '—')}</span>
          <span class="walk-ops">${(w.operation_type || []).map(o => `<span class="tag tag-plain">${esc(o)}</span>`).join('')}</span>
          <span class="walk-datasets">${(w.datasets || []).map(d => `<span class="tag tag-lang">${esc(d)}</span>`).join(' → ')}</span>
        </div>`).join('')}
    </div>`;
}

/* ── Cluster map (2D canvas scatter) ──────────────────────────────────────── */
let clusterHoverData = null;

function renderClusterMap(data) {
  const el = document.getElementById('clusterSection');
  if (!data || !data.available) {
    el.innerHTML = `<div class="banner">No embeddings/cluster data available for this source.</div>`;
    return;
  }
  const points = data.points || [];
  el.innerHTML = `
    <div class="cluster-meta">${data.n_clusters} clusters over ${points.length} tables — hover a point for details.</div>
    <div class="cluster-canvas-wrap">
      <canvas id="clusterCanvas" width="820" height="480"></canvas>
      <div class="cluster-tooltip" id="clusterTooltip" style="display:none"></div>
    </div>
    <details class="collapse" style="margin-top:14px">
      <summary><span class="collapse-arrow">▸</span>Clusters &amp; member tables (${Object.keys(data.clusters || {}).length})</summary>
      <div class="collapse-body">
        <div class="cluster-list">
          ${Object.entries(data.clusters || {}).sort((a, b) => Number(a[0]) - Number(b[0])).map(([cid, members]) => `
            <details class="collapse cluster-item">
              <summary>
                <span class="collapse-arrow">▸</span>
                <span class="cluster-swatch" style="background:${clusterColor(cid)}"></span>
                <span class="attempt-title">Cluster ${esc(cid)}</span>
                <span class="loop-count">${members.length} table${members.length === 1 ? '' : 's'}</span>
              </summary>
              <div class="collapse-body">
                <div class="cluster-members">${members.map(m => `<span class="tag tag-plain">${esc(m)}</span>`).join('')}</div>
              </div>
            </details>`).join('')}
        </div>
      </div>
    </details>`;

  drawClusterCanvas(points);
}

function drawClusterCanvas(points) {
  const canvas = document.getElementById('clusterCanvas');
  if (!canvas || !points.length) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, PAD = 24;

  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1;

  const toPx = (p) => ({
    px: PAD + ((p.x - minX) / spanX) * (W - 2 * PAD),
    py: H - PAD - ((p.y - minY) / spanY) * (H - 2 * PAD),
  });

  ctx.clearRect(0, 0, W, H);
  const plotted = points.map(p => ({ ...p, ...toPx(p) }));
  plotted.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.px, p.py, 3.4, 0, Math.PI * 2);
    ctx.fillStyle = clusterColor(p.cluster);
    ctx.globalAlpha = 0.75;
    ctx.fill();
  });
  ctx.globalAlpha = 1;

  const tooltip = document.getElementById('clusterTooltip');
  canvas.onmousemove = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (ev.clientX - rect.left) * (canvas.width / rect.width);
    const my = (ev.clientY - rect.top) * (canvas.height / rect.height);
    let nearest = null, bestDist = 64; // px^2 radius
    for (const p of plotted) {
      const d = (p.px - mx) ** 2 + (p.py - my) ** 2;
      if (d < bestDist) { bestDist = d; nearest = p; }
    }
    if (nearest) {
      tooltip.style.display = '';
      tooltip.style.left = `${nearest.px + 12}px`;
      tooltip.style.top  = `${nearest.py + 8}px`;
      tooltip.innerHTML = `<span class="cluster-swatch" style="background:${clusterColor(nearest.cluster)}"></span>
        <strong>${esc(nearest.name)}</strong><br><span class="tt-cluster">cluster ${esc(nearest.cluster)}</span>`;
    } else {
      tooltip.style.display = 'none';
    }
  };
  canvas.onmouseleave = () => { tooltip.style.display = 'none'; };
}
