// Fixframe UI: a thin client of the in-game server's API (/api/...). Holds no logic beyond display.
const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...kids);
  return e;
};

let capture = null;   // the capture shown
let selected = -1;    // draw index

// The view lives in the query string (never a #fragment): ?capture=3&draw=120
function remember() {
  const q = new URLSearchParams();
  if (capture) q.set('capture', capture.id);
  if (selected >= 0) q.set('draw', selected);
  history.replaceState(null, '', q.toString() ? '?' + q : location.pathname);
}

async function api(path, options) {
  const r = await fetch('/api/' + path, options);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

async function poll() {
  try {
    const s = await api('status');
    $('status').textContent = `${s.app} · ${s.device} · ${s.backend} · frame ${s.frame} · ${s.fps} fps · ${s.draws} draws`;
    const sel = $('captures');
    const ids = s.captures.map((c) => String(c.id));
    if ([...sel.options].map((o) => o.value).join() !== ids.join()) {
      sel.replaceChildren(...s.captures.slice().reverse().map((c) =>
        el('option', { value: c.id, textContent: `Capture ${c.id} · frame ${c.frame} · ${c.draws} draws` })));
      if (capture) sel.value = capture.id;
    }
  } catch {
    $('status').textContent = 'The game is not running (or the layer is not loaded).';
  }
}

async function load(id, draw = -1) {
  capture = await api('captures/' + id);
  $('captures').value = id;
  selected = -1;
  const img = $('image');
  img.hidden = !capture.image;
  if (capture.image) img.src = `/api/captures/${id}/image.png`;
  $('empty').hidden = true;
  $('frameinfo').textContent =
    `Frame ${capture.frame} · ${capture.draws.length} draws and dispatches · ${capture.passes.length} passes` +
    (capture.image ? '' : ' · the presented image could not be read back');
  renderList();
  select(draw < capture.draws.length ? draw : -1);
}

function shaderNames(d) {
  const p = capture.programs[d.program];
  return p ? p.shaders.map((s) => s.name || s.hash).join(' ') : '';
}

function textureNames(d) {
  return d.textures.map(([, t]) => {
    const x = capture.textures[t];
    return x ? `${x.name} ${x.format} ${x.width}x${x.height}` : '';
  }).join(' ');
}

// What a pass renders into: "1920×1080 · R8G8B8A8_UNORM + D32_SFLOAT". Emulators often begin a pass per draw,
// so the list groups consecutive passes with the same targets under one header.
function targets(i) {
  if (i < 0) return 'Outside render passes';
  const p = capture.passes[i];
  const name = (id) => {
    const t = capture.textures[id];
    return t ? (t.name ? `${t.name} (${t.format})` : t.format) : '?';
  };
  const list = p.colors.map(name).concat(p.depth !== '0' ? [name(p.depth)] : []).join(' + ');
  return `${p.width}×${p.height}${list ? ' · ' + list : ''}`;
}

function describePass(i) {
  return i < 0 ? targets(i) : `Pass ${i} · ${targets(i)}`;
}

function renderList() {
  const words = $('filter').value.toLowerCase().split(/\s+/).filter(Boolean);
  const rows = [];
  let group = null;
  capture.draws.forEach((d, i) => {
    const what = `${d.kind} ${d.count}${d.instances > 1 ? ' ×' + d.instances : ''}`;
    const hay = `${d.label} ${shaderNames(d)} ${textureNames(d)} ${what}`.toLowerCase();
    if (words.some((w) => !hay.includes(w))) return;
    const key = targets(d.pass);
    if (key !== group) {
      group = key;
      rows.push(el('div', { className: 'pass', textContent: key }));
    }
    const row = el('div', { className: 'draw', role: 'option' },
      el('span', { className: 'n', textContent: i }),
      el('span', { className: 'what', textContent: d.label ? `${what} · ${d.label}` : what }));
    row.dataset.index = i;
    row.setAttribute('aria-selected', i === selected);
    rows.push(row);
  });
  $('draws').replaceChildren(...rows);
}

function table(pairs, mono = false) {
  return el('table', {}, ...pairs.map(([k, v]) =>
    el('tr', {}, el('td', { textContent: k }), el('td', { className: mono ? 'mono' : '', textContent: v }))));
}

function select(i) {
  selected = i;
  for (const r of $('draws').querySelectorAll('.draw')) r.setAttribute('aria-selected', +r.dataset.index === i);
  remember();
  const box = $('inspector');
  if (i < 0) {
    box.replaceChildren(el('p', { className: 'hint', textContent: 'Select a draw to see what it draws with.' }));
    return;
  }
  const d = capture.draws[i];
  const p = capture.programs[d.program];
  const parts = [
    el('h2', { textContent: `Draw ${i}` }),
    table([['Kind', d.kind], ['Count', d.count], ['Instances', d.instances], ['Label', d.label || '—'],
           ['Pass', describePass(d.pass)]]),
  ];
  // Effective state: the program's own, with what this draw set dynamically in its place.
  const dyn = new Map(d.dynamic);
  const state = (p ? p.state : []).map(([k, v]) => dyn.has(k) ? [k, `${dyn.get(k)} (set per draw)`] : [k, v]);
  const keys = new Set(state.map(([k]) => k));
  const perDraw = d.dynamic.filter(([k]) => !keys.has(k));
  if (p) {
    if (p.name) parts.push(el('p', { className: 'mono', textContent: p.name }));
    parts.push(el('h2', { textContent: 'Shaders' }), el('table', {}, ...p.shaders.map((s) =>
      el('tr', {}, el('td', { textContent: s.stage }),
        el('td', {}, el('div', { className: 'mono', textContent: s.name || '(unnamed)' }),
          el('div', { className: 'hint mono', textContent: `${s.hash} · ${s.size} B` }))))));
    if (state.length) parts.push(el('h2', { textContent: 'State' }), table(state));
  }
  if (perDraw.length) parts.push(el('h2', { textContent: 'Set for this draw' }), table(perDraw, true));
  parts.push(el('h2', { textContent: 'Textures' }), d.textures.length
    ? table(d.textures.map(([slot, t]) => {
        const x = capture.textures[t] || {};
        return [slot, `${x.name || '(unnamed)'} · ${x.format || '?'} · ${x.width}×${x.height}` +
                      (x.mips > 1 ? ` · ${x.mips} mips` : '')];
      }))
    : el('p', { className: 'hint', textContent: 'None bound.' }));
  box.replaceChildren(...parts);
}

$('draws').addEventListener('click', (e) => {
  const row = e.target.closest('.draw');
  if (row) select(+row.dataset.index);
});
$('draws').addEventListener('keydown', (e) => {
  if (!capture || !['ArrowDown', 'ArrowUp'].includes(e.key)) return;
  e.preventDefault();
  const rows = [...$('draws').querySelectorAll('.draw')];
  const at = rows.findIndex((r) => +r.dataset.index === selected);
  const next = rows[Math.max(0, Math.min(rows.length - 1, at + (e.key === 'ArrowDown' ? 1 : -1)))];
  if (next) { select(+next.dataset.index); next.scrollIntoView({ block: 'nearest' }); }
});
$('draws').tabIndex = 0;
$('filter').addEventListener('input', () => capture && renderList());
$('captures').addEventListener('change', (e) => load(e.target.value));
$('capture').addEventListener('click', async () => {
  const b = $('capture');
  b.disabled = true;
  b.textContent = 'Capturing…';
  try {
    const { id } = await api('capture', { method: 'POST' });
    await poll();
    await load(id);
  } catch (err) {
    $('frameinfo').textContent = 'Capture failed: ' + err.message;
  } finally {
    b.disabled = false;
    b.textContent = 'Capture frame';
  }
});

const q = new URLSearchParams(location.search);  // read before select() rewrites it
select(-1);
await poll();
setInterval(poll, 1000);
if (q.get('capture')) load(q.get('capture'), +(q.get('draw') ?? -1)).catch(() => {});
