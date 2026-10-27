// Fixframe UI: a thin client of the in-game server's API (/api/...). Holds no logic beyond display.
const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...kids);
  return e;
};

let capture = null;   // the capture shown
let selected = -1;    // draw index
let frameSrc = '';    // the frame image shown (a pick capture keeps the image that was clicked)
let frameInfo = '';
let offline = false;  // opened from a saved capture (fixframe open): nothing that needs the game
let pinned = null;    // the draw others are compared with: {capture, index}

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
  return r.headers.get('content-type')?.startsWith('image/') ? r.blob() : r.json();
}

async function poll() {
  try {
    const s = await api('status');
    offline = !!s.offline;
    $('capture').hidden = offline;
    $('status').textContent = offline ? `Opened from ${s.offline} · ${s.app} · ${s.device} · ${s.backend}`
      : `${s.app} · ${s.device} · ${s.backend} · frame ${s.frame} · ${s.fps} fps · ${s.draws} draws`;
    const sel = $('captures');
    const ids = s.captures.map((c) => String(c.id));
    if ([...sel.options].map((o) => o.value).join() !== ids.slice().reverse().join()) {
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
  $('save').hidden = offline;
  $('save').href = `/api/captures/${id}/file`;
  selected = -1;
  if (capture.image) frameSrc = `/api/captures/${id}/image.png`;
  frameInfo = `Frame ${capture.frame} · ${capture.draws.length} draws and dispatches · ${capture.passes.length} passes`;
  const skipped = capture.draws.filter((d) => d.skipped).length;
  if (skipped) frameInfo += ` · ${skipped} skipped by the renderer`;
  if (capture.pick) {
    const n = capture.draws.filter(touched).length;
    frameInfo += ` · ${n} draw${n === 1 ? '' : 's'} wrote the picked pixel`;
  } else if (!capture.image) {
    frameInfo += ' · the presented image could not be read back';
  }
  $('empty').hidden = true;
  showFrame();
  $('pickonly').hidden = !capture.pick;
  renderList();
  if (draw < 0 && capture.pick) draw = likelyDraw();
  select(draw < capture.draws.length ? draw : -1, true);
}

function showFrame() {
  $('view').hidden = !frameSrc;
  $('image').src = frameSrc;
  $('frameinfo').textContent = frameInfo;
  $('back').hidden = true;
  const cross = $('cross');
  cross.hidden = !capture?.pick;
  if (capture?.pick) {
    cross.style.left = capture.pick[0] * 100 + '%';
    cross.style.top = capture.pick[1] * 100 + '%';
  }
}

// --- picks

// A target whose shape differs from the screen's (a shadow map, an atlas) maps the picked pixel nowhere useful.
function onScreen(d) {
  if (d.pass < 0) return false;
  const p = capture.passes[d.pass], img = $('image');
  if (!img.naturalWidth || !p.height) return true;
  return Math.abs(p.width / p.height - img.naturalWidth / img.naturalHeight) < 0.1;
}
const touched = (d) => d.samples > 0 && !d.approximate && onScreen(d);
const maybe = (d) => d.samples > 0 && d.approximate && onScreen(d);  // static scissor: counted over the whole draw
const fullscreen = (d) => d.count <= 6 && d.instances <= 1;  // a quad or a big triangle: a full-screen pass

// The draw most likely meant: the last geometry draw that wrote the pixel (full-screen passes come after).
function likelyDraw() {
  let last = -1, lastAny = -1;
  capture.draws.forEach((d, i) => {
    if (!touched(d)) return;
    lastAny = i;
    if (!fullscreen(d)) last = i;
  });
  return last >= 0 ? last : lastAny;
}

$('image').addEventListener('click', async (e) => {
  if (!capture || $('back').hidden === false) return;
  if (offline) {
    $('frameinfo').textContent = 'Picking a pixel needs the game running (this capture was opened from a file).';
    return;
  }
  const x = e.offsetX / e.target.clientWidth, y = e.offsetY / e.target.clientHeight;
  $('frameinfo').textContent = 'Finding the draws that wrote this pixel… (the game shows one odd frame)';
  try {
    const { id } = await api(`pick?x=${x.toFixed(5)}&y=${y.toFixed(5)}`, { method: 'POST' });
    await poll();
    await load(id);
  } catch (err) {
    $('frameinfo').textContent = 'Pick failed: ' + err.message;
  }
});

// --- the list

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

// What a pass renders into: "1920×1080 · R8G8B8A8_UNORM + D32_SFLOAT" (names: true adds the textures' names).
// Emulators often begin a pass per draw, so the list groups consecutive passes with the same targets.
function targets(i, names = false) {
  if (i < 0) return 'Outside render passes';
  const p = capture.passes[i];
  const name = (id) => {
    const t = capture.textures[id];
    return !t ? '?' : names && t.name ? `${t.name} (${t.format})` : t.format;
  };
  const ids = p.colors.concat(p.depth !== '0' ? [p.depth] : []);
  return `${p.width}×${p.height}${ids.length ? ' · ' + ids.map(name).join(' + ') : ''}`;
}

// "Colour, 2 targets + depth · 1920×1080 · R8G8B8A8_UNORM + D32_SFLOAT" (RenderDoc's pass names, roughly)
function passTitle(i) {
  if (i < 0) return 'Outside render passes';
  const p = capture.passes[i], n = p.colors.length, depth = p.depth !== '0';
  const kind = n ? `Colour, ${n} target${n > 1 ? 's' : ''}${depth ? ' + depth' : ''}` : depth ? 'Depth only' : 'No targets';
  return `${kind} · ${targets(i)}`;
}

function describePass(i) {
  return i < 0 ? targets(i) : `Pass ${i} · ${targets(i, true)}`;
}

function renderList() {
  const words = $('filter').value.toLowerCase().split(/\s+/).filter(Boolean);
  const onlyPicked = capture.pick && $('pickonly').firstElementChild.checked;
  const rows = [];
  let group = null;
  capture.draws.forEach((d, i) => {
    if (onlyPicked && !touched(d) && !maybe(d)) return;
    const what = `${d.kind} ${d.count}${d.instances > 1 ? ' ×' + d.instances : ''}`;
    const hay = `${d.label} ${shaderNames(d)} ${textureNames(d)} ${what} ${d.skipped ? 'skipped ' + d.skipped : ''}`.toLowerCase();
    if (words.some((w) => !hay.includes(w))) return;
    // group by the targets themselves (two targets of one size and format are still two passes)
    const p = capture.passes[d.pass];
    const key = d.pass < 0 ? 'none' : p.colors.join() + '|' + p.depth;
    if (key !== group) {
      group = key;
      rows.push(el('div', { className: 'pass', textContent: passTitle(d.pass), title: targets(d.pass, true) }));
    }
    const text = d.skipped ? `${what} · skipped: ${d.skipped}` : d.label ? `${what} · ${d.label}` : what;
    const row = el('div', { className: d.skipped ? 'draw skipped' : 'draw', role: 'option' },
      el('span', { className: 'n', textContent: i }), el('span', { className: 'what', textContent: text }));
    if (touched(d)) row.append(el('span', { className: 'hit', textContent: `● ${d.samples}`, title: 'Samples it wrote at the picked pixel' }));
    else if (maybe(d)) row.append(el('span', { className: 'hit', textContent: '?', title: 'Maybe: its scissor is fixed in its pipeline, so it was counted over the whole draw' }));
    if (pinned && pinned.capture.id === capture.id && pinned.index === i) row.append(el('span', { className: 'a', textContent: 'A', title: 'Compared with' }));
    row.dataset.index = i;
    row.setAttribute('aria-selected', i === selected);
    rows.push(row);
  });
  if (!rows.length) rows.push(el('p', { className: 'hint', textContent: 'No draws match.', style: 'padding: 0 10px' }));
  $('draws').replaceChildren(...rows);
}

// --- the inspector

// Effective state: the program's own, with what the draw set dynamically in its place; the rest set per draw.
function effective(c, d) {
  const p = c.programs[d.program];
  const dyn = new Map(d.dynamic);
  let state = (p ? p.state : []).map(([k, v]) => dyn.has(k) ? [k, `${dyn.get(k)} (set per draw)`] : [k, v]);
  const keys = new Set(state.map(([k]) => k));
  // the sub-fields of a feature that is off don't matter (and would show as differences in a comparison)
  const val = (k) => (state.find(([x]) => x === k)?.[1] || '').split(' ')[0];
  const off = (k) => keys.has(k) && val(k) === 'off';
  const constants = state.some(([k, v]) => k.startsWith('blend ') && v.includes('CONSTANT'));
  state = state.filter(([k]) => !(off('depth test') && (k === 'depth write' || k === 'depth compare')) &&
    !(off('stencil test') && k.startsWith('stencil ') && k !== 'stencil test') &&
    !(off('depth bias') && k === 'depth bias values') && !(k === 'blend constants' && !constants));
  return { state, perDraw: d.dynamic.filter(([k]) => !keys.has(k)) };
}

// What looks wrong at a glance (RenderDoc paints these rows red).
function problems(c, d) {
  const out = [];
  if (d.skipped) out.push(`The renderer skipped it: ${d.skipped}.`);
  const size = (k) => (d.dynamic.find(([x]) => x === k)?.[1] || '').match(/(-?[\d.]+) x (-?[\d.]+)/);
  for (const k of ['viewport', 'scissor']) {
    const m = size(k);
    if (m && (+m[1] === 0 || +m[2] === 0)) out.push(`Its ${k} is empty (${m[1]} × ${m[2]}): it draws nothing.`);
  }
  for (const [k, v] of effective(c, d).state)
    if (k.startsWith('write mask') && v === 'none') out.push(`${k}: none, it writes no colour to that target.`);
  if (d.approximate && d.samples > 0) out.push('Its pick count is for the whole draw: its scissor is fixed in its pipeline.');
  return out;
}

// Everything recorded about a draw, as label -> text, for comparing two draws (of any captures).
function facts(c, i) {
  const d = c.draws[i], p = c.programs[d.program], f = new Map();
  const tex = (id) => {
    const t = c.textures[id];
    return t ? `${t.format} ${t.width}×${t.height}${t.name ? ' · ' + t.name : ''}` : '?';
  };
  f.set('Kind', d.kind).set('Count', String(d.count)).set('Instances', String(d.instances)).set('Label', d.label || '—');
  f.set('Drawn', d.skipped ? `no: ${d.skipped}` : 'yes');
  if (d.pass >= 0) {
    const ps = c.passes[d.pass];
    f.set('Target size', `${ps.width}×${ps.height}`);
    ps.colors.forEach((t, k) => f.set(`Colour target ${k}`, tex(t)));
    if (ps.depth !== '0') f.set('Depth target', tex(ps.depth));
  }
  if (p) {
    f.set('Pipeline', p.name || '(unnamed)');
    p.shaders.forEach((s) => f.set(`Shader, ${s.stage}`, `${s.name || '(unnamed)'} ${s.hash}`));
  }
  const { state, perDraw } = effective(c, d);
  for (const [k, v] of state.concat(perDraw)) f.set(k, v);
  for (const [slot, t] of d.textures) f.set(`Texture, ${slot}`, tex(t));
  for (const [slot, s] of d.samplers || []) f.set(`Sampler, ${slot}`, s);
  return f;
}

function comparison(i) {
  const a = facts(pinned.capture, pinned.index), b = facts(capture, i);
  const rows = [...new Set([...a.keys(), ...b.keys()])].filter((k) => a.get(k) !== b.get(k));
  const where = pinned.capture.id === capture.id ? `draw ${pinned.index}` : `draw ${pinned.index} of capture ${pinned.capture.id}`;
  const parts = [el('h2', { textContent: `Different from A (${where})` })];
  if (!rows.length) {
    parts.push(el('p', { className: 'hint', textContent: `Nothing: the ${a.size} things Fixframe records are the same.` }));
  } else {
    parts.push(el('table', { className: 'diff' },
      el('tr', {}, el('th', { textContent: '' }), el('th', { textContent: 'A' }), el('th', { textContent: 'This draw' })),
      ...rows.map((k) => el('tr', {}, el('td', { textContent: k }),
        el('td', { className: 'mono', textContent: a.get(k) ?? '—' }), el('td', { className: 'mono', textContent: b.get(k) ?? '—' })))),
      el('p', { className: 'hint', textContent: `${rows.length} different, ${new Set([...a.keys(), ...b.keys()]).size - rows.length} the same.` }));
  }
  return parts;
}

function table(pairs, mono = false) {
  return el('table', {}, ...pairs.map(([k, v]) =>
    el('tr', {}, el('td', { textContent: k }), el('td', { className: mono ? 'mono' : '', textContent: v }))));
}

function select(i, scroll = false) {
  selected = i;
  for (const r of $('draws').querySelectorAll('.draw')) {
    r.setAttribute('aria-selected', +r.dataset.index === i);
    if (scroll && +r.dataset.index === i) r.scrollIntoView({ block: 'center' });
  }
  remember();
  const box = $('inspector');
  if (i < 0) {
    box.replaceChildren(el('p', { className: 'hint', textContent:
      capture ? 'Select a draw, or click the frame to find the draws that wrote a pixel.' : 'Select a draw to see what it draws with.' }));
    return;
  }
  const d = capture.draws[i];
  const p = capture.programs[d.program];
  const isA = pinned && pinned.capture.id === capture.id && pinned.index === i;
  const parts = [];
  if (pinned) {
    const clear = el('button', { type: 'button', className: 'quiet', textContent: 'Stop comparing' });
    clear.addEventListener('click', () => { pinned = null; renderList(); select(selected); });
    parts.push(el('div', { className: 'compare' },
      el('span', { textContent: isA ? 'This is A: select another draw (any capture) to compare.' : 'Comparing with A' }), clear));
    if (!isA) parts.push(...comparison(i));
  }
  parts.push(
    el('h2', { textContent: `Draw ${i}` }),
    table([['Kind', d.kind]].concat(d.skipped ? [['Skipped by the renderer', d.skipped]] : [],
          [['Count', d.count], ['Instances', d.instances], ['Label', d.label || '—'],
           ['Pass', describePass(d.pass)]], d.samples >= 0 ? [['At the picked pixel', `${d.samples} samples`]] : [])));
  const issues = problems(capture, d);
  if (issues.length) parts.push(el('h2', { textContent: 'Problems' }), el('ul', { className: 'problems' },
    ...issues.map((t) => el('li', { textContent: t }))));
  const actions = el('div', { className: 'actions' });
  if (d.pass >= 0 && !offline) {
    const b = el('button', { type: 'button', className: 'quiet', textContent: 'Show its target right after this draw' });
    b.addEventListener('click', () => scrub(i, b));
    actions.append(b);
  }
  if (!isA) {
    const b = el('button', { type: 'button', className: 'quiet', textContent: 'Compare other draws with this one' });
    b.addEventListener('click', () => { pinned = { capture, index: i }; renderList(); select(i); });
    actions.append(b);
  }
  parts.push(actions);
  const { state, perDraw } = effective(capture, d);
  if (p) {
    if (p.name) parts.push(el('p', { className: 'mono', textContent: p.name }));
    parts.push(el('h2', { textContent: 'Shaders' }), el('table', {}, ...p.shaders.map((s) =>
      el('tr', {}, el('td', { textContent: s.stage }),
        el('td', {}, el('div', { className: 'mono', textContent: s.name || '(unnamed)' }),
          el('div', { className: 'hint mono', textContent: `${s.hash} · ${s.size} B` }))))));
    if (state.length) parts.push(el('h2', { textContent: 'State' }), table(state));
  }
  if (perDraw.length) parts.push(el('h2', { textContent: 'Set for this draw' }), table(perDraw, true));
  const samplers = new Map(d.samplers || []);
  parts.push(el('h2', { textContent: 'Textures' }), d.textures.length
    ? table(d.textures.map(([slot, t]) => {
        const x = capture.textures[t] || {};
        return [slot, `${x.name || '(unnamed)'} · ${x.format || '?'} · ${x.width}×${x.height}` +
                      (x.mips > 1 ? ` · ${x.mips} mips` : '') + (samplers.has(slot) ? ` — ${samplers.get(slot)}` : '')];
      }))
    : el('p', { className: 'hint', textContent: 'None bound.' }));
  box.replaceChildren(...parts);
}

// The draw's target as it is right after the draw, from a fresh frame of the game.
async function scrub(i, button) {
  button.disabled = true;
  button.textContent = 'Reading its target…';
  try {
    const blob = await api(`captures/${capture.id}/draws/${i}/target`, { method: 'POST' });
    $('image').src = URL.createObjectURL(blob);
    $('view').hidden = false;
    $('cross').hidden = true;
    $('frameinfo').textContent = `${describePass(capture.draws[i].pass)}, right after draw ${i} (from a new frame)`;
    $('back').hidden = false;
  } catch (err) {
    $('frameinfo').textContent = `Draw ${i}: ${err.message}`;
  } finally {
    button.disabled = false;
    button.textContent = 'Show its target right after this draw';
  }
}

$('back').addEventListener('click', showFrame);
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
$('pickonly').addEventListener('change', () => capture && renderList());
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
// a capture opened from a file has nothing to capture: show it straight away
const first = q.get('capture') ?? (offline ? $('captures').value : '');
if (first) load(first, +(q.get('draw') ?? -1)).catch(() => {});
