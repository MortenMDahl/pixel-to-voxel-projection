// Dashboard client: consumes /events (SSE) + /stream/{port}.mjpg and renders
// the target table + selected detail, charts, event log, sim controls, and
// the three.js scene (voxels, per-target trails and velocity arrows).

import * as THREE from 'three';
import { OrbitControls } from './OrbitControls.js';

const MAX_VOXELS = 2000;
const TRAIL_MAX = 900;
const CHART_MAX = 900;
const CARDINALS = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                   'S','SSW','SW','WSW','W','WNW','NW','NNW'];
// Per-target colors, keyed by id % length — consistent across table and 3D
const PALETTE = ['#ffd747', '#4ea1ff', '#ff4bd8', '#35c98e', '#ff8c42',
                 '#b487ff', '#4fd8eb', '#f2637f', '#a8d94a', '#e0e0e0'];

const $ = id => document.getElementById(id);
const targetColor = id => PALETTE[id % PALETTE.length];

// --------------------------------------------------------------------------
// three.js scene (Z-up world, matching the pipeline convention)
// --------------------------------------------------------------------------

const container = $('scene-container');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1017);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 5000);
camera.up.set(0, 0, 1);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.position.set(60, -40, 120);
scene.add(sun);

let sceneBuilt = false;
let voxelMesh = null;
let voxelSize = 1;
let gridZ = [0, 1];

function buildScene(grid) {
  const lo = new THREE.Vector3(...grid.min);
  const hi = new THREE.Vector3(...grid.max);
  const center = lo.clone().add(hi).multiplyScalar(0.5);
  const ext = hi.clone().sub(lo);
  const maxExt = Math.max(ext.x, ext.y, ext.z);
  voxelSize = grid.size;
  gridZ = [lo.z, hi.z];

  const ground = new THREE.GridHelper(Math.max(ext.x, ext.y), 12, 0x2a3242, 0x1c222e);
  ground.rotation.x = Math.PI / 2;             // GridHelper is XZ; we are Z-up
  ground.position.set(center.x, center.y, 0);
  scene.add(ground);

  scene.add(new THREE.Box3Helper(new THREE.Box3(lo, hi), 0x39445a));

  const axes = new THREE.AxesHelper(0.15 * maxExt);
  axes.position.set(0, 0, 0.02);
  scene.add(axes);

  voxelMesh = new THREE.InstancedMesh(
    new THREE.BoxGeometry(voxelSize, voxelSize, voxelSize),
    new THREE.MeshLambertMaterial(), MAX_VOXELS);
  voxelMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  voxelMesh.count = 0;
  voxelMesh.frustumCulled = false;
  scene.add(voxelMesh);

  const radius = 0.5 * Math.hypot(ext.x, ext.y, ext.z);
  const az = -Math.PI / 3, el = Math.PI / 7.2, dist = 2.2 * radius;
  camera.position.set(
    center.x + dist * Math.cos(el) * Math.cos(az),
    center.y + dist * Math.cos(el) * Math.sin(az),
    dist * Math.sin(el));
  controls.target.copy(center);
  controls.update();
  sceneBuilt = true;
}

const dummy = new THREE.Object3D();
const tint = new THREE.Color();

function updateVoxels(voxels) {
  if (!voxelMesh) return;
  const n = Math.min(voxels.length, MAX_VOXELS);
  for (let i = 0; i < n; i++) {
    dummy.position.set(voxels[i][0], voxels[i][1], voxels[i][2]);
    dummy.updateMatrix();
    voxelMesh.setMatrixAt(i, dummy.matrix);
    const t = Math.min(1, Math.max(0,
      (voxels[i][2] - gridZ[0]) / (gridZ[1] - gridZ[0] || 1)));
    voxelMesh.setColorAt(i, tint.setHSL(0.66 * (1 - t), 0.95, 0.55));
  }
  voxelMesh.count = n;
  voxelMesh.instanceMatrix.needsUpdate = true;
  if (voxelMesh.instanceColor) voxelMesh.instanceColor.needsUpdate = true;
}

// Per-target 3D objects: id -> {trailGeo, trailPos, trailLen, line, arrow}
const targets3d = new Map();

function ensureTarget3d(id) {
  if (targets3d.has(id)) return targets3d.get(id);
  const trailPos = new Float32Array(TRAIL_MAX * 3);
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
  trailGeo.setDrawRange(0, 0);
  const line = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({
    color: targetColor(id), transparent: true, opacity: 1.0 }));
  line.frustumCulled = false;
  scene.add(line);
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(), 1, targetColor(id), 2.5, 1.4);
  arrow.visible = false;
  scene.add(arrow);
  const entry = { trailGeo, trailPos, trailLen: 0, line, arrow };
  targets3d.set(id, entry);
  return entry;
}

function removeTarget3d(id) {
  const entry = targets3d.get(id);
  if (!entry) return;
  scene.remove(entry.line);
  scene.remove(entry.arrow);
  entry.trailGeo.dispose();
  targets3d.delete(id);
}

function updateTarget3d(target, selected) {
  const entry = ensureTarget3d(target.id);
  if (entry.trailLen === TRAIL_MAX) {
    entry.trailPos.copyWithin(0, 3);
    entry.trailLen -= 1;
  }
  entry.trailPos.set(target.position, entry.trailLen * 3);
  entry.trailLen += 1;
  entry.trailGeo.setDrawRange(0, entry.trailLen);
  entry.trailGeo.attributes.position.needsUpdate = true;
  entry.line.material.opacity = selected ? 1.0 : 0.35;

  const v = new THREE.Vector3(...(target.velocity ?? [0, 0, 0]));
  entry.arrow.visible = false;
  if (target.speed > 0.1 && v.lengthSq() > 1e-4) {
    entry.arrow.position.set(...target.position);
    entry.arrow.setDirection(v.normalize());
    entry.arrow.setLength(target.speed,
      Math.min(3, target.speed * 0.25), Math.min(1.6, target.speed * 0.12));
    entry.arrow.visible = true;
  }
}

function clearAll3dTargets() {
  for (const id of [...targets3d.keys()]) removeTarget3d(id);
}

function resize() {
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

(function animate() {
  requestAnimationFrame(animate);
  if (renderer.domElement.width !== container.clientWidth * window.devicePixelRatio) resize();
  controls.update();
  renderer.render(scene, camera);
})();
resize();

// --------------------------------------------------------------------------
// Charts (hand-drawn sparklines, following the selected target)
// --------------------------------------------------------------------------

const charts = {
  speed: { canvas: $('chart-speed'), data: [], color: '#4ea1ff' },
  alt:   { canvas: $('chart-alt'),   data: [], color: '#35c98e' },
};

function drawChart(chart) {
  const ctx = chart.canvas.getContext('2d');
  const { width: w, height: h } = chart.canvas;
  ctx.clearRect(0, 0, w, h);
  const d = chart.data;
  if (d.length < 2) return;
  let lo = Math.min(...d), hi = Math.max(...d);
  if (hi - lo < 1e-6) { hi = lo + 1; }
  const pad = 6;
  ctx.beginPath();
  d.forEach((v, i) => {
    const x = pad + (w - 2 * pad) * i / (d.length - 1);
    const y = h - pad - (h - 2 * pad) * (v - lo) / (hi - lo);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = chart.color;
  ctx.lineWidth = 1.6;
  ctx.stroke();
  ctx.fillStyle = '#77808f';
  ctx.font = '10px Consolas, monospace';
  ctx.fillText(hi.toFixed(1), 4, 11);
  ctx.fillText(lo.toFixed(1), 4, h - 3);
  const last = d[d.length - 1];
  ctx.fillStyle = chart.color;
  ctx.fillText(last.toFixed(1), w - 34, 11);
}

function pushChart(chart, value) {
  chart.data.push(value);
  if (chart.data.length > CHART_MAX) chart.data.shift();
  drawChart(chart);
}

function clearCharts() {
  for (const c of Object.values(charts)) { c.data = []; drawChart(c); }
}

// --------------------------------------------------------------------------
// State handling
// --------------------------------------------------------------------------

let camerasBuilt = false;
let lastRunId = null;
let lastEventId = 0;
let selectedId = null;

function buildCameras(ports) {
  for (const port of ports) {
    const fig = document.createElement('figure');
    const img = document.createElement('img');
    img.src = `/stream/${port}.mjpg`;
    img.alt = `camera ${port}`;
    const cap = document.createElement('figcaption');
    cap.textContent = `CAM ${port} — undistorted | motion mask`;
    fig.append(img, cap);
    $('cameras').appendChild(fig);
  }
  camerasBuilt = true;
}

function setPill(el, text, cls) {
  el.textContent = text;
  el.className = 'pill' + (cls ? ' ' + cls : '');
}

function fmt(x, digits = 1) {
  return x === null || x === undefined ? '—' : x.toFixed(digits);
}

function camsClass(n) {
  return n >= 2 ? 'cams-ok' : (n === 1 ? 'cams-one' : 'cams-none');
}

function renderTable(targets) {
  const rows = targets.map(t => {
    const sel = t.id === selectedId ? ' class="selected"' : '';
    const cams = t.cameras.length ? t.cameras.join(',') : '—';
    return `<tr data-id="${t.id}"${sel}>` +
      `<td><span class="swatch" style="background:${targetColor(t.id)}"></span></td>` +
      `<td>${t.id}</td>` +
      `<td class="${camsClass(t.cameras.length)}">${cams}</td>` +
      `<td>${t.speed.toFixed(1)}</td>` +
      `<td>${String(Math.round(t.heading)).padStart(3, '0')}</td>` +
      `<td>${t.position[2].toFixed(0)}</td>` +
      `<td>${(t.climb >= 0 ? '+' : '') + t.climb.toFixed(1)}</td></tr>`;
  }).join('');
  $('target-rows').innerHTML = rows;
  for (const tr of $('target-rows').querySelectorAll('tr')) {
    tr.addEventListener('click', () => { selectedId = Number(tr.dataset.id); });
  }
}

function renderDetail(target) {
  if (!target) {
    $('sel-id').textContent = '—';
    $('sel-cams').textContent = 'NO TARGET';
    $('sel-cams').className = 'tag tag-idle';
    for (const id of ['t-speed','t-speed-kmh','t-heading','t-climb','t-alt']) $(id).textContent = '—';
    $('t-cardinal').textContent = '';
    $('t-pos').textContent = '—';
    return;
  }
  const n = target.cameras.length;
  $('sel-id').textContent = `#${target.id}`;
  $('sel-cams').textContent = n >= 2 ? `${n} CAMS` : (n === 1 ? '1 CAM — COASTING' : 'COASTING');
  $('sel-cams').className = 'tag ' + (n >= 2 ? 'tag-good' : (n === 1 ? 'tag-warn' : 'tag-idle'));
  $('t-speed').textContent = target.speed.toFixed(1);
  $('t-speed-kmh').textContent = (target.speed * 3.6).toFixed(0);
  $('t-heading').textContent = String(Math.round(target.heading)).padStart(3, '0');
  $('t-cardinal').textContent = CARDINALS[Math.round(target.heading / 22.5) % 16];
  $('t-climb').textContent = (target.climb >= 0 ? '+' : '') + target.climb.toFixed(1);
  $('t-alt').textContent = target.position[2].toFixed(1);
  $('t-pos').textContent =
    `x ${target.position[0].toFixed(1)}   y ${target.position[1].toFixed(1)}   z ${target.position[2].toFixed(1)}`;
}

function update(state) {
  if (!camerasBuilt && state.ports) buildCameras(state.ports);
  if (!sceneBuilt && state.grid) buildScene(state.grid);

  if (lastRunId !== null && state.run_id !== lastRunId) {
    clearAll3dTargets();
    clearCharts();
    selectedId = null;
  }
  lastRunId = state.run_id;

  setPill($('pill-projection'), state.projection ? 'PROJECTION OK' : 'NO EXTRINSICS',
          state.projection ? 'ok' : 'bad');
  setPill($('pill-voxels'), `VOXELS ${state.voxel_count}`);
  setPill($('pill-fps'), `FPS ${fmt(state.fps)}`);

  updateVoxels(state.voxels || []);

  const targets = state.targets || [];
  const count = $('target-count');
  count.textContent = String(targets.length);
  count.className = 'tag ' + (targets.length ? 'tag-good' : 'tag-idle');

  // Selection: keep if still present, else auto-select the first target
  if (!targets.some(t => t.id === selectedId)) {
    const previous = selectedId;
    selectedId = targets.length ? targets[0].id : null;
    if (previous !== selectedId) clearCharts();
  }

  renderTable(targets);
  const selected = targets.find(t => t.id === selectedId) || null;
  renderDetail(selected);

  // 3D trails/arrows: update present targets, drop vanished ones
  const present = new Set(targets.map(t => t.id));
  for (const id of [...targets3d.keys()]) {
    if (!present.has(id)) removeTarget3d(id);
  }
  for (const target of targets) updateTarget3d(target, target.id === selectedId);

  if (selected) {
    pushChart(charts.speed, selected.speed);
    pushChart(charts.alt, selected.position[2]);
  }

  // Sim panel
  const sim = state.sim || { active: false };
  $('sim-panel').hidden = !sim.active;
  if (sim.active) {
    $('sim-progress').style.width = `${100 * sim.frame / sim.total}%`;
    $('sim-frames').textContent = sim.ended ? 'ended' : `${sim.frame}/${sim.total}`;
  }

  // Events (rebuild only when something new arrived)
  const events = state.events || [];
  const newest = events.length ? events[events.length - 1].id : 0;
  if (newest !== lastEventId) {
    lastEventId = newest;
    $('events').innerHTML = events.slice().reverse().map(e => {
      const t = new Date(e.t * 1000).toLocaleTimeString();
      return `<li><span class="t">${t}</span>${e.text}</li>`;
    }).join('');
  }
}

$('sim-restart').addEventListener('click', async () => {
  $('sim-restart').disabled = true;
  try { await fetch('/api/restart', { method: 'POST' }); } catch (e) { /* ignore */ }
  setTimeout(() => { $('sim-restart').disabled = false; }, 500);
});

const source = new EventSource('/events');
source.onmessage = e => update(JSON.parse(e.data));
