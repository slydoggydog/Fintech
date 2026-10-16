import * as THREE from 'three';
import { Builder } from './builder.js';
import { buildHouse, roomAt, TELEPORTS } from './house.js';
import { createAvatar } from './avatar.js';

const RADIUS = 0.75;
const HEIGHT = 5.9;
const WALK = 7;
const RUN = 13;
const GRAVITY = 40;

const container = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
{
  const c = document.createElement('canvas');
  c.width = 4;
  c.height = 256;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, '#5f9bd6');
  grad.addColorStop(0.55, '#a9cff0');
  grad.addColorStop(1, '#e6f1f7');
  g.fillStyle = grad;
  g.fillRect(0, 0, 4, 256);
  const bg = new THREE.CanvasTexture(c);
  bg.colorSpace = THREE.SRGBColorSpace;
  scene.background = bg;
}
scene.fog = new THREE.Fog('#d7e8f3', 180, 520);

const camera = new THREE.PerspectiveCamera(62, window.innerWidth / window.innerHeight, 0.1, 1200);

scene.add(new THREE.HemisphereLight('#dcecff', '#7a7258', 1.1));
scene.add(new THREE.AmbientLight('#ffffff', 0.35));
const sun = new THREE.DirectionalLight('#fff3dc', 2.6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
const sc = sun.shadow.camera;
sc.left = -48; sc.right = 48; sc.top = 48; sc.bottom = -48; sc.near = 1; sc.far = 260;
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.04;
scene.add(sun, sun.target);
const SUN_OFFSET = new THREE.Vector3(55, 90, 70);

const world = new Builder(scene);
buildHouse(world);
world.optimize();

const avatar = createAvatar();
scene.add(avatar);

const player = { x: 0, y: -2.5, z: 34, vy: 0, facing: Math.PI, visualY: -2.5 };
const view = { yaw: 0, pitch: 0.28, dist: 10, targetDist: 10, firstPerson: false };

// ------------------------------------------------------------------ input
const keys = new Set();
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  keys.add(e.code);
  if (e.code === 'KeyV') toggleView();
  if (e.code === 'KeyH') toggleHelp();
  if (e.code === 'KeyM') document.getElementById('places').classList.toggle('collapsed');
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) e.preventDefault();
});
window.addEventListener('keyup', (e) => keys.delete(e.code));
window.addEventListener('blur', () => keys.clear());

let dragging = null;
const canvas = renderer.domElement;
canvas.addEventListener('pointerdown', (e) => {
  if (e.pointerType === 'touch') return;
  dragging = { x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', (e) => {
  if (!dragging || e.pointerType === 'touch') return;
  look(e.clientX - dragging.x, e.clientY - dragging.y);
  dragging = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener('pointerup', () => { dragging = null; });
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  view.targetDist = THREE.MathUtils.clamp(view.targetDist + e.deltaY * 0.01, 3, 22);
  if (view.firstPerson) setFirstPerson(false);
}, { passive: false });

function look(dx, dy) {
  view.yaw -= dx * 0.005;
  view.pitch = THREE.MathUtils.clamp(view.pitch + dy * 0.004, -0.9, 1.25);
}

// Touch: left half = movement joystick, right half = look.
const joy = { id: null, ox: 0, oy: 0, dx: 0, dy: 0 };
const lookTouch = { id: null, x: 0, y: 0 };
const joyEl = document.getElementById('joystick');
const knobEl = document.getElementById('knob');
canvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  for (const t of e.changedTouches) {
    if (t.clientX < window.innerWidth / 2 && joy.id === null) {
      Object.assign(joy, { id: t.identifier, ox: t.clientX, oy: t.clientY, dx: 0, dy: 0 });
      joyEl.style.left = `${t.clientX - 60}px`;
      joyEl.style.top = `${t.clientY - 60}px`;
      joyEl.classList.add('active');
    } else if (lookTouch.id === null) {
      Object.assign(lookTouch, { id: t.identifier, x: t.clientX, y: t.clientY });
    }
  }
}, { passive: false });
canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  for (const t of e.changedTouches) {
    if (t.identifier === joy.id) {
      let dx = t.clientX - joy.ox, dy = t.clientY - joy.oy;
      const len = Math.hypot(dx, dy);
      if (len > 50) { dx *= 50 / len; dy *= 50 / len; }
      joy.dx = dx / 50;
      joy.dy = dy / 50;
      knobEl.style.transform = `translate(${dx}px, ${dy}px)`;
    } else if (t.identifier === lookTouch.id) {
      look((t.clientX - lookTouch.x) * 1.4, (t.clientY - lookTouch.y) * 1.4);
      lookTouch.x = t.clientX;
      lookTouch.y = t.clientY;
    }
  }
}, { passive: false });
const endTouch = (e) => {
  for (const t of e.changedTouches) {
    if (t.identifier === joy.id) {
      Object.assign(joy, { id: null, dx: 0, dy: 0 });
      knobEl.style.transform = '';
      joyEl.classList.remove('active');
    }
    if (t.identifier === lookTouch.id) lookTouch.id = null;
  }
};
canvas.addEventListener('touchend', endTouch);
canvas.addEventListener('touchcancel', endTouch);

// --------------------------------------------------------------------- HUD
const roomEl = document.getElementById('room');
const viewBtn = document.getElementById('view-btn');
const placesEl = document.getElementById('place-list');
let currentRoom = '';
let roomTimer = 0;

function setFirstPerson(on) {
  view.firstPerson = on;
  viewBtn.textContent = on ? 'Third-person view' : 'First-person view';
}
function toggleView() { setFirstPerson(!view.firstPerson); }
viewBtn.addEventListener('click', toggleView);

function toggleHelp() { document.getElementById('help').classList.toggle('hidden'); }
document.getElementById('help-btn').addEventListener('click', toggleHelp);

for (const t of TELEPORTS) {
  const btn = document.createElement('button');
  btn.textContent = t.name;
  btn.addEventListener('click', () => teleport(t));
  placesEl.appendChild(btn);
}
document.getElementById('places-toggle').addEventListener('click', () => {
  document.getElementById('places').classList.toggle('collapsed');
});

function teleport(t) {
  [player.x, player.y, player.z] = t.pos;
  player.visualY = player.y;
  player.vy = 0;
  view.yaw = t.yaw;
  player.facing = t.yaw + Math.PI;
  view.pitch = 0.22;
}

const outfits = [
  { shirt: '#3d6fa8', pants: '#2f3540' },
  { shirt: '#b5473a', pants: '#3a3f4a' },
  { shirt: '#2f6e4f', pants: '#c8b48e' },
  { shirt: '#f0eee6', pants: '#3b4f72' },
  { shirt: '#7a4d8c', pants: '#2c2c2c' },
];
let outfit = 0;
document.getElementById('outfit-btn').addEventListener('click', () => {
  outfit = (outfit + 1) % outfits.length;
  avatar.userData.setColors(outfits[outfit]);
});

document.getElementById('start').addEventListener('click', () => {
  document.getElementById('intro').classList.add('gone');
  canvas.focus();
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --------------------------------------------------------------- movement
function collide(px, pz, y) {
  let x = px, z = pz;
  for (let iter = 0; iter < 3; iter++) {
    let moved = false;
    for (const c of world.colliders) {
      if (c.maxY <= y + 1.3 || c.minY >= y + HEIGHT - 0.3) continue;
      const cx = Math.max(c.minX, Math.min(x, c.maxX));
      const cz = Math.max(c.minZ, Math.min(z, c.maxZ));
      const dx = x - cx, dz = z - cz;
      const d2 = dx * dx + dz * dz;
      if (d2 >= RADIUS * RADIUS) continue;
      if (d2 > 1e-8) {
        const d = Math.sqrt(d2);
        x = cx + (dx / d) * RADIUS;
        z = cz + (dz / d) * RADIUS;
      } else {
        const pushes = [[c.minX - RADIUS - x, 0], [c.maxX + RADIUS - x, 0], [0, c.minZ - RADIUS - z], [0, c.maxZ + RADIUS - z]];
        pushes.sort((a, b) => Math.hypot(...a) - Math.hypot(...b));
        x += pushes[0][0];
        z += pushes[0][1];
      }
      moved = true;
    }
    if (!moved) break;
  }
  return [x, z];
}

function update(dt) {
  let fwd = 0, strafe = 0;
  if (keys.has('KeyW') || keys.has('ArrowUp')) fwd += 1;
  if (keys.has('KeyS') || keys.has('ArrowDown')) fwd -= 1;
  if (keys.has('KeyA') || keys.has('ArrowLeft')) strafe -= 1;
  if (keys.has('KeyD') || keys.has('ArrowRight')) strafe += 1;
  if (keys.has('KeyQ')) view.yaw += dt * 1.8;
  if (keys.has('KeyE')) view.yaw -= dt * 1.8;
  fwd -= joy.dy;
  strafe += joy.dx;

  const sin = Math.sin(view.yaw), cos = Math.cos(view.yaw);
  let mx = -sin * fwd + cos * strafe;
  let mz = -cos * fwd - sin * strafe;
  const mag = Math.hypot(mx, mz);
  const running = keys.has('ShiftLeft') || keys.has('ShiftRight') || Math.hypot(joy.dx, joy.dy) > 0.95;
  const speed = (running ? RUN : WALK) * Math.min(1, mag);
  if (mag > 0.05) {
    mx /= mag;
    mz /= mag;
    const steps = Math.ceil((speed * dt) / 0.3);
    for (let i = 0; i < steps; i++) {
      const nx = player.x + (mx * speed * dt) / steps;
      const nz = player.z + (mz * speed * dt) / steps;
      const floor = world.floorAt(nx, nz, player.y);
      [player.x, player.z] = collide(nx, nz, Math.max(player.y, floor));
    }
    const target = Math.atan2(mx, mz);
    let diff = target - player.facing;
    diff = Math.atan2(Math.sin(diff), Math.cos(diff));
    player.facing += diff * Math.min(1, dt * 12);
  }

  const floor = world.floorAt(player.x, player.z, player.y);
  if (floor >= player.y - 0.7 && player.vy <= 0) {
    player.y = floor;
    player.vy = 0;
  } else {
    player.vy -= GRAVITY * dt;
    player.y += player.vy * dt;
    if (player.y <= floor) {
      player.y = floor;
      player.vy = 0;
    }
  }
  player.visualY += (player.y - player.visualY) * Math.min(1, dt * 14);

  avatar.position.set(player.x, player.visualY, player.z);
  avatar.rotation.y = player.facing;
  avatar.userData.animate(dt, mag > 0.05 ? speed : 0);

  const room = roomAt(player.x, player.y, player.z);
  if (room !== currentRoom) {
    currentRoom = room;
    roomEl.textContent = room;
    roomEl.classList.add('show');
    roomTimer = 3.5;
  }
  if (roomTimer > 0) {
    roomTimer -= dt;
    if (roomTimer <= 0) roomEl.classList.remove('show');
  }
}

// ----------------------------------------------------------------- camera
const raycaster = new THREE.Raycaster();
const head = new THREE.Vector3();
const desired = new THREE.Vector3();
const dir = new THREE.Vector3();
const side = new THREE.Vector3();
const SHOULDER = 1.1;

function updateCamera(dt) {
  head.set(player.x, player.visualY + (view.firstPerson ? 5.45 : 5.1), player.z);
  const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  dir.set(Math.sin(view.yaw) * cp, sp, Math.cos(view.yaw) * cp);

  if (view.firstPerson) {
    camera.position.copy(head);
    camera.lookAt(head.x - dir.x, head.y - dir.y, head.z - dir.z);
    avatar.visible = false;
    return;
  }

  view.dist += (view.targetDist - view.dist) * Math.min(1, dt * 8);
  let dist = view.dist;
  // Over-the-shoulder offset, shortened if a wall is right beside the player.
  side.set(Math.cos(view.yaw), 0, -Math.sin(view.yaw));
  raycaster.set(head, side);
  raycaster.far = SHOULDER + 0.4;
  const sideHit = raycaster.intersectObjects(world.occluders, true)[0];
  const shoulder = sideHit ? Math.max(0, sideHit.distance - 0.4) : SHOULDER;
  head.addScaledVector(side, shoulder);
  raycaster.set(head, dir);
  raycaster.far = dist + 0.5;
  const hits = raycaster.intersectObjects(world.occluders, true);
  if (hits.length) dist = Math.max(0.3, Math.min(dist, hits[0].distance - 0.45));
  desired.copy(head).addScaledVector(dir, dist);
  camera.position.copy(desired);
  camera.lookAt(head);
  avatar.visible = dist > 0.9;
  avatar.userData.setOpacity(THREE.MathUtils.clamp((dist - 1.2) / 3.6, 0.15, 1));
}

// ------------------------------------------------------------------- loop
const clock = new THREE.Clock();
function frame() {
  const dt = Math.min(0.05, clock.getDelta());
  update(dt);
  updateCamera(dt);
  sun.target.position.set(player.x, 0, player.z);
  sun.position.copy(sun.target.position).add(SUN_OFFSET);
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
updateCamera(0);
frame();

// Debug hook: advance the simulation with fixed steps (used by tests).
window.__walkthrough = {
  player, view, teleport, TELEPORTS, renderer, camera,
  simulate(codes, seconds) {
    codes.forEach((c) => keys.add(c));
    for (let t = 0; t < seconds; t += 1 / 60) update(1 / 60);
    codes.forEach((c) => keys.delete(c));
    return { x: player.x, y: player.y, z: player.z, room: currentRoom };
  },
};
