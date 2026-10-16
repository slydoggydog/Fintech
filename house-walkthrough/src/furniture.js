import * as THREE from 'three';
import { worldUVBox } from './builder.js';

const cache = new Map();
export function mat(color, opts = {}) {
  const key = `${color}|${JSON.stringify(opts)}`;
  if (!cache.has(key)) cache.set(key, new THREE.MeshStandardMaterial({ color, roughness: 0.8, ...opts }));
  return cache.get(key);
}

// Furniture is modeled facing +z; `rot` turns it around y.
function part(group, w, h, d, x, y, z, material) {
  const geo = worldUVBox(new THREE.BoxGeometry(w, h, d), w, h, d);
  const m = new THREE.Mesh(geo, material);
  m.position.set(x, y + h / 2, z);
  group.add(m);
  return m;
}

function cyl(group, rTop, rBot, h, x, y, z, material, seg = 16) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBot, h, seg), material);
  m.position.set(x, y + h / 2, z);
  group.add(m);
  return m;
}

function place(b, group, x, y, z, rot, { collide = true, pad = 0 } = {}) {
  group.position.set(x, y, z);
  group.rotation.y = rot;
  b.add(group, { occlude: false });
  if (collide) b.colliderFromObject(group, pad);
  return group;
}

const WOOD = () => mat('#8a5a35');
const WHITE = () => mat('#f4f3ee');

export function sofa(b, x, z, rot, { w = 7, color = '#24406b', y = 0, pillows = '#e9eef3', legs = '#3b2a1f', armColor } = {}) {
  const g = new THREE.Group();
  const fab = mat(color);
  const arm = armColor ? mat(armColor) : fab;
  part(g, w, 0.35, 3, 0, 0.3, 0, fab);
  part(g, w - 1, 0.55, 2.4, 0, 0.65, 0.25, fab);
  part(g, w, 1.6, 0.7, 0, 0.65, -1.15, fab);
  part(g, 0.6, 1.2, 3, -w / 2 + 0.3, 0.65, 0, arm);
  part(g, 0.6, 1.2, 3, w / 2 - 0.3, 0.65, 0, arm);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) part(g, 0.2, 0.3, 0.2, sx * (w / 2 - 0.3), 0, sz * 1.2, mat(legs));
  part(g, 1.3, 1.1, 0.35, -w / 2 + 1.3, 1.2, -0.65, mat(pillows));
  part(g, 1.3, 1.1, 0.35, w / 2 - 1.3, 1.2, -0.65, mat(pillows));
  return place(b, g, x, y, z, rot);
}

export function armchair(b, x, z, rot, { color = '#e8e2d4', stripe, y = 0 } = {}) {
  const g = new THREE.Group();
  const fab = stripe ? new THREE.MeshStandardMaterial({ map: stripe, roughness: 0.9 }) : mat(color);
  part(g, 2.6, 0.5, 2.6, 0, 0.9, 0, fab);
  part(g, 2.6, 2.2, 0.45, 0, 1.2, -1.1, fab);
  part(g, 0.35, 0.9, 2.4, -1.15, 1.3, 0, fab);
  part(g, 0.35, 0.9, 2.4, 1.15, 1.3, 0, fab);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) part(g, 0.18, 0.9, 0.18, sx * 1.1, 0, sz * 1.1, WOOD());
  return place(b, g, x, y, z, rot);
}

export function recliner(b, x, z, rot, { color = '#8f9499' } = {}) {
  const g = new THREE.Group();
  const fab = mat(color);
  part(g, 3, 1.4, 3, 0, 0, 0.2, fab);
  part(g, 3, 2.1, 0.9, 0, 1.4, -1.05, fab);
  part(g, 0.6, 0.8, 3, -1.2, 1.4, 0.2, fab);
  part(g, 0.6, 0.8, 3, 1.2, 1.4, 0.2, fab);
  return place(b, g, x, 0, z, rot);
}

export function wickerSofa(b, x, z, rot, { w = 6.5, y = 0, cushion = '#1f3a6b' } = {}) {
  const g = new THREE.Group();
  const wick = mat('#e3d6bb');
  part(g, w, 1.0, 2.8, 0, 0.3, 0, wick);
  part(g, w, 1.8, 0.4, 0, 0.3, -1.2, wick);
  part(g, 0.4, 1.9, 2.8, -w / 2 + 0.2, 0.3, 0, wick);
  part(g, 0.4, 1.9, 2.8, w / 2 - 0.2, 0.3, 0, wick);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) part(g, 0.2, 0.3, 0.2, sx * (w / 2 - 0.3), 0, sz * 1.2, wick);
  part(g, w - 0.9, 0.4, 2.2, 0, 1.3, 0.15, mat(cushion));
  part(g, w - 0.9, 1.2, 0.35, 0, 1.6, -0.85, mat(cushion));
  return place(b, g, x, y, z, rot);
}

export function wickerChair(b, x, z, rot, { y = 0, stripe } = {}) {
  const g = new THREE.Group();
  const wick = mat('#e3d6bb');
  part(g, 2.4, 1.0, 2.4, 0, 0.3, 0, wick);
  part(g, 2.4, 1.7, 0.35, 0, 0.3, -1.05, wick);
  part(g, 0.3, 1.6, 2.4, -1.05, 0.3, 0, wick);
  part(g, 0.3, 1.6, 2.4, 1.05, 0.3, 0, wick);
  const c = stripe ? new THREE.MeshStandardMaterial({ map: stripe }) : mat('#1f3a6b');
  part(g, 1.8, 0.35, 1.9, 0, 1.3, 0.1, c);
  part(g, 1.8, 1.1, 0.3, 0, 1.6, -0.75, c);
  return place(b, g, x, y, z, rot);
}

export function table(b, x, z, rot, { w = 3, d = 2, h = 2.5, y = 0, top = '#8a5a35', legs, glass = false } = {}) {
  const g = new THREE.Group();
  const tm = glass ? mat('#cfe3ea', { transparent: true, opacity: 0.5 }) : mat(top);
  part(g, w, 0.12, d, 0, h - 0.12, 0, tm);
  const lm = mat(legs || top);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) part(g, 0.15, h - 0.12, 0.15, sx * (w / 2 - 0.15), 0, sz * (d / 2 - 0.15), lm);
  return place(b, g, x, y, z, rot);
}

export function roundTable(b, x, z, { r = 1, h = 2.2, top = '#f2f0ea', y = 0, collide = true } = {}) {
  const g = new THREE.Group();
  cyl(g, r, r, 0.1, 0, h - 0.1, 0, mat(top));
  cyl(g, 0.1, 0.1, h - 0.1, 0, 0, 0, mat(top));
  cyl(g, r * 0.5, r * 0.6, 0.1, 0, 0, 0, mat(top));
  return place(b, g, x, y, z, 0, { collide });
}

export function lamp(b, x, y, z, { base = '#6d9bb8', shade = '#f7f2e6', light = true } = {}) {
  const g = new THREE.Group();
  cyl(g, 0.25, 0.4, 1.1, 0, 0, 0, mat(base, { roughness: 0.4 }));
  cyl(g, 0.55, 0.75, 0.9, 0, 1.2, 0, mat(shade, { emissive: light ? '#ffdca8' : '#000000', emissiveIntensity: 0.6, side: THREE.DoubleSide }));
  return place(b, g, x, y, z, 0, { collide: false });
}

export function floorLamp(b, x, z, { y = 0 } = {}) {
  const g = new THREE.Group();
  cyl(g, 0.5, 0.6, 0.15, 0, 0, 0, mat('#c79a5d'));
  cyl(g, 0.15, 0.15, 4.4, 0, 0.15, 0, mat('#c79a5d'));
  cyl(g, 0.7, 0.85, 1.0, 0, 4.5, 0, mat('#f6efdc', { emissive: '#ffdca8', emissiveIntensity: 0.7, side: THREE.DoubleSide }));
  return place(b, g, x, y, z, 0, { collide: false });
}

export function chair(b, x, z, rot, { color = '#ece4d0', cushion = '#2c3e5c', y = 0 } = {}) {
  const g = new THREE.Group();
  const m = mat(color);
  part(g, 1.5, 0.12, 1.5, 0, 1.45, 0, m);
  part(g, 1.4, 0.12, 1.4, 0, 1.57, 0, mat(cushion));
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) part(g, 0.1, 1.45, 0.1, sx * 0.62, 0, sz * 0.62, m);
  for (let i = -2; i <= 2; i++) part(g, 0.07, 1.6, 0.07, i * 0.28, 1.57, -0.68, m);
  part(g, 1.4, 0.16, 0.12, 0, 3.15, -0.68, m);
  return place(b, g, x, y, z, rot, { collide: false });
}

export function diningTable(b, x, z, rot) {
  const g = new THREE.Group();
  const m = mat('#e9e0cc');
  const top = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 0.14, 40), m);
  top.scale.set(3, 1, 1.9);
  top.position.y = 2.45;
  g.add(top);
  for (const sx of [-1, 1]) cyl(g, 0.14, 0.18, 2.4, sx * 1.6, 0, 0, m);
  part(g, 3.6, 0.15, 0.3, 0, 0.5, 0, m);
  place(b, g, x, 0, z, rot);
  const c = Math.cos(rot), s = Math.sin(rot);
  const at = (lx, lz) => [x + lx * c + lz * s, z - lx * s + lz * c];
  const seats = [[-1.4, 1.9, 0], [1.4, 1.9, 0], [-1.4, -1.9, Math.PI], [1.4, -1.9, Math.PI], [3.6, 0, -Math.PI / 2], [-3.6, 0, Math.PI / 2]];
  for (const [lx, lz, r] of seats) {
    const [wx, wz] = at(lx, lz);
    chair(b, wx, wz, rot + r + Math.PI);
  }
}

export function stool(b, x, z, { y = 0 } = {}) {
  const g = new THREE.Group();
  const m = mat('#2d3540', { roughness: 0.5, metalness: 0.4 });
  cyl(g, 0.6, 0.6, 0.15, 0, 2.6, 0, m);
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2 + Math.PI / 4;
    const leg = part(g, 0.08, 2.6, 0.08, Math.cos(a) * 0.45, 0, Math.sin(a) * 0.45, m);
    leg.rotation.set(Math.sin(a) * 0.1, 0, -Math.cos(a) * 0.1);
  }
  part(g, 0.9, 1.0, 0.1, 0, 2.75, -0.5, m);
  return place(b, g, x, y, z, 0, { collide: false });
}

export function bed(b, x, z, rot, { w = 5, l = 6.8, quilt, sheet = '#f3f1ea', frame = '#6b4a30', y = 0, pillows = 2 } = {}) {
  const g = new THREE.Group();
  part(g, w + 0.3, 0.9, l + 0.2, 0, 0.3, 0, mat(frame));
  part(g, w, 0.9, l, 0, 1.2, 0, mat(sheet));
  const q = quilt ? new THREE.MeshStandardMaterial({ map: quilt, roughness: 0.95 }) : mat('#8fa9c4');
  part(g, w + 0.15, 0.35, l * 0.7, 0, 1.95, l * 0.15 + 0.05, q);
  part(g, w + 0.3, 3.4, 0.3, 0, 0, -l / 2 - 0.15, mat(frame));
  for (let i = 0; i < pillows; i++) {
    const px = pillows === 1 ? 0 : (i - (pillows - 1) / 2) * (w / pillows);
    part(g, w / pillows - 0.3, 0.45, 1.2, px, 2.1, -l / 2 + 0.9, mat('#ffffff'));
  }
  return place(b, g, x, y, z, rot);
}

export function dresser(b, x, z, rot, { w = 4, h = 3, d = 1.6, color = '#7a5236', y = 0 } = {}) {
  const g = new THREE.Group();
  const m = mat(color);
  part(g, w, h, d, 0, 0.2, 0, m);
  const rows = Math.max(2, Math.round(h / 0.9));
  for (let i = 0; i < rows; i++) {
    const yy = 0.35 + (i * (h - 0.3)) / rows;
    part(g, w - 0.3, (h - 0.5) / rows - 0.12, 0.05, 0, yy, d / 2, mat(new THREE.Color(color).offsetHSL(0, 0, 0.04).getStyle()));
    part(g, 0.4, 0.08, 0.08, 0, yy + 0.3, d / 2 + 0.05, mat('#c9b17a', { metalness: 0.6, roughness: 0.4 }));
  }
  part(g, w, 0.2, d, 0, 0, 0, m);
  return place(b, g, x, y, z, rot);
}

export function tv(b, x, y, z, rot, { w = 4, h = 2.3 } = {}) {
  const g = new THREE.Group();
  part(g, w, h, 0.15, 0, 0, 0, mat('#111214', { roughness: 0.3 }));
  part(g, w - 0.15, h - 0.15, 0.02, 0, 0.075, 0.08, mat('#1d2633', { roughness: 0.1, emissive: '#0e1622', emissiveIntensity: 0.6 }));
  return place(b, g, x, y, z, rot, { collide: false });
}

export function hutch(b, x, z, rot, { color = '#8f9aa1' } = {}) {
  const g = new THREE.Group();
  const m = mat(color);
  part(g, 3, 3, 1.6, 0, 0, 0, m);
  part(g, 2.8, 3.2, 1.2, 0, 3, -0.15, m);
  part(g, 2.4, 2.7, 0.05, 0, 3.2, 0.47, mat('#cfe0e8', { transparent: true, opacity: 0.4 }));
  part(g, 3.2, 0.2, 1.4, 0, 6.2, -0.1, m);
  return place(b, g, x, 0, z, rot);
}

export function shelfConsole(b, x, z, rot, { w = 4, h = 2.7, d = 1.1, y = 0 } = {}) {
  const g = new THREE.Group();
  const m = mat('#2e2a26', { metalness: 0.5, roughness: 0.5 });
  for (const yy of [0.4, 1.5, h - 0.1]) part(g, w, 0.08, d, 0, yy, 0, mat('#6b4a30'));
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) part(g, 0.08, h, 0.08, sx * (w / 2 - 0.05), 0, sz * (d / 2 - 0.05), m);
  return place(b, g, x, y, z, rot);
}

export function toilet(b, x, z, rot, { y = 0 } = {}) {
  const g = new THREE.Group();
  const m = mat('#fbfbf8', { roughness: 0.25 });
  cyl(g, 0.6, 0.45, 1.3, 0, 0, 0.2, m);
  part(g, 1.6, 1.3, 0.7, 0, 1.3, -0.55, m);
  return place(b, g, x, y, z, rot);
}

export function tub(b, x0, x1, z0, z1, { y = 0 } = {}) {
  const g = new THREE.Group();
  const m = mat('#fbfbf8', { roughness: 0.25 });
  const w = x1 - x0, d = z1 - z0;
  part(g, w, 1.8, d, 0, 0, 0, m);
  part(g, w - 0.5, 0.05, d - 0.5, 0, 1.76, 0, mat('#dfe9ee', { roughness: 0.1 }));
  return place(b, g, (x0 + x1) / 2, y, (z0 + z1) / 2, 0);
}

export function vanity(b, x, z, rot, { w = 3, y = 0, color = '#f2f0ea' } = {}) {
  const g = new THREE.Group();
  part(g, w, 2.6, 1.8, 0, 0, 0, mat(color));
  part(g, w + 0.1, 0.15, 1.9, 0, 2.6, 0, mat('#f6f6f2', { roughness: 0.3 }));
  part(g, 1.2, 0.05, 0.9, 0, 2.72, 0.1, mat('#dfe5e8', { roughness: 0.1 }));
  part(g, w * 0.8, 2.2, 0.08, 0, 3.6, -0.86, mat('#c9d8e0', { roughness: 0.05, metalness: 0.9 }));
  return place(b, g, x, y, z, rot);
}

export function fridge(b, x, z, rot) {
  const g = new THREE.Group();
  const m = mat('#f7f7f5', { roughness: 0.35 });
  part(g, 2.6, 5.8, 2.4, 0, 0, 0, m);
  part(g, 2.62, 0.04, 2.42, 0, 4.1, 0, mat('#c7c7c7'));
  part(g, 0.1, 1.2, 0.12, -1.1, 4.4, 1.25, mat('#d9d9d9'));
  part(g, 0.1, 1.8, 0.12, -1.1, 2.0, 1.25, mat('#d9d9d9'));
  return place(b, g, x, 0, z, rot);
}

export function range(b, x, z, rot) {
  const g = new THREE.Group();
  const m = mat('#f5f5f3', { roughness: 0.35 });
  part(g, 2.5, 3, 2.1, 0, 0, 0, m);
  part(g, 2.5, 0.05, 2.1, 0, 3, 0, mat('#2a2a2a', { roughness: 0.2 }));
  for (const sx of [-0.6, 0.6]) for (const sz of [-0.45, 0.45]) cyl(g, 0.3, 0.3, 0.03, sx, 3.05, sz, mat('#111111'));
  part(g, 2.5, 0.8, 0.2, 0, 3, -0.95, m);
  part(g, 2.1, 1.2, 0.05, 0, 0.9, 1.06, mat('#1a1a1a', { roughness: 0.1 }));
  return place(b, g, x, 0, z, rot);
}

export function rug(b, x0, x1, z0, z1, y, material) {
  return b.box(x0, x1, y, y + 0.03, z0, z1, material, { occlude: false, cast: false });
}

export function picture(b, axis, at, c, yC, w, h, facing, texture) {
  const g = new THREE.Group();
  part(g, w + 0.25, h + 0.25, 0.1, 0, -h / 2 - 0.125, 0, mat('#4a3020'));
  const img = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshStandardMaterial({ map: texture, roughness: 0.6 }));
  img.position.set(0, 0, 0.06);
  g.add(img);
  let x, z, rot;
  if (axis === 'x') { x = c; z = at + facing * 0.08; rot = facing > 0 ? 0 : Math.PI; }
  else { x = at + facing * 0.08; z = c; rot = facing > 0 ? Math.PI / 2 : -Math.PI / 2; }
  return place(b, g, x, yC, z, rot, { collide: false });
}

export function ceilingLight(b, x, y, z, { r = 0.6, chandelier = false } = {}) {
  const g = new THREE.Group();
  const glow = mat('#fff6e0', { emissive: '#ffe7b8', emissiveIntensity: 1.2 });
  if (chandelier) {
    cyl(g, 0.04, 0.04, 1.0, 0, -1.0, 0, mat('#b8934a', { metalness: 0.8, roughness: 0.3 }));
    cyl(g, 0.9, 0.5, 1.0, 0, -2.0, 0, mat('#f3e3b5', { emissive: '#ffd890', emissiveIntensity: 0.8, transparent: true, opacity: 0.9 }), 8);
  } else {
    cyl(g, r, r * 0.8, 0.35, 0, -0.35, 0, glow);
  }
  return place(b, g, x, y, z, 0, { collide: false });
}

export function tree(b, x, z, { trunk = 6, crown = 7, color = '#4f7f35', y = -2.5, lean = 0, maple = false, seed = 1 } = {}) {
  const g = new THREE.Group();
  const tm = mat('#5b4636', { roughness: 1 });
  const t = new THREE.Mesh(new THREE.CylinderGeometry(trunk * 0.06, trunk * 0.09, trunk + crown * 0.4, 8), tm);
  t.position.y = (trunk + crown * 0.4) / 2;
  t.rotation.z = lean;
  g.add(t);
  let s = seed;
  const r = () => ((s = (s * 9301 + 49297) % 233280) / 233280);
  const colors = maple ? ['#7c8a3a', '#9a6a2e', '#a3472f', '#5f7d33'] : [color, new THREE.Color(color).offsetHSL(0, 0, 0.05).getStyle(), new THREE.Color(color).offsetHSL(0.02, 0, -0.05).getStyle()];
  const blobs = maple ? 9 : 6;
  for (let i = 0; i < blobs; i++) {
    const rad = crown * (0.35 + r() * 0.25);
    const m = new THREE.Mesh(new THREE.IcosahedronGeometry(rad, 1), mat(colors[i % colors.length], { flatShading: true, roughness: 1 }));
    const a = r() * Math.PI * 2;
    const d = crown * 0.45 * r();
    m.position.set(Math.cos(a) * d + lean * -trunk, trunk + crown * 0.3 + r() * crown * 0.5, Math.sin(a) * d);
    m.scale.y = maple ? 0.65 : 0.9;
    g.add(m);
  }
  g.position.set(x, y, z);
  b.add(g, { occlude: false });
  b.collider(x - trunk * 0.1, x + trunk * 0.1, y, y + trunk, z - trunk * 0.1, z + trunk * 0.1);
  return g;
}

export function bush(b, x, z, { r = 1.6, color = '#3f6e2c', flowers, y = -2.5 } = {}) {
  const g = new THREE.Group();
  const m = new THREE.Mesh(new THREE.IcosahedronGeometry(r, 1), mat(color, { flatShading: true, roughness: 1 }));
  m.scale.y = 0.75;
  m.position.y = r * 0.6;
  g.add(m);
  if (flowers) {
    for (let i = 0; i < 6; i++) {
      const f = new THREE.Mesh(new THREE.IcosahedronGeometry(r * 0.22, 0), mat(flowers, { flatShading: true }));
      const a = (i / 6) * Math.PI * 2;
      f.position.set(Math.cos(a) * r * 0.75, r * 0.85, Math.sin(a) * r * 0.75);
      g.add(f);
    }
  }
  g.position.set(x, y, z);
  b.add(g, { occlude: false });
  return g;
}

export function hedge(b, x0, x1, z0, z1, h, y = -2.5) {
  const m = mat('#355f27', { roughness: 1 });
  b.box(x0, x1, y, y + h, z0, z1, m, { collide: true, occlude: false });
}

export function picnicTable(b, x, z, rot, { y = -2.5 } = {}) {
  const g = new THREE.Group();
  const m = mat('#5d6166');
  part(g, 6, 0.15, 2.4, 0, 2.4, 0, m);
  for (const sz of [-1.9, 1.9]) part(g, 6, 0.12, 0.9, 0, 1.45, sz, m);
  for (const sx of [-2.4, 2.4]) {
    part(g, 0.25, 2.4, 0.25, sx, 0, -0.8, m);
    part(g, 0.25, 2.4, 0.25, sx, 0, 0.8, m);
    part(g, 0.25, 0.2, 4.6, sx, 1.25, 0, m);
  }
  return place(b, g, x, y, z, rot);
}

export function shed(b, x, z, { y = -2.5 } = {}) {
  const g = new THREE.Group();
  const wall = mat('#e7ebec');
  part(g, 8, 7, 10, 0, 0, 0, wall);
  part(g, 3, 6, 0.1, 0, 0, 5.02, mat('#f7f7f5'));
  const roof = mat('#62666b');
  const r1 = part(g, 5, 0.3, 11, -2.1, 7.6, 0, roof);
  r1.rotation.z = 0.55;
  const r2 = part(g, 5, 0.3, 11, 2.1, 7.6, 0, roof);
  r2.rotation.z = -0.55;
  const gable = new THREE.Shape([new THREE.Vector2(-4, 0), new THREE.Vector2(4, 0), new THREE.Vector2(0, 2.4)]);
  for (const sz of [-5, 5]) {
    const gm = new THREE.Mesh(new THREE.ShapeGeometry(gable), new THREE.MeshStandardMaterial({ color: '#e7ebec', side: THREE.DoubleSide }));
    gm.position.set(0, 7, sz);
    g.add(gm);
  }
  return place(b, g, x, y, z, 0);
}

export function neighborHouse(b, x, z, { w = 26, d = 34, h = 18, color = '#dfe6ea', roof = '#5d6168', rot = 0 } = {}) {
  const g = new THREE.Group();
  part(g, w, h, d, 0, 0, 0, mat(color));
  const shape = new THREE.Shape([new THREE.Vector2(-w / 2 - 1, 0), new THREE.Vector2(w / 2 + 1, 0), new THREE.Vector2(0, w * 0.38)]);
  const geo = new THREE.ExtrudeGeometry(shape, { depth: d + 2, bevelEnabled: false });
  const rm = new THREE.Mesh(geo, mat(roof, { flatShading: true }));
  rm.position.set(0, h, -d / 2 - 1);
  g.add(rm);
  const win = mat('#3b4a57', { roughness: 0.2 });
  for (const wy of [3, 11]) {
    for (const wx of [-w / 3, 0, w / 3]) {
      part(g, 2.6, 4, 0.1, wx, wy, d / 2 + 0.03, win);
      part(g, 0.1, 4, 2.6, w / 2 + 0.03, wy, wx * 0.9, win);
      part(g, 0.1, 4, 2.6, -w / 2 - 0.03, wy, wx * 0.9, win);
    }
  }
  return place(b, g, x, -2.5, z, rot);
}
