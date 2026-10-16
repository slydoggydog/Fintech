import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

// Scale geometry UVs so that 1 UV unit == 1 foot; textures set their repeat
// to 1/tileFeet so they keep a consistent real-world size on every piece.
export function worldUVBox(geo, w, h, d) {
  const uv = geo.attributes.uv;
  for (let i = 0; i < uv.count; i++) {
    const face = Math.floor(i / 4);
    let su, sv;
    if (face < 2) { su = d; sv = h; }
    else if (face < 4) { su = w; sv = d; }
    else { su = w; sv = h; }
    uv.setXY(i, uv.getX(i) * su, uv.getY(i) * sv);
  }
  uv.needsUpdate = true;
  return geo;
}

export class Builder {
  constructor(scene) {
    this.scene = scene;
    this.colliders = [];
    this.surfaces = [];
    this.occluders = [];
    this.groundY = -2.5;
  }

  add(obj, { occlude = true, cast = true, receive = true } = {}) {
    obj.traverse((o) => {
      if (o.isMesh) {
        o.castShadow = cast;
        o.receiveShadow = receive;
      }
    });
    this.scene.add(obj);
    if (occlude) this.occluders.push(obj);
    return obj;
  }

  collider(x0, x1, y0, y1, z0, z1) {
    this.colliders.push({
      minX: Math.min(x0, x1), maxX: Math.max(x0, x1),
      minY: Math.min(y0, y1), maxY: Math.max(y0, y1),
      minZ: Math.min(z0, z1), maxZ: Math.max(z0, z1),
    });
  }

  colliderFromObject(obj, pad = 0) {
    obj.updateMatrixWorld(true);
    const b = new THREE.Box3().setFromObject(obj);
    this.collider(b.min.x - pad, b.max.x + pad, b.min.y, b.max.y, b.min.z - pad, b.max.z + pad);
  }

  // Axis-aligned box given by its extents.
  box(x0, x1, y0, y1, z0, z1, mat, opts = {}) {
    const w = Math.abs(x1 - x0), h = Math.abs(y1 - y0), d = Math.abs(z1 - z0);
    if (w < 1e-3 || h < 1e-3 || d < 1e-3) return null;
    const geo = worldUVBox(new THREE.BoxGeometry(w, h, d), w, h, d);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
    this.add(mesh, opts);
    if (opts.collide) this.collider(x0, x1, y0, y1, z0, z1);
    return mesh;
  }

  // Thin slab between two points in the y/z plane (roof planes, ramps).
  slopeZ(z0, y0, z1, y1, x0, x1, thick, mat, opts = {}) {
    const dz = z1 - z0, dy = y1 - y0;
    const L = Math.hypot(dz, dy);
    const w = x1 - x0;
    const geo = worldUVBox(new THREE.BoxGeometry(w, thick, L), w, thick, L);
    const mesh = new THREE.Mesh(geo, mat);
    const theta = Math.atan2(-dy, dz);
    mesh.rotation.x = theta;
    const s = Math.cos(theta) < 0 ? -1 : 1;
    const ny = s * Math.cos(theta), nz = s * Math.sin(theta);
    mesh.position.set((x0 + x1) / 2, (y0 + y1) / 2 + (ny * thick) / 2, (z0 + z1) / 2 + (nz * thick) / 2);
    return this.add(mesh, opts);
  }

  // Same as slopeZ but sloping along x (used for the small porch gable).
  slopeX(x0, y0, x1, y1, z0, z1, thick, mat, opts = {}) {
    const dx = x1 - x0, dy = y1 - y0;
    const L = Math.hypot(dx, dy);
    const d = z1 - z0;
    const geo = worldUVBox(new THREE.BoxGeometry(L, thick, d), L, thick, d);
    const mesh = new THREE.Mesh(geo, mat);
    const theta = Math.atan2(dy, dx);
    mesh.rotation.z = theta;
    const s = Math.cos(theta) < 0 ? -1 : 1;
    mesh.position.set((x0 + x1) / 2 - (s * Math.sin(theta) * thick) / 2, (y0 + y1) / 2 + (s * Math.cos(theta) * thick) / 2, (z0 + z1) / 2);
    return this.add(mesh, opts);
  }

  surface(x0, x1, z0, z1, h) {
    this.surfaces.push({
      minX: Math.min(x0, x1), maxX: Math.max(x0, x1),
      minZ: Math.min(z0, z1), maxZ: Math.max(z0, z1),
      h,
    });
  }

  floorAt(x, z, y, stepUp = 1.3) {
    let best = this.groundY;
    for (const s of this.surfaces) {
      if (x < s.minX || x > s.maxX || z < s.minZ || z > s.maxZ) continue;
      const h = typeof s.h === 'function' ? s.h(x, z) : s.h;
      if (h <= y + stepUp && h > best) best = h;
    }
    return best;
  }

  /**
   * Straight wall with door/window openings.
   * axis 'x': runs along x at z=at. matPos faces +z, matNeg faces -z.
   * axis 'z': runs along z at x=at. matPos faces +x, matNeg faces -x.
   * openings: [{a, b, bottom, top, window, glass}] in wall coordinates.
   */
  wall({ axis, at, from, to, y0 = 0, y1 = 8, t = 0.4, matPos, matNeg, edge, openings = [], trim, glass, collide = true, trimBoth = true }) {
    const lo = Math.min(from, to), hi = Math.max(from, to);
    edge = edge || matNeg;
    const mats = axis === 'x'
      ? [edge, edge, edge, edge, matPos, matNeg]
      : [matPos, matNeg, edge, edge, edge, edge];
    const piece = (a, b, ya, yb) => {
      if (b - a < 1e-3 || yb - ya < 1e-3) return;
      if (axis === 'x') this.box(a, b, ya, yb, at - t / 2, at + t / 2, mats, { collide });
      else this.box(at - t / 2, at + t / 2, ya, yb, a, b, mats, { collide });
    };
    const ops = [...openings].sort((p, q) => p.a - q.a);
    let cur = lo;
    for (const o of ops) {
      piece(cur, o.a, y0, y1);
      const bottom = o.bottom ?? y0;
      const top = o.top ?? y0 + 6.8;
      piece(o.a, o.b, y0, bottom);
      piece(o.a, o.b, top, y1);
      if (trim) this.openingTrim(axis, at, o.a, o.b, bottom, top, t, trim, !!o.window, trimBoth);
      if (o.window && glass) this.windowGlass(axis, at, o.a, o.b, bottom, top, glass, trim);
      cur = o.b;
    }
    piece(cur, hi, y0, y1);
  }

  openingTrim(axis, at, a, b, bottom, top, t, mat, isWindow, both = true) {
    const k = 0.28;
    const depth = t + 0.12;
    const z0 = both ? at - depth / 2 : at;
    const z1 = at + depth / 2;
    const P = (x0, x1, ya, yb) => {
      if (axis === 'x') this.box(x0, x1, ya, yb, z0, z1, mat, { occlude: false });
      else this.box(z0, z1, ya, yb, x0, x1, mat, { occlude: false });
    };
    P(a - k, a, bottom - (isWindow ? k : 0), top + k);
    P(b, b + k, bottom - (isWindow ? k : 0), top + k);
    P(a - k, b + k, top, top + k);
    if (isWindow) {
      const sill = axis === 'x'
        ? () => this.box(a - k * 1.5, b + k * 1.5, bottom - 0.12, bottom, at - depth / 2 - 0.15, at + depth / 2 + 0.15, mat, { occlude: false })
        : () => this.box(at - depth / 2 - 0.15, at + depth / 2 + 0.15, bottom - 0.12, bottom, a - k * 1.5, b + k * 1.5, mat, { occlude: false });
      sill();
    }
  }

  windowGlass(axis, at, a, b, bottom, top, glass, trim) {
    const g = 0.04;
    const mk = (x0, x1, ya, yb, m, th) => {
      const opts = { occlude: m === glass, cast: m !== glass };
      if (axis === 'x') this.box(x0, x1, ya, yb, at - th, at + th, m, opts);
      else this.box(at - th, at + th, ya, yb, x0, x1, m, opts);
    };
    mk(a, b, bottom, top, glass, g);
    if (!trim) return;
    const mid = (bottom + top) / 2;
    mk(a, b, mid - 0.08, mid + 0.08, trim, 0.09);
    const w = b - a;
    const panes = w > 4.5 ? Math.round(w / 2.6) : w > 3.2 ? 2 : 1;
    for (let i = 1; i < panes; i++) {
      const x = a + (w * i) / panes;
      mk(x - 0.1, x + 0.1, bottom, top, trim, 0.09);
    }
  }

  // Merge every static mesh into one mesh per (material, shadow, occluder)
  // bucket. Cuts ~1800 draw calls down to roughly one per material.
  optimize() {
    const occluderMeshes = new Set();
    for (const o of this.occluders) o.traverse((c) => c.isMesh && occluderMeshes.add(c));
    const buckets = new Map();
    const victims = [];
    this.scene.updateMatrixWorld(true);
    this.scene.traverse((o) => {
      if (!o.isMesh) return;
      victims.push(o);
      let geo = o.geometry.index ? o.geometry.toNonIndexed() : o.geometry.clone();
      for (const name of Object.keys(geo.attributes)) {
        if (!['position', 'normal', 'uv'].includes(name)) geo.deleteAttribute(name);
      }
      if (!geo.attributes.uv) {
        geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(geo.attributes.position.count * 2), 2));
      }
      geo.applyMatrix4(o.matrixWorld);
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      const groups = Array.isArray(o.material) && geo.groups.length
        ? geo.groups
        : [{ start: 0, count: geo.attributes.position.count, materialIndex: 0 }];
      for (const g of groups) {
        const material = mats[g.materialIndex ?? 0];
        if (!material) continue;
        const key = `${material.uuid}|${o.castShadow}|${o.receiveShadow}|${occluderMeshes.has(o)}`;
        if (!buckets.has(key)) buckets.set(key, { material, cast: o.castShadow, receive: o.receiveShadow, occlude: occluderMeshes.has(o), parts: [] });
        const part = new THREE.BufferGeometry();
        for (const name of ['position', 'normal', 'uv']) {
          const a = geo.attributes[name];
          part.setAttribute(name, new THREE.BufferAttribute(a.array.slice(g.start * a.itemSize, (g.start + g.count) * a.itemSize), a.itemSize));
        }
        buckets.get(key).parts.push(part);
      }
      geo.dispose();
    });
    for (const o of victims) o.removeFromParent();
    this.occluders = [];
    for (const { material, cast, receive, occlude, parts } of buckets.values()) {
      const merged = mergeGeometries(parts, false);
      parts.forEach((p) => p.dispose());
      if (!merged) continue;
      merged.computeBoundingSphere();
      merged.computeBoundingBox();
      const mesh = new THREE.Mesh(merged, material);
      mesh.castShadow = cast;
      mesh.receiveShadow = receive;
      mesh.matrixAutoUpdate = false;
      this.scene.add(mesh);
      if (occlude) this.occluders.push(mesh);
    }
  }

  // Simple railing: posts, balusters, top rail. Adds a collider.
  railing(x0, z0, x1, z1, yBase, height, mat, { spacing = 0.45, collide = true } = {}) {
    const len = Math.hypot(x1 - x0, z1 - z0);
    const group = new THREE.Group();
    const ang = Math.atan2(x1 - x0, z1 - z0);
    const rail = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.14, len), mat);
    rail.position.set(0, height, len / 2);
    group.add(rail);
    const bottom = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.1, len), mat);
    bottom.position.set(0, 0.25, len / 2);
    group.add(bottom);
    const n = Math.max(1, Math.floor(len / spacing));
    const balGeo = new THREE.BoxGeometry(0.1, height - 0.25, 0.1);
    for (let i = 1; i < n; i++) {
      const b = new THREE.Mesh(balGeo, mat);
      b.position.set(0, (height + 0.25) / 2, (len * i) / n);
      group.add(b);
    }
    const postGeo = new THREE.BoxGeometry(0.35, height + 0.25, 0.35);
    for (const p of [0, len]) {
      const post = new THREE.Mesh(postGeo, mat);
      post.position.set(0, (height + 0.25) / 2, p);
      group.add(post);
    }
    group.position.set(x0, yBase, z0);
    group.rotation.y = ang;
    this.add(group, { occlude: false });
    if (collide) {
      this.collider(Math.min(x0, x1) - 0.15, Math.max(x0, x1) + 0.15, yBase, yBase + height + 0.3, Math.min(z0, z1) - 0.15, Math.max(z0, z1) + 0.15);
    }
    return group;
  }
}
