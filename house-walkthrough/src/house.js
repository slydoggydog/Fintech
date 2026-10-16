import * as THREE from 'three';
import * as T from './textures.js';
import * as F from './furniture.js';
import { mat } from './furniture.js';

// Layout notes (feet). Origin is the center of the main house footprint at
// first-floor level. +z points to Jersey Avenue, +x is the chimney side.
//   First floor  y = 0, ceiling 8, upstairs floor 8.5, upstairs ceiling 16.
//   Grade        y = -2.5 (five steps up to the front porch).
//   Main block   x -19..19, z -15..15; screened porch tucked under the main
//                roof at the back right (x 6..19, z -15..-3), one step down.

const RISE = 8.5 / 13;
const RUN = 0.85;
const STAIR_BOTTOM = 11.5;
const STAIR_TOP = STAIR_BOTTOM - 12 * RUN;

export function stairHeight(z) {
  const i = Math.floor((STAIR_BOTTOM - z) / RUN);
  if (i < 0) return 0;
  if (i > 11) return 8.5;
  return (i + 1) * RISE;
}

function paint(color) {
  return mat(color, { roughness: 0.92 });
}

function texMat(tex, color = '#ffffff', extra = {}) {
  return new THREE.MeshStandardMaterial({ map: tex, color, roughness: 0.9, ...extra });
}

export function buildHouse(b) {
  const sidingTex = T.cedarShake();
  const M = {
    siding: texMat(sidingTex),
    roof: texMat(T.roofShingles()),
    brick: texMat(T.brick()),
    oak: texMat(T.oakFloor(), '#ffffff', { roughness: 0.55 }),
    trim: mat('#f7f6f1', { roughness: 0.6 }),
    ceiling: mat('#f7f6f2', { roughness: 0.95 }),
    concrete: texMat(T.concrete()),
    glass: new THREE.MeshPhysicalMaterial({ color: '#bcd6e6', transparent: true, opacity: 0.22, roughness: 0.05, metalness: 0, depthWrite: false }),
    screen: new THREE.MeshStandardMaterial({ color: '#2f3438', transparent: true, opacity: 0.18, depthWrite: false, side: THREE.DoubleSide }),
    shutter: mat('#22324f', { roughness: 0.6 }),
    kitchenTile: texMat(T.tile('#ece6da', '#d5cdbd', 4)),
    bathTile: texMat(T.tile('#f2f2ef', '#c9cbc8', 8)),
    beigeCarpet: texMat(T.carpet('#cfc6b6')),
    blueCarpet: texMat(T.carpet('#34507a')),
    porchFloor: mat('#4d6a86', { roughness: 0.7 }),
    maple: mat('#dcc7a6', { roughness: 0.6 }),
    counter: mat('#f4f2ec', { roughness: 0.35 }),
    clapboard: texMat(T.clapboard()),
  };

  const P = {
    cream: paint('#efe7d4'),
    dining: paint('#a7c4e0'),
    kitchen: paint('#94b7d9'),
    br1: paint('#dcd6c8'),
    br2: paint('#d9d2c4'),
    br3: paint('#d8dcdc'),
    bath: paint('#cdd3d2'),
    upW: paint('#a9c3dd'),
    upE: paint('#e9e1d0'),
    bath2: paint('#d5dad8'),
    landing: paint('#efe7d4'),
    porch: paint('#f5f5f2'),
  };

  const ext = { y0: 0, y1: 8.5, t: 0.6, trim: M.trim, glass: M.glass, edge: M.trim };
  const int = { y0: 0, y1: 8, t: 0.4, trim: M.trim, glass: M.glass };
  const win = (a, b2, bottom = 2.6, top = 7) => ({ a, b: b2, bottom, top, window: true });
  const door = (a, b2, top = 6.9) => ({ a, b: b2, bottom: 0, top });
  const up = (o) => ({ ...o, bottom: (o.bottom ?? 0) + 8.5, top: (o.top ?? 6.9) + 8.5 });

  // ---------------------------------------------------------------- foundation
  b.box(-19.3, 6, -2.5, 0, -15.3, 15.3, M.concrete, { collide: true });
  b.box(6, 19.3, -2.5, 0, -3, 15.3, M.concrete, { collide: true });
  b.box(6, 19.3, -2.5, -1, -15.3, -3, M.concrete, { collide: true });

  // ------------------------------------------------------------ exterior walls
  b.wall({ ...ext, axis: 'x', at: 15, from: -19.3, to: -6, matPos: M.siding, matNeg: P.br1, openings: [win(-15.4, -12.6), win(-9.9, -7.1)] });
  b.wall({ ...ext, axis: 'x', at: 15, from: -6, to: 6, matPos: M.siding, matNeg: P.cream, openings: [door(-1.5, 1.5)] });
  b.wall({ ...ext, axis: 'x', at: 15, from: 6, to: 19.3, matPos: M.siding, matNeg: P.cream, openings: [{ a: 9, b: 16, bottom: 1.6, top: 7 }] });

  b.wall({ ...ext, axis: 'x', at: -15, from: -19.3, to: -6, matPos: P.br2, matNeg: M.siding, openings: [win(-14.5, -11.5)] });
  b.wall({ ...ext, axis: 'x', at: -15, from: -6, to: 6, matPos: P.kitchen, matNeg: M.siding, openings: [win(-1.8, 1.8, 3.6, 7)] });

  b.wall({ ...ext, axis: 'z', at: -19, from: -14.7, to: -6, matPos: P.br2, matNeg: M.siding, openings: [win(-11.8, -9.2)] });
  b.wall({ ...ext, axis: 'z', at: -19, from: -6, to: 1, matPos: P.br3, matNeg: M.siding, openings: [win(-3.8, -1.2)] });
  b.wall({ ...ext, axis: 'z', at: -19, from: 1, to: 6, matPos: P.bath, matNeg: M.siding, openings: [win(2.8, 4.4, 4.4, 6.6)] });
  b.wall({ ...ext, axis: 'z', at: -19, from: 6, to: 14.7, matPos: P.br1, matNeg: M.siding, openings: [win(9.2, 11.8)] });

  b.wall({ ...ext, axis: 'z', at: 19, from: -3, to: 14.7, matPos: M.siding, matNeg: P.cream, openings: [win(0.4, 3.0), win(8.0, 10.6)] });

  // Navy shutters + window boxes on the front, like the listing photos.
  const shutter = (x0, x1, y0, y1) => b.box(x0, x1, y0, y1, 15.3, 15.42, M.shutter, { occlude: false });
  for (const [a, c] of [[-15.4, -12.6], [-9.9, -7.1]]) {
    shutter(a - 1.55, a - 0.35, 2.4, 7.2);
    shutter(c + 0.35, c + 1.55, 2.4, 7.2);
    b.box(a - 0.1, c + 0.1, 1.75, 2.35, 15.35, 16.2, mat('#3d4248'), { occlude: false });
    b.box(a + 0.05, c - 0.05, 2.35, 2.85, 15.45, 16.1, mat('#4f8a36', { flatShading: true }), { occlude: false });
  }
  shutter(7.35, 8.55, 1.6, 7.2);
  shutter(16.45, 17.65, 1.6, 7.2);

  // Bay window in the living room.
  b.box(9, 16, 0.9, 1.6, 15.3, 16.8, M.trim, { collide: true });
  b.box(8.8, 16.2, 7, 7.5, 15.3, 17, M.trim);
  b.box(8.7, 16.3, 7.5, 7.75, 15.3, 17.1, M.roof);
  b.windowGlass('x', 16.65, 9.2, 15.8, 1.6, 7, M.glass, M.trim);
  b.windowGlass('z', 9.1, 15.3, 16.7, 1.6, 7, M.glass, M.trim);
  b.windowGlass('z', 15.9, 15.3, 16.7, 1.6, 7, M.glass, M.trim);
  for (const x of [9, 16]) b.box(x - 0.15, x + 0.15, 1.6, 7, 16.55, 16.85, M.trim, { occlude: false });
  b.collider(9, 16, 0, 8, 15.3, 16.9);

  // ------------------------------------------------------------ interior walls
  b.wall({ ...int, axis: 'z', at: -6, from: 6, to: 14.7, matPos: P.cream, matNeg: P.br1, openings: [door(7.5, 10.2)] });
  b.wall({ ...int, axis: 'x', at: 6, from: -18.7, to: -9, matPos: P.br1, matNeg: P.bath });
  b.wall({ ...int, axis: 'x', at: 6, from: -9, to: -6, matPos: P.br1, matNeg: P.cream });
  b.wall({ ...int, axis: 'z', at: -9, from: 1, to: 6, matPos: P.cream, matNeg: P.bath, openings: [door(2, 4.6)] });
  b.wall({ ...int, axis: 'x', at: 1, from: -18.7, to: -9, matPos: P.bath, matNeg: P.br3 });
  b.wall({ ...int, axis: 'z', at: -9, from: -6, to: 1, matPos: P.cream, matNeg: P.br3, openings: [door(-2.6, 0)] });
  b.wall({ ...int, axis: 'x', at: -6, from: -18.7, to: -9, matPos: P.br3, matNeg: P.br2 });
  b.wall({ ...int, axis: 'x', at: -6, from: -9, to: -6, matPos: P.cream, matNeg: P.br2, openings: [door(-8.6, -6.4)] });
  b.wall({ ...int, axis: 'z', at: -6, from: -6, to: 6, matPos: P.dining, matNeg: P.cream, openings: [{ a: -0.5, b: 3, bottom: 0, top: 7 }] });
  b.wall({ ...int, axis: 'z', at: -6, from: -14.7, to: -6, matPos: P.kitchen, matNeg: P.br2 });
  b.wall({ ...int, axis: 'x', at: -5, from: -6, to: 6, matPos: P.dining, matNeg: P.kitchen, openings: [{ a: -2, b: 2, bottom: 0, top: 7 }] });
  b.wall({ ...int, axis: 'z', at: 6, from: -14.7, to: -5, matPos: M.siding, matNeg: P.kitchen, openings: [door(-12, -9.4)] });
  b.wall({ ...int, axis: 'z', at: 6, from: -5, to: -3, matPos: M.siding, matNeg: P.dining });
  b.wall({ ...int, axis: 'z', at: 6, from: -3, to: 5, matPos: P.cream, matNeg: P.dining, openings: [{ a: -2.5, b: 1.1, bottom: 0, top: 7 }] });
  b.wall({ ...int, axis: 'z', at: 6, from: 5, to: 14.7, matPos: P.cream, matNeg: P.cream, openings: [{ a: 12, b: 14.5, bottom: 0, top: 7 }] });
  b.wall({ ...int, t: 0.5, axis: 'x', at: -3, from: 6, to: 18.7, matPos: P.cream, matNeg: M.siding, openings: [win(8.6, 11.4, 3, 7), win(13.4, 16.4, 3, 7)] });
  b.wall({ ...int, axis: 'x', at: 5, from: -6, to: 3, matPos: P.cream, matNeg: P.dining, openings: [{ a: -3.6, b: -0.4, bottom: 0, top: 7 }] });

  // Front door, swung open into the foyer.
  b.box(-1.5, -1.35, 0, 6.8, 11.8, 14.7, M.trim, { collide: true });
  b.box(-1.62, -1.5, 3.1, 3.4, 12.0, 12.2, mat('#c9a44f', { metalness: 0.8, roughness: 0.3 }), { occlude: false });

  // --------------------------------------------------------------- staircase
  const prismShape = new THREE.Shape([
    new THREE.Vector2(-STAIR_BOTTOM, 0),
    new THREE.Vector2(-STAIR_TOP, 0),
    new THREE.Vector2(-STAIR_TOP, 8.0),
  ]);
  const prism = new THREE.Mesh(new THREE.ExtrudeGeometry(prismShape, { depth: 3, bevelEnabled: false }), M.trim);
  prism.rotation.y = Math.PI / 2;
  prism.position.set(3, 0, 0);
  b.add(prism);
  const treadMats = [M.trim, M.trim, M.oak, M.trim, M.trim, M.trim];
  for (let i = 0; i < 12; i++) {
    const zf = STAIR_BOTTOM - i * RUN;
    b.box(3.05, 5.95, i * RISE, (i + 1) * RISE, zf - RUN, zf + (i === 0 ? 0 : 0.08), treadMats, { occlude: false });
  }
  b.surface(3, 6, STAIR_TOP, STAIR_BOTTOM, (x, z) => stairHeight(z));
  b.collider(2.85, 3.15, 0, 17, STAIR_TOP, STAIR_BOTTOM - 0.3);
  b.collider(3, 6, 0, 7, STAIR_TOP - 0.25, STAIR_TOP + 0.1);
  // Turned balusters and handrail on the open (foyer) side.
  const railWood = mat('#a8692f', { roughness: 0.5 });
  for (let i = 0; i < 8; i++) {
    const zc = STAIR_BOTTOM - (i + 0.5) * RUN;
    const yb = (i + 1) * RISE;
    b.box(3.0, 3.14, yb, yb + 3.0 + 0.0, zc - 0.07, zc + 0.07, railWood, { occlude: false });
  }
  b.box(2.9, 3.3, 0, 4.0, STAIR_BOTTOM - 0.45, STAIR_BOTTOM - 0.05, railWood, { occlude: false });
  b.slopeZ(STAIR_BOTTOM - 0.25, 3.6, STAIR_BOTTOM - 8 * RUN, 8 * RISE + 3.2, 2.95, 3.2, 0.18, railWood, { occlude: false });
  // Basement door under the stairs, seen from the dining room.
  b.box(3.4, 5.8, 0, 6.8, STAIR_TOP - 0.12, STAIR_TOP - 0.02, M.trim, { occlude: false });
  b.box(5.3, 5.5, 3.1, 3.3, STAIR_TOP - 0.22, STAIR_TOP - 0.1, mat('#c9a44f', { metalness: 0.8, roughness: 0.3 }), { occlude: false });
  // Louvered coat closet across from the stairs.
  b.box(-5.8, -5.75, 0, 6.8, 11.2, 13.8, mat('#f2f1ec'), { occlude: false });
  for (let y = 0.6; y < 6.6; y += 0.3) b.box(-5.75, -5.7, y, y + 0.12, 11.3, 13.7, M.trim, { occlude: false });

  // ------------------------------------------------------- floors & ceilings
  b.box(-18.7, 6, -0.3, 0.01, -14.7, 14.7, M.oak, { occlude: false, cast: false });
  b.box(6, 18.7, -0.3, 0.01, -2.75, 14.7, M.oak, { occlude: false, cast: false });
  b.box(-5.8, 5.8, 0.01, 0.025, -14.7, -5.2, M.kitchenTile, { occlude: false, cast: false });
  b.box(-18.7, -9.2, 0.01, 0.025, 1.2, 5.8, M.bathTile, { occlude: false, cast: false });
  b.box(-18.7, -6.2, 0.01, 0.025, -14.7, -6.2, M.beigeCarpet, { occlude: false, cast: false });
  b.box(6.3, 18.7, -1.02, -0.99, -14.7, -3.25, M.blueCarpet, { occlude: false, cast: false });
  b.surface(-19, 6, -15, 15, 0);
  b.surface(6, 19, -3, 15, 0);
  b.surface(6, 19, -15, -3, -1);

  // Upstairs floor slab (stairwell left open).
  const slab = [[-19.3, 3, -9, 15.3], [6, 19.3, -9, 15.3], [3, 6, -9, STAIR_TOP], [3, 6, STAIR_BOTTOM, 15.3]];
  for (const [x0, x1, z0, z1] of slab) {
    b.box(x0, x1, 8, 8.5, z0, z1, M.ceiling);
    b.box(Math.max(x0, -18.7), Math.min(x1, 18.7), 8.5, 8.52, z0, Math.min(z1, 11), M.oak, { occlude: false, cast: false });
  }
  b.box(-19.3, 19.3, 8, 8.4, -15.3, -9, M.ceiling);
  b.box(5.8, 6.2, 8, 8.5, STAIR_TOP, STAIR_BOTTOM, M.trim);
  b.surface(-19, 3, -9, 11, 8.5);
  b.surface(6, 19, -9, 11, 8.5);
  b.surface(3, 6, -9, STAIR_TOP, 8.5);
  // Kitchen skylight.
  b.box(-1.5, 1.5, 7.9, 7.98, -12.2, -10.2, mat('#ffffff', { emissive: '#e8f4ff', emissiveIntensity: 1.4 }), { occlude: false, cast: false });

  // --------------------------------------------------------------- upstairs
  const dorm = { y0: 8.5, y1: 17.6, t: 0.6, trim: M.trim, glass: M.glass, edge: M.trim };
  b.wall({ ...dorm, axis: 'x', at: -9, from: -19.3, to: -1, matPos: P.upW, matNeg: M.siding, openings: [up(win(-12.5, -9.5, 2.5, 6))] });
  b.wall({ ...dorm, axis: 'x', at: -9, from: -1, to: 6, matPos: P.bath2, matNeg: M.siding, openings: [up(win(1.2, 3.6, 3.5, 6))] });
  b.wall({ ...dorm, axis: 'x', at: -9, from: 6, to: 19.3, matPos: P.upE, matNeg: M.siding, openings: [up(win(11, 14, 2.5, 6))] });
  for (const [a, c] of [[-12.5, -9.5], [1.2, 3.6], [11, 14]]) {
    b.slopeZ(-9.3, 15.3, -10.8, 14.3, a - 0.3, c + 0.3, 0.08, M.trim, { occlude: false });
  }

  const upI = { y0: 8.5, y1: 16, t: 0.4, trim: M.trim, glass: M.glass };
  b.wall({ ...upI, y1: 12.5, axis: 'x', at: 11, from: -18.7, to: -1, matPos: P.upW, matNeg: P.upW });
  b.wall({ ...upI, y1: 12.5, axis: 'x', at: 11, from: -1, to: 3, matPos: P.landing, matNeg: P.landing });
  b.wall({ ...upI, y1: 12.5, axis: 'x', at: 11, from: 6, to: 18.7, matPos: P.upE, matNeg: P.upE });
  b.wall({ ...upI, axis: 'z', at: -1, from: -8.7, to: -3, matPos: P.bath2, matNeg: P.upW });
  b.wall({ ...upI, axis: 'z', at: -1, from: -3, to: 6.8, matPos: P.landing, matNeg: P.upW, openings: [up(door(-2.2, 0.4))] });
  b.wall({ ...upI, axis: 'z', at: 6, from: -8.7, to: -3, matPos: P.upE, matNeg: P.bath2 });
  b.wall({ ...upI, axis: 'z', at: 6, from: -3, to: 6.8, matPos: P.upE, matNeg: P.landing, openings: [up(door(-2.2, 0.4))] });
  b.wall({ ...upI, axis: 'x', at: -3, from: -1, to: 6, matPos: P.landing, matNeg: P.bath2, openings: [up(door(0, 2.5))] });
  b.wall({ ...upI, axis: 'z', at: 3, from: STAIR_TOP, to: 6.8, matPos: P.landing, matNeg: P.landing });
  // Partition ends that follow the sloped ceiling between z 6.8 and the knee wall.
  const underSlope = new THREE.Shape([[6.8, 8.5], [11, 8.5], [11, 12.5], [6.8, 16]].map(([z, y]) => new THREE.Vector2(-z, y)));
  for (const [x, m] of [[-1, P.upW], [3, P.landing], [6, P.upE]]) {
    const part = new THREE.Mesh(new THREE.ExtrudeGeometry(underSlope, { depth: 0.4, bevelEnabled: false }), m);
    part.rotation.y = Math.PI / 2;
    part.position.x = x - 0.2;
    b.add(part);
    b.collider(x - 0.2, x + 0.2, 8.5, 16, 6.8, 11);
  }
  b.wall({ ...upI, axis: 'x', at: STAIR_TOP, from: -1, to: 3, matPos: P.landing, matNeg: P.landing });
  b.box(-0.4, 2.4, 8.5, 15.3, STAIR_TOP - 0.3, STAIR_TOP - 0.2, M.trim, { occlude: false });

  // Sloped ceiling under the front roof, then flat ceiling.
  b.slopeZ(11, 12.5, 6.8, 16, -18.7, 18.7, 0.15, M.ceiling);
  b.box(-18.7, 18.7, 16, 16.3, -8.7, 6.8, M.ceiling);

  // Gable ends (polygon follows the roof profile) with upstairs windows.
  const profile = [[15.3, 8.5], [15.3, 9.0], [3, 19.25], [-9.3, 17.6], [-9.3, 11.5], [-15.3, 9.0], [-15.3, 8.5]];
  const gable = (side, windows, liningMat) => {
    const shape = new THREE.Shape(profile.map(([z, y]) => new THREE.Vector2(-z, y)));
    for (const [za, zb, ya, yb] of windows) {
      shape.holes.push(new THREE.Path([new THREE.Vector2(-za, ya), new THREE.Vector2(-zb, ya), new THREE.Vector2(-zb, yb), new THREE.Vector2(-za, yb)]));
    }
    const geo = new THREE.ExtrudeGeometry(shape, { depth: 0.6, bevelEnabled: false });
    const mesh = new THREE.Mesh(geo, [M.siding, M.trim]);
    mesh.rotation.y = Math.PI / 2;
    mesh.position.x = side > 0 ? 18.7 : -19.3;
    b.add(mesh);
    const liningShape = new THREE.Shape([[11, 8.5], [11, 12.5], [6.8, 16], [-8.7, 16], [-8.7, 8.5]].map(([z, y]) => new THREE.Vector2(-z, y)));
    liningShape.holes = shape.holes;
    const lining = new THREE.Mesh(new THREE.ShapeGeometry(liningShape), liningMat.clone());
    lining.material.side = THREE.DoubleSide;
    lining.rotation.y = Math.PI / 2;
    lining.position.x = side > 0 ? 18.68 : -18.68;
    b.add(lining);
    for (const [za, zb, ya, yb] of windows) {
      b.openingTrim('z', side * 19, Math.min(za, zb), Math.max(za, zb), ya, yb, 0.6, M.trim, true);
      b.windowGlass('z', side * 19, Math.min(za, zb), Math.max(za, zb), ya, yb, M.glass, M.trim);
    }
    b.box(side * 19 - 0.35, side * 19 + 0.35, 16.6, 18.2, 2.3, 3.7, M.trim, { occlude: false });
    b.collider(side * 19 - 0.3, side * 19 + 0.3, 8.5, 17, -9, 11);
  };
  gable(-1, [[0, 3, 10.5, 14.5]], P.upW);
  gable(1, [[-4, -1.5, 10.5, 14.5]], P.upE);

  // ------------------------------------------------------------------- roofs
  b.slopeZ(16.3, 8.17, 3, 19.25, -20, 20, 0.35, M.roof);
  b.slopeZ(3, 19.25, -9.9, 17.5, -20, 20, 0.35, M.roof);
  b.slopeZ(-9.3, 11.5, -16.3, 8.58, -20, 20, 0.35, M.roof);
  b.box(-20.1, 20.1, 19.3, 19.7, 2.7, 3.5, M.roof);
  b.box(-20, 20, 7.75, 8.4, 16.2, 16.45, M.trim, { occlude: false });
  b.box(-20, 20, 8.15, 8.75, -16.45, -16.2, M.trim, { occlude: false });
  b.box(-20, 20, 16.9, 17.6, -10.1, -9.85, M.trim, { occlude: false });
  for (const s of [-1, 1]) {
    const x0 = s > 0 ? 19.3 : -19.65, x1 = s > 0 ? 19.65 : -19.3;
    b.slopeZ(16.3, 7.9, 3, 18.95, x0, x1, 0.55, M.trim, { occlude: false });
    b.slopeZ(3, 18.95, -9.9, 17.2, x0, x1, 0.55, M.trim, { occlude: false });
    b.slopeZ(-9.3, 11.2, -16.3, 8.3, x0, x1, 0.55, M.trim, { occlude: false });
  }
  const skyY = 9.0 + (-12 + 15.3) * (2.5 / 6);
  b.slopeZ(-11, skyY + 0.85, -13, skyY + 0.02, -1.4, 1.4, 0.3, mat('#9fbdd2', { roughness: 0.1, metalness: 0.3 }), { occlude: false });

  // Chimney on the right gable end.
  b.box(19.3, 22.3, -2.5, 7.5, 3.6, 7.4, M.brick, { collide: true });
  b.box(19.3, 21.6, 7.5, 22, 4.3, 6.7, M.brick, { collide: true });
  b.box(19.1, 21.8, 22, 22.4, 4.1, 6.9, M.concrete);

  // ----------------------------------------------------------- front porch
  const porchTop = [M.brick, M.brick, M.porchFloor, M.brick, M.brick, M.brick];
  b.box(-4.5, 4.5, -2.5, 0, 15.3, 20.8, porchTop, { collide: false });
  b.surface(-4.5, 4.5, 15, 20.8, 0);
  b.collider(-4.6, -4.4, -2.5, 0.2, 15.3, 20.8);
  b.collider(4.4, 4.6, -2.5, 0.2, 15.3, 20.8);
  b.collider(-4.6, -3.6, -2.5, 0.2, 20.6, 20.9);
  b.collider(3.6, 4.6, -2.5, 0.2, 20.6, 20.9);
  const stepMats = [M.brick, M.brick, mat('#8b6f5a', { roughness: 0.8 }), M.brick, M.brick, M.brick];
  for (let i = 0; i < 4; i++) {
    const top = -0.5 * (i + 1);
    const z0 = 20.8 + i, z1 = 20.8 + i + 1;
    b.box(-3.6, 3.6, -2.5, top, z0, z1, stepMats, { occlude: false });
    b.surface(-3.6, 3.6, z0, z1, top);
  }
  for (const s of [-1, 1]) {
    b.railing(s * 4.3, 15.6, s * 4.3, 20.4, 0, 2.8, M.trim, { spacing: 0.4 });
    b.box(s * 4.3 - 0.25, s * 4.3 + 0.25, 0, 7.9, 20.15, 20.65, M.trim, { collide: true });
    b.box(s * 4.3 - 0.2, s * 4.3 + 0.2, 0, 7.9, 15.3, 15.6, M.trim, { occlude: false });
    const rx = s * 3.8;
    b.box(rx - 0.2, rx + 0.2, -2.5, 0.8, 24.4, 24.8, M.trim, { collide: true });
    b.slopeZ(20.6, 2.8, 24.6, 0.8, rx - 0.11, rx + 0.11, 0.14, M.trim, { occlude: false });
    for (let i = 0; i < 4; i++) {
      const z = 21.3 + i;
      const top = 2.8 - ((z - 20.6) / 4) * 2.0;
      b.box(rx - 0.05, rx + 0.05, -0.5 * (i + 1), top, z - 0.05, z + 0.05, M.trim, { occlude: false });
    }
    b.collider(rx - 0.15, rx + 0.15, -2.5, 3, 20.8, 24.8);
  }
  b.box(-4.7, 4.7, 7.6, 8.4, 20.2, 20.8, M.trim);
  b.box(-4.6, 4.6, 7.9, 8.2, 15.3, 20.8, M.ceiling);
  b.slopeX(-5.3, 8.3, 0, 11.7, 15.3, 21.3, 0.3, M.roof);
  b.slopeX(0, 11.7, 5.3, 8.3, 15.3, 21.3, 0.3, M.roof);
  const ped = new THREE.Mesh(
    new THREE.ShapeGeometry(new THREE.Shape([new THREE.Vector2(-4.9, 8.4), new THREE.Vector2(4.9, 8.4), new THREE.Vector2(0, 11.5)])),
    new THREE.MeshStandardMaterial({ map: M.clapboard.map, side: THREE.DoubleSide, roughness: 0.8 })
  );
  ped.position.z = 20.85;
  b.add(ped);
  const awning = new THREE.MeshStandardMaterial({ map: T.stripes('#1f3557', '#f2f2ec', 16), roughness: 0.9, side: THREE.DoubleSide });
  awning.map.repeat.set(1 / 8.4, 1);
  b.slopeZ(20.7, 8.0, 21.9, 7.0, -4.2, 4.2, 0.05, awning, { occlude: false, cast: true });
  b.box(-4.2, 4.2, 6.5, 7.02, 21.86, 21.92, awning, { occlude: false });
  for (const x of [-2.15, 2.15]) {
    b.box(x - 0.2, x + 0.2, 5.0, 5.8, 15.3, 15.6, mat('#1b1b1b', { emissive: '#ffcf80', emissiveIntensity: 0.5 }), { occlude: false });
  }
  const sign = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 0.5), new THREE.MeshStandardMaterial({ map: T.signTexture('419'), roughness: 0.6 }));
  sign.position.set(0, -0.25, 20.81);
  b.add(sign, { occlude: false });
  // Black metal bench on the porch.
  const bench = new THREE.Group();
  const iron = mat('#1c1c1c', { roughness: 0.5, metalness: 0.6 });
  const seat = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.1, 3.4), iron);
  seat.position.y = 1.5;
  bench.add(seat);
  const back = new THREE.Mesh(new THREE.BoxGeometry(0.1, 1.4, 3.4), iron);
  back.position.set(-0.75, 2.2, 0);
  bench.add(back);
  for (const sz of [-1.6, 1.6]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(1.4, 1.5, 0.1), iron);
    leg.position.set(0, 0.75, sz);
    bench.add(leg);
  }
  bench.position.set(-3.2, 0, 18);
  b.add(bench, { occlude: false });
  b.colliderFromObject(bench);

  // --------------------------------------------------------- screened porch
  b.wall({ axis: 'z', at: 19, from: -14.7, to: -3, y0: -1, y1: 1.4, t: 0.6, matPos: M.siding, matNeg: P.porch, edge: M.trim });
  b.wall({ axis: 'x', at: -15, from: 6, to: 19.3, y0: -1, y1: 1.4, t: 0.6, matPos: P.porch, matNeg: M.siding, edge: M.trim, openings: [{ a: 7, b: 9.8, bottom: -1, top: 6.4 }] });
  b.box(18.7, 19.3, 7.3, 8.5, -14.7, -3, M.trim);
  b.box(6, 19.3, 7.3, 8.5, -15.3, -14.7, M.trim);
  b.box(18.97, 19.03, 1.4, 7.3, -14.7, -3, M.screen, { cast: false });
  b.box(9.8, 18.7, 1.4, 7.3, -15.03, -14.97, M.screen, { cast: false });
  b.box(6, 7, 1.4, 7.3, -15.03, -14.97, M.screen, { cast: false });
  b.box(7, 9.8, 6.4, 7.3, -15.03, -14.97, M.screen, { cast: false });
  b.collider(18.7, 19.3, -1, 8.5, -15.3, -3);
  b.collider(9.8, 19.3, -1, 8.5, -15.3, -14.7);
  b.collider(6, 7, -1, 8.5, -15.3, -14.7);
  for (let z = -14.7; z <= -3; z += 2.9) b.box(18.75, 19.25, 1.4, 7.3, z - 0.2, z + 0.2, M.trim, { occlude: false });
  for (const x of [6.5, 7, 9.8, 12.8, 15.8, 18.9]) b.box(x - 0.2, x + 0.2, 1.4, 7.3, -15.25, -14.75, M.trim, { occlude: false });
  b.box(6.6, 9.8, 6.4, 6.8, -15.25, -14.75, M.trim, { occlude: false });
  b.box(7, 9.8, -2.5, -1.75, -16.5, -15.3, M.concrete, { occlude: false });
  b.surface(7, 9.8, -16.5, -15.3, -1.75);

  // -------------------------------------------------------- living room
  const hearth = mat('#9a4b3a', { roughness: 0.7 });
  b.box(16.5, 18.7, 0, 0.07, 3.7, 7.3, hearth, { occlude: false });
  b.box(18.45, 18.7, 0, 3.3, 4.2, 6.8, M.brick, { occlude: false });
  b.box(18.38, 18.46, 0.15, 2.85, 4.55, 6.45, mat('#b8933f', { metalness: 0.8, roughness: 0.35 }), { occlude: false });
  b.box(18.35, 18.4, 0.3, 2.7, 4.7, 6.3, mat('#0d0d0d', { roughness: 0.15 }), { occlude: false });
  b.box(18.2, 18.7, 0, 3.7, 3.4, 4.2, M.trim, { collide: true });
  b.box(18.2, 18.7, 0, 3.7, 6.8, 7.6, M.trim, { collide: true });
  b.box(18.2, 18.7, 3.3, 3.85, 3.4, 7.6, M.trim);
  b.box(17.85, 18.7, 3.85, 4.05, 3.15, 7.85, M.trim);
  F.tv(b, 18.6, 4.6, 5.5, -Math.PI / 2, { w: 4.2, h: 2.4 });

  F.sofa(b, 7.9, 7.6, Math.PI / 2, { w: 7.5, color: '#26416b' });
  F.table(b, 7.4, 2.6, 0, { w: 1.8, d: 1.8, h: 2.2, top: '#7a4f2e' });
  F.lamp(b, 7.4, 2.2, 2.6, { base: '#6c9ab6' });
  F.recliner(b, 9.8, 0.6, Math.PI / 2 + 0.5, { color: '#8e9398' });
  const stripeFab = T.stripes('#2b3f63', '#efe9db', 14);
  stripeFab.repeat.set(0.6, 0.6);
  F.armchair(b, 10.3, 13.2, Math.PI, { stripe: stripeFab });
  F.armchair(b, 15.2, 13.2, Math.PI, { stripe: stripeFab });
  F.roundTable(b, 12.75, 13.6, { r: 0.75, h: 2.2 });
  F.floorLamp(b, 17.4, 12.3);
  const wicker = mat('#a68457', { roughness: 1 });
  b.box(17.2, 18.4, 0, 1.4, 1.6, 2.9, wicker, { collide: true, occlude: false });
  b.box(17.2, 18.4, 0, 1.4, 8.1, 9.4, wicker, { collide: true, occlude: false });
  F.roundTable(b, 11, -0.3, { r: 1.5, h: 3.3, top: '#2d3540' });
  F.stool(b, 9.8, -1.4);
  F.stool(b, 12.3, -1.2);
  F.stool(b, 11.2, 1.1);
  F.hutch(b, 17.9, -1.4, -Math.PI / 2);
  F.picture(b, 'z', 6.2, 7.6, 6.6, 3.4, 2.3, 1, T.painting(3));
  b.box(15.6, 16.6, 4.6, 6.0, 14.65, 14.7, mat('#9a6a3d'), { occlude: false });

  // ------------------------------------------------------------ dining room
  F.diningTable(b, -0.6, -1.4, 0);
  F.ceilingLight(b, -0.6, 8, -1.4, { chandelier: true });
  F.picture(b, 'x', -4.8, -4.0, 6.0, 2.4, 1.8, 1, T.painting(8));
  F.picture(b, 'x', 5, 1.6, 6.2, 1.8, 1.4, -1, T.painting(5));

  // --------------------------------------------------------------- kitchen
  const counterRun = (x0, x1, z0, z1) => {
    b.box(x0, x1, 0, 2.9, z0, z1, M.maple, { collide: true, occlude: false });
    b.box(x0 - 0.05, x1 + 0.05, 2.9, 3.05, z0 - 0.05, z1 + 0.05, M.counter, { occlude: false });
  };
  counterRun(-5.8, 4.4, -14.7, -12.7);
  b.box(-0.8, 0.8, 3.03, 3.07, -14.3, -13.1, mat('#ffffff', { roughness: 0.2 }), { occlude: false });
  b.box(-0.08, 0.08, 3.05, 4.0, -14.5, -14.35, mat('#c9ccd0', { metalness: 0.9, roughness: 0.2 }), { occlude: false });
  b.box(1.0, 3.4, 0.15, 2.85, -12.72, -12.62, mat('#fafaf8', { roughness: 0.3 }), { occlude: false });
  for (let x = -5.6; x < 0.9; x += 1.6) b.box(x, x + 1.4, 0.4, 2.6, -12.72, -12.66, mat('#e4d2b4'), { occlude: false });
  b.box(-5.8, -2.3, 4.7, 7.6, -14.7, -13.5, M.maple, { occlude: false });
  b.box(2.3, 4.6, 4.7, 7.6, -14.7, -13.5, M.maple, { occlude: false });
  counterRun(-5.8, -3.8, -12.7, -10.6);
  counterRun(-5.8, -3.8, -8.0, -6.3);
  F.range(b, -4.75, -9.3, Math.PI / 2);
  b.box(-5.8, -4.6, 4.7, 7.6, -12.7, -10.6, M.maple, { occlude: false });
  b.box(-5.8, -4.6, 4.7, 7.6, -8.0, -6.3, M.maple, { occlude: false });
  b.box(-5.8, -4.0, 5.6, 6.6, -10.6, -8.0, mat('#f2f2f0'), { occlude: false });
  F.fridge(b, 4.55, -7.0, -Math.PI / 2);
  b.box(4.4, 5.2, 3.05, 4.0, -14.3, -13.8, mat('#1c1c1c'), { occlude: false });
  F.ceilingLight(b, 0, 8, -8.2);

  // ---------------------------------------------------------- screened porch
  F.wickerSofa(b, 17.0, -8.8, -Math.PI / 2, { w: 6.8, y: -1 });
  F.wickerSofa(b, 12.6, -4.9, Math.PI, { w: 5.2, y: -1, cushion: '#2a4a7c' });
  const navyStripe = T.stripes('#1f3557', '#f4f4f0', 10);
  F.wickerChair(b, 11.6, -12.9, 0.2, { y: -1, stripe: navyStripe });
  F.wickerChair(b, 15.2, -13.0, -0.2, { y: -1 });
  F.wickerChair(b, 8.4, -8.5, Math.PI / 2, { y: -1 });
  F.table(b, 13.2, -8.8, 0, { w: 3.2, d: 2, h: 1.6, y: -1, top: '#e3d6bb' });

  // -------------------------------------------- front bedroom (used as TV room)
  F.sofa(b, -12.5, 13.1, Math.PI, { w: 7, color: '#6f7d8c' });
  F.dresser(b, -12.5, 7.1, 0, { w: 4.8, h: 2.2, d: 1.5, color: '#3c3a38' });
  F.tv(b, -12.5, 2.45, 7.1, 0, { w: 4.4, h: 2.5 });
  F.armchair(b, -17.0, 9.5, Math.PI / 2, { color: '#c8b99a' });
  F.rug(b, -16, -9, 8.6, 11.8, 0.012, mat('#b7a58a', { roughness: 1 }));
  F.ceilingLight(b, -12.5, 8, 10.5);

  // ------------------------------------------------------------- hall bath
  F.tub(b, -18.6, -16.1, 1.3, 5.8);
  b.box(-16.08, -16.02, 1.8, 7.2, 1.3, 5.8, mat('#ffffff', { transparent: true, opacity: 0.85 }), { occlude: false });
  F.toilet(b, -14.4, 5.0, Math.PI);
  F.vanity(b, -12.0, 2.15, 0, { w: 2.8 });

  // --------------------------------------------------- downstairs bedrooms
  F.bed(b, -15.3, -2.6, Math.PI / 2, { w: 3.4, l: 6.6, pillows: 1, quilt: T.quilt(), frame: '#f0ede4' });
  F.dresser(b, -11.2, -5.0, 0, { w: 3.6, h: 3.4, color: '#e9e4d8' });
  F.ceilingLight(b, -14, 8, -2.5);
  F.picture(b, 'x', 0.8, -15.5, 5.6, 2.0, 1.4, -1, T.painting(11));

  F.bed(b, -13.0, -9.8, Math.PI, { w: 5, l: 6.8, quilt: T.quilt('#efe6d6') });
  F.dresser(b, -16.4, -6.95, 0, { w: 1.6, h: 2.1, d: 1.4 });
  F.dresser(b, -9.6, -6.95, 0, { w: 1.6, h: 2.1, d: 1.4 });
  F.lamp(b, -16.4, 2.3, -6.95, { base: '#8a6a4a' });
  F.lamp(b, -9.6, 2.3, -6.95, { base: '#8a6a4a' });
  F.dresser(b, -7.0, -11.8, -Math.PI / 2, { w: 4.2, h: 4, d: 1.6, color: '#6b4428' });
  F.table(b, -10.2, -13.9, 0, { w: 3.6, d: 1.6, h: 2.5, top: '#7a4f2e' });
  F.ceilingLight(b, -12.5, 8, -10.5);
  F.picture(b, 'x', -6, -13.0, 6.3, 1.4, 1.1, -1, T.painting(17));

  F.ceilingLight(b, -7.5, 8, 0);
  F.ceilingLight(b, 0, 8, 10);
  F.ceilingLight(b, 12.5, 8, 6);
  F.ceilingLight(b, 12.5, 8, -9);
  F.ceilingLight(b, -14, 8, 3.5, { r: 0.4 });

  // ----------------------------------------------------------- upstairs rooms
  const U = 8.5;
  F.bed(b, -15.3, -4.6, Math.PI / 2, { w: 3.4, l: 6.6, pillows: 1, y: U, quilt: T.quilt('#f1e9dc'), frame: '#efece4' });
  F.bed(b, -15.3, 6.0, Math.PI / 2, { w: 3.4, l: 6.6, pillows: 1, y: U, quilt: T.quilt('#dfe8f0'), frame: '#efece4' });
  F.dresser(b, -5.2, -7.9, 0, { w: 3.6, h: 3.6, y: U, color: '#dad6cc' });
  F.dresser(b, -2.1, 3.5, -Math.PI / 2, { w: 3, h: 4.2, y: U, color: '#c8c6be' });
  F.rug(b, -14, -6, -2, 4, U + 0.03, mat('#e8dfcc', { roughness: 1 }));
  const fan = new THREE.Group();
  const fanMat = mat('#f1efe9');
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.35, 0.5, 12), fanMat);
  fan.add(hub);
  for (let i = 0; i < 5; i++) {
    const blade = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.05, 0.5), fanMat);
    blade.position.x = 1.2;
    const holder = new THREE.Group();
    holder.rotation.y = (i / 5) * Math.PI * 2;
    holder.add(blade);
    fan.add(holder);
  }
  fan.position.set(-10, 15.4, 0.5);
  b.add(fan, { occlude: false });
  F.picture(b, 'z', -1.2, 6.0, U + 5.6, 2.0, 1.5, -1, T.painting(23));

  F.bed(b, 15.1, 3.4, -Math.PI / 2, { w: 5, l: 6.8, y: U, quilt: T.quilt('#ece2d2') });
  F.dresser(b, 17.9, -0.4, -Math.PI / 2, { w: 1.6, h: 2.1, d: 1.4, y: U });
  F.dresser(b, 17.9, 7.2, -Math.PI / 2, { w: 1.6, h: 2.1, d: 1.4, y: U });
  F.lamp(b, 17.9, U + 2.3, 7.2, { base: '#7a96ad' });
  F.dresser(b, 8.6, -7.9, 0, { w: 4, h: 3.4, y: U });
  F.armchair(b, 8.5, 6.5, Math.PI / 2 + 0.4, { color: '#c3cdd6', y: U });
  F.rug(b, 10, 17, 0, 7, U + 0.03, mat('#9fb2c6', { roughness: 1 }));
  F.picture(b, 'z', 6.2, 3.2, U + 5.6, 2.6, 1.8, 1, T.painting(29));

  b.box(-0.6, 5.6, U + 0.02, U + 0.035, -8.6, -3.2, M.bathTile, { occlude: false, cast: false });
  F.tub(b, -0.8, 4.0, -8.6, -6.2, { y: U });
  F.toilet(b, 5.0, -7.4, -Math.PI / 2, { y: U });
  F.vanity(b, 4.9, -4.6, -Math.PI / 2, { w: 2.4, y: U });

  F.ceilingLight(b, -10, 16, -4, { r: 0.5 });
  F.ceilingLight(b, 12.5, 16, -2);
  F.ceilingLight(b, 2.2, 16, -1);
  F.ceilingLight(b, 2.5, 16, -6, { r: 0.4 });

  // Interior point lights (no shadows; sunlight provides those).
  const lights = [
    [12.5, 6.8, 7, 30], [12.5, 6.8, -0.5, 18], [-0.6, 5.8, -1.4, 24], [0, 7.2, -9.5, 24],
    [0, 7.2, 10, 20], [12.5, 6.5, -9, 22], [-12.5, 7.2, 10.5, 20], [-12.5, 7.2, -10.5, 20],
    [-13, 7.2, -1.5, 18], [-10, 15, 0, 28], [12, 15, 0, 28], [2.2, 15, -3, 14],
  ];
  for (const [x, y, z, i] of lights) {
    const l = new THREE.PointLight('#ffe9c8', i, 26, 1.5);
    l.position.set(x, y, z);
    b.scene.add(l);
  }

  buildYard(b, M);
}

function buildYard(b, M) {
  const G = -2.5;
  const grassMat = new THREE.MeshStandardMaterial({ map: T.grass(), roughness: 1 });
  b.box(-400, 400, G - 0.2, G, -400, 400, grassMat, { occlude: false, cast: false });

  const asphalt = new THREE.MeshStandardMaterial({ map: T.asphalt(), roughness: 0.95 });
  b.box(-400, 400, G, G + 0.02, 44, 74, asphalt, { occlude: false, cast: false });
  b.box(-400, 400, G, G + 0.4, 43.4, 44, M.concrete, { occlude: false });
  b.box(-400, 400, G, G + 0.06, 38, 43, M.concrete, { occlude: false, cast: false });
  b.box(-400, 400, G, G + 0.06, 76, 81, M.concrete, { occlude: false, cast: false });
  b.box(-400, 400, G, G + 0.4, 74, 74.6, M.concrete, { occlude: false });
  b.box(-1.6, 1.6, G, G + 0.05, 24.8, 38, M.concrete, { occlude: false, cast: false });

  const gravelMat = new THREE.MeshStandardMaterial({ map: T.gravel(), roughness: 1 });
  b.box(-25, -20.5, G, G + 0.04, -110, 43.4, gravelMat, { occlude: false, cast: false });
  b.box(-25, 25, G, G + 0.04, -125, -110, gravelMat, { occlude: false, cast: false });
  b.box(-25, -19.3, G, G + 0.03, -22, -15.3, gravelMat, { occlude: false, cast: false });
  const flag = new THREE.MeshStandardMaterial({ map: T.flagstone(), roughness: 0.95 });
  b.box(-19.3, 19.3, G, G + 0.05, -26, -15.3, flag, { occlude: false, cast: false });

  // Outdoor shower enclosure.
  const vinyl = mat('#f4f4f1', { roughness: 0.5 });
  b.box(-9.1, -8.8, G, 4.3, -18.8, -15.3, vinyl, { collide: true });
  b.box(-4.7, -4.4, G, 4.3, -18.8, -15.3, vinyl, { collide: true });
  b.box(-8.8, -7.1, G, 4.3, -18.8, -18.5, vinyl, { collide: true });
  b.box(-4.9, -4.7, G, 4.3, -18.8, -18.5, vinyl, { collide: true });
  b.box(-7.1, -6.95, G + 0.2, 4.0, -21.2, -18.8, vinyl, { collide: true });
  b.box(-8.8, -4.7, G, G + 0.25, -18.5, -15.3, mat('#8a6a4a'), { occlude: false });
  b.box(-6.85, -6.65, G + 1, 4.6, -15.45, -15.3, mat('#c9ccd0', { metalness: 0.9, roughness: 0.2 }), { occlude: false });
  b.box(-7.1, -6.4, 4.3, 4.45, -16.3, -15.3, mat('#c9ccd0', { metalness: 0.9, roughness: 0.2 }), { occlude: false });

  // Bilco basement doors.
  b.slopeZ(-15.3, -1.3, -19, -2.35, -17.5, -13.5, 0.2, mat('#f0f0ec', { roughness: 0.5 }));
  b.collider(-17.5, -13.5, G, -1.25, -19, -15.3);

  // AC condenser by the porch.
  b.box(15.8, 18.6, G, G + 2.8, -18.4, -15.8, mat('#c9c6bd', { roughness: 0.6 }), { collide: true });
  b.box(16.2, 18.2, G + 2.8, G + 2.85, -18, -16.2, mat('#2a2a2a'), { occlude: false });

  F.picnicTable(b, 11, -21, 0.15);
  F.shed(b, 16, -100);

  // Hedges and plantings.
  F.hedge(b, -27.5, -25.5, -110, 30, 8);
  F.hedge(b, 24.5, 26.5, -110, -22, 9);
  F.hedge(b, -25.5, 24.5, -112, -110, 7);
  for (let x = 7; x <= 18; x += 2.6) F.bush(b, x, 16.6, { r: 1.5, color: '#3d6b2b', flowers: x % 2 < 1.3 ? '#d98fb4' : '#8fb0de' });
  for (const x of [-18, -16, -11, -5.5]) F.bush(b, x, 16.4, { r: 1.2, color: '#4a7a35' });
  F.bush(b, -7.8, 23.8, { r: 3.2, color: '#3b6b2a' });
  F.bush(b, 22.5, -4, { r: 2, color: '#3b6b2a' });
  F.bush(b, 22, 10, { r: 1.8, color: '#3f6e2c', flowers: '#d98fb4' });

  F.tree(b, -8.5, 25, { trunk: 5, crown: 9, maple: true, seed: 5, lean: 0.08 });
  F.tree(b, 18, -48, { trunk: 12, crown: 14, seed: 3 });
  F.tree(b, -12, -72, { trunk: 14, crown: 15, color: '#466f2f', seed: 9 });
  F.tree(b, 8, -88, { trunk: 10, crown: 11, seed: 13 });
  F.tree(b, -18, -36, { trunk: 8, crown: 9, color: '#527f36', seed: 21 });
  F.tree(b, 30, 30, { trunk: 10, crown: 12, seed: 17 });
  F.tree(b, -40, 34, { trunk: 11, crown: 12, seed: 27, maple: true });

  // Neighbors.
  F.neighborHouse(b, -52, 0, { w: 26, d: 40, h: 20, color: '#e8ecee' });
  F.neighborHouse(b, 50, -2, { w: 24, d: 36, h: 19, color: '#f1f0ea', roof: '#55595f' });
  F.neighborHouse(b, -100, 4, { w: 28, d: 38, h: 22, color: '#a8bed1' });
  F.neighborHouse(b, 100, 0, { w: 26, d: 34, h: 18, color: '#e7dfcc' });
  F.neighborHouse(b, -45, 110, { w: 28, d: 34, h: 21, color: '#dbe3e8' });
  F.neighborHouse(b, 5, 112, { w: 26, d: 36, h: 18, color: '#f3f2ec' });
  F.neighborHouse(b, 55, 110, { w: 30, d: 36, h: 22, color: '#9fb6c9' });
  F.neighborHouse(b, -10, -160, { w: 28, d: 36, h: 20, color: '#ece9df' });
  F.neighborHouse(b, 45, -150, { w: 24, d: 30, h: 17, color: '#d9e2e6' });
  for (const x of [-80, -30, 30, 80]) F.tree(b, x, 41, { trunk: 12, crown: 11, seed: Math.abs(x) + 3 });

  // Street sign at the curb.
  const post = mat('#2f3a2f', { metalness: 0.4, roughness: 0.6 });
  b.box(-30.15, -29.85, G, G + 9, 42.2, 42.5, post, { collide: true });
  const st = new THREE.Mesh(new THREE.PlaneGeometry(5, 1.25), new THREE.MeshStandardMaterial({ map: T.signTexture('Jersey Ave'), side: THREE.DoubleSide }));
  st.position.set(-30, G + 8.2, 42.35);
  b.add(st, { occlude: false });

  b.collider(-400, 400, G, 30, 120, 122);
  b.collider(-400, 400, G, 30, -200, -198);
  b.collider(-200, -198, G, 30, -200, 122);
  b.collider(198, 200, G, 30, -200, 122);
}

export const ROOMS = [
  { name: 'Living Room', level: 0, box: [6, 19, -3, 15] },
  { name: 'Foyer', level: 0, box: [-6, 6, 5, 15] },
  { name: 'Dining Room', level: 0, box: [-6, 6, -5, 5] },
  { name: 'Kitchen', level: 0, box: [-6, 6, -15, -5] },
  { name: 'Screened Porch', level: 0, box: [6, 19, -15, -3] },
  { name: 'Front Bedroom (TV Room)', level: 0, box: [-19, -6, 6, 15] },
  { name: 'Hall Bathroom', level: 0, box: [-19, -9, 1, 6] },
  { name: 'Hallway', level: 0, box: [-9, -6, -6, 6] },
  { name: 'Middle Bedroom', level: 0, box: [-19, -9, -6, 1] },
  { name: 'Back Bedroom', level: 0, box: [-19, -6, -15, -6] },
  { name: 'Front Porch', level: 0, box: [-4.6, 4.6, 15, 21] },
  { name: 'Upstairs Bedroom (Left)', level: 1, box: [-19, -1, -9, 11] },
  { name: 'Upstairs Bedroom (Right)', level: 1, box: [6, 19, -9, 11] },
  { name: 'Upstairs Bathroom', level: 1, box: [-1, 6, -9, -3] },
  { name: 'Upstairs Landing', level: 1, box: [-1, 6, -3, 11] },
];

export function roomAt(x, y, z) {
  if (x > 3 && x < 6 && z > STAIR_TOP && z < STAIR_BOTTOM && y > 0.3 && y < 8.3) return 'Staircase';
  const level = y > 4 ? 1 : 0;
  for (const r of ROOMS) {
    const [x0, x1, z0, z1] = r.box;
    if (r.level === level && x >= x0 && x <= x1 && z >= z0 && z <= z1) return r.name;
  }
  if (z > 43.4 && z < 74) return 'Jersey Avenue';
  if (z >= 21 && z <= 43.4 && x > -26 && x < 26) return 'Front Yard';
  if (z < -15 && z > -112 && x > -26 && x < 26) return z < -95 ? 'Back of the Yard' : 'Backyard';
  if (x > -26 && x < 26 && z > -112 && z < 43) return x < -19 ? 'Driveway' : 'Side Yard';
  return 'Spring Lake';
}

export const TELEPORTS = [
  { name: 'Front walk', pos: [0, -2.5, 34], yaw: 0 },
  { name: 'Front door', pos: [0, 0, 12.5], yaw: 0 },
  { name: 'Living room', pos: [12.5, 0, 4], yaw: 0.3 },
  { name: 'Dining room', pos: [-3, 0, 3.2], yaw: 0 },
  { name: 'Kitchen', pos: [0, 0, -7.5], yaw: 0 },
  { name: 'Screened porch', pos: [10.5, -1, -10.5], yaw: -0.8 },
  { name: 'Bedrooms', pos: [-7.5, 0, 3], yaw: 0 },
  { name: 'Upstairs', pos: [2.5, 8.5, -1], yaw: Math.PI / 2 },
  { name: 'Backyard', pos: [2, -2.5, -38], yaw: Math.PI },
];
