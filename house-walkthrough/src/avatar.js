import * as THREE from 'three';

// A simple stylized person (~5'10"), built from primitives so it needs no
// external model files. Faces +z; pivots let the limbs swing while walking.
export function createAvatar({ shirt = '#3d6fa8', pants = '#2f3540', skin = '#e2b895', hair = '#4a3322', shoes = '#f2f2f2' } = {}) {
  const root = new THREE.Group();
  const m = (c, r = 0.75) => new THREE.MeshStandardMaterial({ color: c, roughness: r });
  const shirtM = m(shirt), pantsM = m(pants), skinM = m(skin, 0.6), hairM = m(hair, 0.9), shoeM = m(shoes, 0.5);

  const body = new THREE.Group();
  root.add(body);

  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.62, 1.15, 6, 14), shirtM);
  torso.scale.set(1, 1, 0.62);
  torso.position.y = 3.95;
  body.add(torso);

  const hips = new THREE.Mesh(new THREE.CapsuleGeometry(0.55, 0.2, 4, 12), pantsM);
  hips.scale.set(1.05, 1, 0.65);
  hips.position.y = 3.05;
  body.add(hips);

  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.18, 0.35, 10), skinM);
  neck.position.y = 5.0;
  body.add(neck);

  const head = new THREE.Group();
  head.position.y = 5.48;
  body.add(head);
  const skull = new THREE.Mesh(new THREE.SphereGeometry(0.42, 20, 16), skinM);
  skull.scale.set(0.92, 1.05, 1);
  head.add(skull);
  const hairCap = new THREE.Mesh(new THREE.SphereGeometry(0.45, 20, 12, 0, Math.PI * 2, 0, Math.PI * 0.55), hairM);
  hairCap.position.set(0, 0.05, -0.04);
  hairCap.scale.set(0.95, 1, 1.02);
  head.add(hairCap);
  const eyeM = m('#20252b', 0.3);
  for (const sx of [-0.14, 0.14]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.045, 8, 8), eyeM);
    eye.position.set(sx, 0.04, 0.39);
    head.add(eye);
  }
  const nose = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 8), skinM);
  nose.position.set(0, -0.05, 0.42);
  head.add(nose);

  const limb = (len, r, mat) => {
    const g = new THREE.Group();
    const mesh = new THREE.Mesh(new THREE.CapsuleGeometry(r, len, 4, 10), mat);
    mesh.position.y = -len / 2 - r * 0.3;
    g.add(mesh);
    return g;
  };

  const legs = [];
  for (const sx of [-0.3, 0.3]) {
    const hip = limb(1.25, 0.21, pantsM);
    hip.position.set(sx, 3.0, 0);
    const knee = limb(1.15, 0.18, pantsM);
    knee.position.y = -1.42;
    hip.add(knee);
    const shoe = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.22, 0.72), shoeM);
    shoe.position.set(0, -1.38, 0.14);
    knee.add(shoe);
    body.add(hip);
    legs.push({ hip, knee });
  }

  const arms = [];
  for (const sx of [-0.78, 0.78]) {
    const shoulder = limb(0.95, 0.15, shirtM);
    shoulder.position.set(sx, 4.62, 0);
    const elbow = limb(0.85, 0.13, skinM);
    elbow.position.y = -1.08;
    shoulder.add(elbow);
    const hand = new THREE.Mesh(new THREE.SphereGeometry(0.15, 10, 8), skinM);
    hand.position.y = -1.05;
    elbow.add(hand);
    body.add(shoulder);
    arms.push({ shoulder, elbow, side: Math.sign(sx) });
  }

  root.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true;
      o.receiveShadow = true;
    }
  });

  let phase = 0;
  let blend = 0;
  root.userData.animate = (dt, speed) => {
    const moving = Math.min(1, speed / 6);
    blend += (moving - blend) * Math.min(1, dt * 8);
    phase += dt * (3.2 + speed * 0.75);
    const s = Math.sin(phase);
    const amp = 0.55 * blend;
    legs[0].hip.rotation.x = s * amp;
    legs[1].hip.rotation.x = -s * amp;
    legs[0].knee.rotation.x = Math.max(0, -Math.cos(phase)) * amp * 1.3;
    legs[1].knee.rotation.x = Math.max(0, Math.cos(phase)) * amp * 1.3;
    for (const a of arms) {
      const dir = a.side < 0 ? -1 : 1;
      a.shoulder.rotation.x = s * amp * 0.8 * dir;
      a.shoulder.rotation.z = a.side * 0.08;
      a.elbow.rotation.x = -0.25 - blend * 0.2;
    }
    body.position.y = Math.abs(Math.cos(phase)) * 0.12 * blend;
    torso.rotation.y = s * 0.06 * blend;
    const idle = Math.sin(performance.now() * 0.0015) * 0.02 * (1 - blend);
    torso.scale.y = 1 + idle;
  };

  const allMats = [shirtM, pantsM, skinM, hairM, shoeM, eyeM];
  let lastOpacity = 1;
  root.userData.setOpacity = (o) => {
    if (Math.abs(o - lastOpacity) < 0.01) return;
    lastOpacity = o;
    for (const mm of allMats) {
      mm.transparent = o < 0.99;
      mm.opacity = o;
      mm.depthWrite = o >= 0.99;
    }
  };

  root.userData.setColors = ({ shirt: sc, pants: pc }) => {
    if (sc) shirtM.color.set(sc);
    if (pc) pantsM.color.set(pc);
  };
  return root;
}
