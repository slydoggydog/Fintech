import * as THREE from 'three';

// All textures are drawn at runtime so the walkthrough works offline from a
// single file. Each texture covers `tile` feet; geometry UVs are in feet.

function rand(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function makeTexture(size, tile, draw, seed = 1) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  draw(ctx, size, rand(seed));
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  tex.repeat.set(1 / tile, 1 / tile);
  return tex;
}

function noise(ctx, size, r, amount, alpha = 0.06, light = true) {
  for (let i = 0; i < amount; i++) {
    const v = r() < 0.5 ? 0 : 255;
    ctx.fillStyle = `rgba(${v},${v},${v},${alpha * r()})`;
    const s = 1 + r() * 2;
    ctx.fillRect(r() * size, r() * size, s, s);
  }
  if (!light) return;
}

function shade(hex, amt) {
  const c = new THREE.Color(hex);
  c.offsetHSL(0, 0, amt);
  return `#${c.getHexString()}`;
}

export function cedarShake(base = '#e6eaeb') {
  return makeTexture(512, 4, (ctx, S, r) => {
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, S, S);
    const rows = 8;
    const rh = S / rows;
    for (let row = 0; row < rows; row++) {
      const y = row * rh;
      let x = -r() * 40;
      while (x < S) {
        const w = 30 + r() * 60;
        ctx.fillStyle = shade(base, (r() - 0.5) * 0.05);
        ctx.fillRect(x, y, w, rh);
        ctx.fillStyle = 'rgba(0,0,0,0.18)';
        ctx.fillRect(x, y, 2, rh);
        for (let g = 0; g < 4; g++) {
          ctx.fillStyle = 'rgba(0,0,0,0.035)';
          ctx.fillRect(x + r() * w, y, 1, rh);
        }
        x += w;
      }
      const grad = ctx.createLinearGradient(0, y + rh - 7, 0, y + rh);
      grad.addColorStop(0, 'rgba(0,0,0,0)');
      grad.addColorStop(1, 'rgba(0,0,0,0.28)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, y + rh - 7, S, 7);
    }
    noise(ctx, S, r, 3000, 0.08);
  }, 7);
}

export function clapboard(base = '#f1f3f2') {
  return makeTexture(256, 2, (ctx, S, r) => {
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, S, S);
    const rows = 6;
    const rh = S / rows;
    for (let i = 0; i < rows; i++) {
      const g = ctx.createLinearGradient(0, i * rh, 0, (i + 1) * rh);
      g.addColorStop(0, 'rgba(0,0,0,0.0)');
      g.addColorStop(0.85, 'rgba(0,0,0,0.05)');
      g.addColorStop(1, 'rgba(0,0,0,0.25)');
      ctx.fillStyle = g;
      ctx.fillRect(0, i * rh, S, rh);
    }
    noise(ctx, S, r, 600, 0.05);
  }, 3);
}

export function roofShingles(base = '#6f7378') {
  return makeTexture(512, 4, (ctx, S, r) => {
    ctx.fillStyle = shade(base, -0.08);
    ctx.fillRect(0, 0, S, S);
    const rows = 10;
    const rh = S / rows;
    for (let row = 0; row < rows; row++) {
      let x = -r() * 50;
      while (x < S) {
        const w = 25 + r() * 55;
        ctx.fillStyle = shade(base, (r() - 0.5) * 0.12);
        ctx.fillRect(x + 1, row * rh, w - 2, rh - 3);
        x += w;
      }
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.fillRect(0, row * rh + rh - 3, S, 3);
    }
    for (let i = 0; i < 9000; i++) {
      const v = Math.floor(r() * 255);
      ctx.fillStyle = `rgba(${v},${v},${v},0.12)`;
      ctx.fillRect(r() * S, r() * S, 1.5, 1.5);
    }
  }, 11);
}

export function brick() {
  return makeTexture(512, 4, (ctx, S, r) => {
    ctx.fillStyle = '#b9b2a6';
    ctx.fillRect(0, 0, S, S);
    const rows = 18;
    const bw = S / 6;
    const bh = S / rows;
    for (let row = 0; row < rows; row++) {
      const off = row % 2 ? bw / 2 : 0;
      for (let x = -bw; x < S + bw; x += bw) {
        const h = 2 + r() * 8;
        const l = 30 + r() * 10;
        ctx.fillStyle = `hsl(${h}, ${45 + r() * 15}%, ${l}%)`;
        ctx.fillRect(x + off + 2, row * bh + 2, bw - 4, bh - 4);
      }
    }
    noise(ctx, S, r, 6000, 0.12);
  }, 3);
}

export function oakFloor(base = '#d3a067') {
  return makeTexture(512, 4, (ctx, S, r) => {
    const strips = 21;
    const sw = S / strips;
    for (let i = 0; i < strips; i++) {
      let y = -r() * 200;
      while (y < S) {
        const len = 80 + r() * 260;
        ctx.fillStyle = shade(base, (r() - 0.5) * 0.12);
        ctx.fillRect(i * sw, y, sw, len);
        for (let g = 0; g < 6; g++) {
          ctx.fillStyle = `rgba(90,50,20,${0.05 + r() * 0.08})`;
          ctx.fillRect(i * sw + r() * sw, y, 1, len);
        }
        ctx.fillStyle = 'rgba(60,30,10,0.35)';
        ctx.fillRect(i * sw, y, sw, 1.5);
        y += len;
      }
      ctx.fillStyle = 'rgba(60,30,10,0.3)';
      ctx.fillRect(i * sw, 0, 1, S);
    }
  }, 5);
}

export function carpet(base) {
  return makeTexture(256, 3, (ctx, S, r) => {
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, S, S);
    noise(ctx, S, r, 14000, 0.14);
  }, 13);
}

export function tile(base = '#ebe5d8', grout = '#cfc7b6', count = 4) {
  return makeTexture(256, 4, (ctx, S, r) => {
    ctx.fillStyle = grout;
    ctx.fillRect(0, 0, S, S);
    const t = S / count;
    for (let i = 0; i < count; i++) {
      for (let j = 0; j < count; j++) {
        ctx.fillStyle = shade(base, (r() - 0.5) * 0.04);
        ctx.fillRect(i * t + 1.5, j * t + 1.5, t - 3, t - 3);
      }
    }
    noise(ctx, S, r, 2000, 0.05);
  }, 17);
}

export function grass() {
  return makeTexture(512, 8, (ctx, S, r) => {
    ctx.fillStyle = '#4f8a31';
    ctx.fillRect(0, 0, S, S);
    for (let i = 0; i < 26000; i++) {
      const g = 90 + r() * 80;
      ctx.fillStyle = `rgba(${40 + r() * 40},${g},${20 + r() * 30},0.5)`;
      ctx.fillRect(r() * S, r() * S, 1.5, 3 + r() * 3);
    }
  }, 19);
}

export function concrete(base = '#c9c5bc') {
  return makeTexture(256, 6, (ctx, S, r) => {
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, S, S);
    noise(ctx, S, r, 9000, 0.1);
    ctx.fillStyle = 'rgba(0,0,0,0.15)';
    ctx.fillRect(0, 0, S, 2);
  }, 23);
}

export function asphalt() {
  return makeTexture(256, 8, (ctx, S, r) => {
    ctx.fillStyle = '#4a4b4d';
    ctx.fillRect(0, 0, S, S);
    noise(ctx, S, r, 16000, 0.2);
  }, 29);
}

export function gravel() {
  return makeTexture(256, 4, (ctx, S, r) => {
    ctx.fillStyle = '#c8c3b8';
    ctx.fillRect(0, 0, S, S);
    for (let i = 0; i < 5000; i++) {
      const v = 150 + r() * 90;
      ctx.fillStyle = `rgb(${v},${v - 5},${v - 15})`;
      ctx.beginPath();
      ctx.arc(r() * S, r() * S, 1 + r() * 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }, 31);
}

export function flagstone() {
  return makeTexture(512, 8, (ctx, S, r) => {
    ctx.fillStyle = '#8c877c';
    ctx.fillRect(0, 0, S, S);
    for (let i = 0; i < 90; i++) {
      const cx = r() * S;
      const cy = r() * S;
      const rad = 25 + r() * 35;
      const pts = 6 + Math.floor(r() * 3);
      ctx.fillStyle = `hsl(${30 + r() * 25}, ${8 + r() * 12}%, ${55 + r() * 15}%)`;
      ctx.beginPath();
      for (let p = 0; p < pts; p++) {
        const a = (p / pts) * Math.PI * 2;
        const rr = rad * (0.7 + r() * 0.3);
        const x = cx + Math.cos(a) * rr;
        const y = cy + Math.sin(a) * rr;
        if (p === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
    }
    noise(ctx, S, r, 6000, 0.1);
  }, 37);
}

export function stripes(a = '#1f3557', b = '#f4f4f0', count = 8, vertical = true) {
  return makeTexture(256, 2, (ctx, S) => {
    const w = S / count;
    for (let i = 0; i < count; i++) {
      ctx.fillStyle = i % 2 ? b : a;
      if (vertical) ctx.fillRect(i * w, 0, w, S);
      else ctx.fillRect(0, i * w, S, w);
    }
  });
}

export function quilt(base = '#e9dfcf') {
  return makeTexture(256, 3, (ctx, S, r) => {
    const n = 8;
    const t = S / n;
    const colors = ['#c9b79a', '#8ea4bf', '#d9c8b4', '#b48d7a', '#e8e0d0', '#9bb39a'];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        ctx.fillStyle = colors[Math.floor(r() * colors.length)];
        ctx.fillRect(i * t, j * t, t, t);
        ctx.strokeStyle = base;
        ctx.lineWidth = 2;
        ctx.strokeRect(i * t + 3, j * t + 3, t - 6, t - 6);
      }
    }
  }, 41);
}

export function painting(seed = 3) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 180;
  const ctx = canvas.getContext('2d');
  const r = rand(seed);
  const sky = ctx.createLinearGradient(0, 0, 0, 110);
  sky.addColorStop(0, '#9cc3e4');
  sky.addColorStop(1, '#e8eef2');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, 256, 110);
  ctx.fillStyle = '#4f7fa6';
  ctx.fillRect(0, 100, 256, 30);
  ctx.fillStyle = '#d8c39a';
  ctx.fillRect(0, 125, 256, 55);
  for (let i = 0; i < 40; i++) {
    ctx.fillStyle = `rgba(110,130,70,${0.4 + r() * 0.4})`;
    ctx.fillRect(r() * 256, 120 + r() * 50, 2, -10 - r() * 15);
  }
  ctx.fillStyle = '#f2f2ee';
  ctx.fillRect(170, 50, 14, 60);
  ctx.fillStyle = '#b8413a';
  ctx.fillRect(168, 42, 18, 10);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export function signTexture(text, sub) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#1f3557';
  ctx.fillRect(0, 0, 512, 128);
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 64px Georgia, serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 256, sub ? 52 : 64);
  if (sub) {
    ctx.font = '28px Georgia, serif';
    ctx.fillText(sub, 256, 100);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
