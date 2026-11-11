// 마블 스캔 분석 엔진 (규칙 기반 영상 처리)
// 1) 고기 영역 분할  2) 지방을 근간지방(근육 사이 굵은 지방)과 근내지방(마블링)으로 분리
// 3) 근육 덩어리 분할  4) 덩어리별 마블링(부분 화소 추정)  5) 떡심(황색 연골조직) 검출
(function () {
  'use strict';
  // z: 살코기 중앙값 대비 (밝기 차 − 0.8 × 붉은기 차). 블로그 사진 6장으로 맞춘 임시값
  const Z_LEAN = 8, Z_FAT = 20, Z_SEAM = 26;

  function morph(m, W, H, k, isDil) {
    const tmp = new Uint8Array(W * H), out = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) { const row = y * W;
      for (let x = 0; x < W; x++) { let v = isDil ? 0 : 1;
        for (let dx = -k; dx <= k; dx++) { const xx = x + dx; const p = (xx < 0 || xx >= W) ? 0 : m[row + xx]; if (isDil ? p : !p) { v = isDil ? 1 : 0; break; } }
        tmp[row + x] = v; } }
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { let v = isDil ? 0 : 1;
      for (let dy = -k; dy <= k; dy++) { const yy = y + dy; const p = (yy < 0 || yy >= H) ? 0 : tmp[yy * W + x]; if (isDil ? p : !p) { v = isDil ? 1 : 0; break; } }
      out[y * W + x] = v; }
    return out;
  }
  const dilate = (m, W, H, k) => morph(m, W, H, k, true);
  const erode = (m, W, H, k) => morph(m, W, H, k, false);

  function components(m, W, H, eight) {
    const lab = new Int32Array(W * H), sizes = [0], q = new Int32Array(W * H); let n = 0;
    for (let i = 0; i < W * H; i++) { if (!m[i] || lab[i]) continue;
      n++; let h = 0, t = 0, s = 0; q[t++] = i; lab[i] = n;
      while (h < t) { const p = q[h++], x = p % W, y = (p / W) | 0; s++;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue; if (!eight && dx && dy) continue;
          const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
          const j = yy * W + xx; if (m[j] && !lab[j]) { lab[j] = n; q[t++] = j; } } }
      sizes.push(s); }
    return { lab, sizes };
  }

  function fillHoles(m, W, H) {
    const seen = new Uint8Array(W * H), q = new Int32Array(W * H); let h = 0, t = 0;
    const push = i => { if (!m[i] && !seen[i]) { seen[i] = 1; q[t++] = i; } };
    for (let x = 0; x < W; x++) { push(x); push((H - 1) * W + x); }
    for (let y = 0; y < H; y++) { push(y * W); push(y * W + W - 1); }
    while (h < t) { const p = q[h++], x = p % W, y = (p / W) | 0;
      if (x > 0) push(p - 1); if (x < W - 1) push(p + 1); if (y > 0) push(p - W); if (y < H - 1) push(p + W); }
    const out = new Uint8Array(W * H); for (let i = 0; i < W * H; i++) out[i] = seen[i] ? 0 : 1; return out;
  }

  // 씨앗 라벨을 마스크 안으로 키워 덩어리 경계를 정한다 (다중 출발 BFS)
  function growLabels(seed, mask, W, H) {
    const lab = Int32Array.from(seed), q = new Int32Array(W * H); let h = 0, t = 0;
    for (let i = 0; i < W * H; i++) if (lab[i]) q[t++] = i;
    while (h < t) { const p = q[h++], x = p % W, y = (p / W) | 0, l = lab[p];
      const nb = [x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1, y > 0 ? p - W : -1, y < H - 1 ? p + W : -1];
      for (const j of nb) if (j >= 0 && mask[j] && !lab[j]) { lab[j] = l; q[t++] = j; } }
    return lab;
  }

  function toLab(r, g, b) {
    const f = c => { c /= 255; return c > 0.04045 ? Math.pow((c + 0.055) / 1.055, 2.4) : c / 12.92; };
    const R = f(r), G = f(g), B = f(b);
    let X = (R * 0.4124 + G * 0.3576 + B * 0.1805) / 0.95047, Y = R * 0.2126 + G * 0.7152 + B * 0.0722, Z = (R * 0.0193 + G * 0.1192 + B * 0.9505) / 1.08883;
    const h = t => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116; X = h(X); Y = h(Y); Z = h(Z);
    return [116 * Y - 16, 500 * (X - Y), 200 * (Y - Z)];
  }

  // 0 기타, 1 살코기, 2 지방, 3 반사광
  function classify(r, g, b) {
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), v = mx / 255, s = mx ? (mx - mn) / mx : 0;
    if (r > 60 && r > g * 1.35 && r > b * 1.25 && s > 0.3) return 1;
    if (v > 0.96 && s < 0.1) return 3;
    if (v > 0.55 && s < 0.4 && r >= g - 5 && r >= b - 12) return 2;
    return 0;
  }
  // 떡심 후보: 밝고 노란 기가 있는 연골조직 색 (지방보다 b*가 높음)
  function yellowish(r, g, b) {
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), v = mx / 255, s = mx ? (mx - mn) / mx : 0;
    return v > 0.5 && s > 0.18 && s < 0.75 && r >= g && g > b * 1.15 && (r - g) < 70;
  }


  // ---------------- 등급 판정식 (AI허브 '축산물 품질(QC) 이미지' 도체 사진 400장으로 보정) ----------------
  // 지표: area(지방 면적 지수), hf(결의 섬세함), smallDens(미세 지방 입자 밀도), Am(살코기 붉은 기)
  // 교차검증: 5등급 정확히 맞힘 43%, ±1등급 이내 78%
  const MODEL = { b0: 1.964, beta: [0.052, 0.658, 0.038, -0.537], mu: [19.215, 5.833, 5.298, 27.518], sd: [9.569, 2.596, 3.216, 6.022],
    cuts: [1.479, 1.489, 2.306, 2.502], grades: ['3', '2', '1', '1+', '1++'], acc: { exact: 43, within1: 78, n: 400 } };
  const ZREF_DEFAULT = 72; // 기준 지방이 화면에 없을 때 쓰는 값(보정 데이터 중앙값)

  function boxBlur(src, W, H, r) {
    const N = W * H, t = new Float32Array(N), o = new Float32Array(N), cl = (v, a) => v < 0 ? 0 : v > a ? a : v;
    for (let y = 0; y < H; y++) { let s = 0; for (let x = -r; x <= r; x++) s += src[y * W + cl(x, W - 1)];
      for (let x = 0; x < W; x++) { t[y * W + x] = s / (2 * r + 1); s += src[y * W + cl(x + r + 1, W - 1)] - src[y * W + cl(x - r, W - 1)]; } }
    for (let x = 0; x < W; x++) { let s = 0; for (let y = -r; y <= r; y++) s += t[cl(y, H - 1) * W + x];
      for (let y = 0; y < H; y++) { o[y * W + x] = s / (2 * r + 1); s += t[cl(y + r + 1, H - 1) * W + x] - t[cl(y - r, H - 1) * W + x]; } }
    return o;
  }
  const pctOf = (arr, q) => { const s = Float32Array.from(arr).sort(); return s.length ? s[Math.min(s.length - 1, Math.floor(s.length * q))] : 0; };

  // region: 고기 영역 마스크. pix: {W,H,L,A,V,S}
  function gradeFeatures(pix, region) {
    const { W, H, L, A, V, S } = pix, N = W * H;
    let area = 0; for (let i = 0; i < N; i++) area += region[i];
    const diam = Math.sqrt(area), core = erode(region, W, H, Math.max(1, Math.round(diam * 0.06)));
    const lL = [], lA = []; let nc = 0;
    for (let i = 0; i < N; i++) if (core[i]) { nc++; if (A[i] > 15 && S[i] > 0.3) { lL.push(L[i]); lA.push(A[i]); } }
    if (nc < 200 || lL.length < 50) return null;
    const Lm = pctOf(lL, 0.5), Am = pctOf(lA, 0.5);
    const z = new Float32Array(N); for (let i = 0; i < N; i++) z[i] = (L[i] - Lm) - 0.8 * (A[i] - Am);
    const ref = []; for (let i = 0; i < N; i++) if (!region[i] && V[i] > 0.45 && S[i] < 0.45) ref.push(z[i]);
    const zref = ref.length > N * 0.01 ? Math.max(25, pctOf(ref, 0.7)) : ZREF_DEFAULT;
    const rs = Math.max(2, Math.round(diam * 0.015));
    const b1 = boxBlur(z, W, H, 1), bs = boxBlur(z, W, H, rs);
    let hf = 0, a25 = 0; const fm = new Uint8Array(N);
    for (let i = 0; i < N; i++) { if (!core[i]) continue; hf += Math.abs(b1[i] - bs[i]); if (z[i] > 0.25 * zref) { a25++; fm[i] = 1; } }
    const cc = components(fm, W, H, false), small = cc.sizes.slice(1).filter(s => s <= Math.max(4, nc / 1000 * 0.5)).length;
    const f = { area: a25 / nc * 100, hf: hf / nc / zref * 100, smallDens: small / nc * 1000, Am, zref, refFound: ref.length > N * 0.01 };
    let s = MODEL.b0; [f.area, f.hf, f.smallDens, f.Am].forEach((v, k) => s += MODEL.beta[k] * (v - MODEL.mu[k]) / MODEL.sd[k]);
    let gi = 0; for (const c of MODEL.cuts) if (s >= c) gi++;
    return { ...f, s, gi, grade: MODEL.grades[gi], score: Math.round(Math.min(100, Math.max(0, s / 4 * 100))), core, fm };
  }

  // 화면 안의 고기 덩어리 여러 개 찾기 (여러 팩 비교용)
  function findRegions(lean, W, H, maxN) {
    const N = W * H, cc = components(dilate(lean, W, H, 2), W, H, false);
    const ids = cc.sizes.map((s, l) => [s, l]).filter(([s, l]) => l && s >= N * 0.025).sort((a, b) => b[0] - a[0]).slice(0, maxN);
    const k = Math.max(3, Math.round(Math.max(W, H) * 0.015));
    return ids.map(([s, l]) => { const m = new Uint8Array(N); for (let i = 0; i < N; i++) if (cc.lab[i] === l) m[i] = 1;
      return erode(fillHoles(dilate(m, W, H, k), W, H), W, H, k); });
  }

  function pixelsOf(d, N) {
    const L = new Float32Array(N), A = new Float32Array(N), V = new Float32Array(N), S = new Float32Array(N), lean = new Uint8Array(N);
    for (let i = 0; i < N; i++) { const r = d[i * 4], g = d[i * 4 + 1], b = d[i * 4 + 2], lab = toLab(r, g, b);
      L[i] = lab[0]; A[i] = lab[1]; const mx = Math.max(r, g, b), mn = Math.min(r, g, b); V[i] = mx / 255; S[i] = mx ? (mx - mn) / mx : 0;
      if (classify(r, g, b) === 1) lean[i] = 1; }
    return { L, A, V, S, lean };
  }

  // 여러 팩 비교: 한 장의 사진에서 고기 덩어리를 찾아 각각 판정
  function compare(src, sw, sh, roi, opts) {
    roi = roi || { x: 0, y: 0, w: 1, h: 1 }; opts = opts || {};
    const max = opts.max || 640, cw = sw * roi.w, ch = sh * roi.h, sc = Math.min(1, max / Math.max(cw, ch));
    const W = Math.max(1, Math.round(cw * sc)), H = Math.max(1, Math.round(ch * sc)), N = W * H;
    const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
    const g = cv.getContext('2d', { willReadFrequently: true }); g.drawImage(src, sw * roi.x, sh * roi.y, cw, ch, 0, 0, W, H);
    const img = g.getImageData(0, 0, W, H), px = pixelsOf(img.data, N);
    const regions = findRegions(px.lean, W, H, opts.maxN || 6);
    const pix = { W, H, ...px };
    const items = regions.map(reg => { const f = gradeFeatures(pix, reg); if (!f) return null;
      let sx = 0, sy = 0, n = 0; for (let i = 0; i < N; i++) if (reg[i]) { sx += i % W; sy += (i / W) | 0; n++; }
      return { ...f, region: reg, cx: sx / n, cy: sy / n, size: n / N }; }).filter(Boolean);
    return { img, W, H, items };
  }

  function analyze(src, sw, sh, roi, opts) {
    roi = roi || { x: 0, y: 0, w: 1, h: 1 }; opts = opts || {};
    const max = opts.max || 360, cw = sw * roi.w, ch = sh * roi.h, sc = Math.min(1, max / Math.max(cw, ch));
    const W = Math.max(1, Math.round(cw * sc)), H = Math.max(1, Math.round(ch * sc)), N = W * H;
    const work = document.createElement('canvas'); work.width = W; work.height = H;
    const wctx = work.getContext('2d', { willReadFrequently: true });
    wctx.drawImage(src, sw * roi.x, sh * roi.y, cw, ch, 0, 0, W, H);
    const img = wctx.getImageData(0, 0, W, H), d = img.data;

    const lean0 = new Uint8Array(N), glare = new Uint8Array(N), yel = new Uint8Array(N);
    const Lc = new Float32Array(N), Ac = new Float32Array(N);
    let vSum = 0;
    for (let i = 0; i < N; i++) { const r = d[i * 4], g = d[i * 4 + 1], b = d[i * 4 + 2]; vSum += Math.max(r, g, b);
      const c = classify(r, g, b);
      if (c === 1) lean0[i] = 1; else if (c === 3) glare[i] = 1;
      if (c !== 1 && yellowish(r, g, b)) yel[i] = 1;
      const lab = toLab(r, g, b); Lc[i] = lab[0]; Ac[i] = lab[1]; }
    const brightness = vSum / N / 255;
    const lean = lean0;

    // 고기 영역: 윤곽 좌표(opts.polygon, 원본 화소 좌표)가 주어지면 그 안만 측정한다 (데이터 보정용)
    let region, core;
    if (opts.polygon) {
      const pc = document.createElement('canvas'); pc.width = W; pc.height = H;
      const pg = pc.getContext('2d'); pg.fillStyle = '#fff'; pg.beginPath();
      opts.polygon.forEach(([x, y], j) => { const px = (x - sw * roi.x) * sc, py = (y - sh * roi.y) * sc; if (j) pg.lineTo(px, py); else pg.moveTo(px, py); });
      pg.closePath(); pg.fill();
      const pd = pg.getImageData(0, 0, W, H).data; region = new Uint8Array(N);
      for (let i = 0; i < N; i++) region[i] = pd[i * 4] > 127 ? 1 : 0;
      core = erode(region, W, H, 2);
    } else {
      const cc = components(dilate(lean, W, H, 2), W, H, false);
      let best = 0; for (let i = 1; i < cc.sizes.length; i++) if (cc.sizes[i] > (cc.sizes[best] || 0)) best = i;
      if (!best || cc.sizes[best] < N * 0.08) return { img, W, H, error: '가이드 틀 안에서 고기 단면을 찾지 못했습니다. 틀을 고기로 가득 채워 다시 찍어 주세요.' };
      const main = new Uint8Array(N); for (let i = 0; i < N; i++) main[i] = cc.lab[i] === best ? 1 : 0;
      const k = Math.max(3, Math.round(Math.max(W, H) * 0.03));
      region = erode(fillHoles(dilate(main, W, H, k), W, H), W, H, k);
      core = erode(region, W, H, Math.max(2, Math.round(k * 0.8)));
    }
    let nc = 0; for (let i = 0; i < N; i++) nc += core[i];
    if (nc < 400) return { img, W, H, error: '분석할 영역이 너무 작습니다. 조금 더 가까이에서 찍어 주세요.' };

    // 적응형 지방 판별: 이 사진의 살코기 색(중앙값)을 기준으로 밝고 덜 붉은 정도(z)를 잰다
    const median = arr => { const s = Float32Array.from(arr).sort(); return s.length ? s[s.length >> 1] : 0; };
    const lL = [], lA = []; for (let i = 0; i < N; i += 2) if (core[i] && lean0[i]) { lL.push(Lc[i]); lA.push(Ac[i]); }
    const Lm = median(lL), Am = median(lA);
    const z = new Float32Array(N);
    for (let i = 0; i < N; i++) z[i] = (Lc[i] - Lm) - 0.8 * (Ac[i] - Am);
    const fat = new Uint8Array(N);
    for (let i = 0; i < N; i++) if (core[i] && !glare[i] && z[i] > Z_FAT) { fat[i] = 1; lean[i] = 0; }
    for (let i = 0; i < N; i++) if (core[i] && !glare[i] && !fat[i] && z[i] < Z_FAT) lean[i] = 1;

    // 근간지방(굵고 긴 지방 띠) 분리: 열림 연산으로 굵은 지방만 남긴 뒤 길게 뻗은 것만 채택
    const fatIn = new Uint8Array(N); for (let i = 0; i < N; i++) fatIn[i] = core[i] && ((fat[i] && z[i] > Z_SEAM) || yel[i]) ? 1 : 0;
    const diam = Math.sqrt(nc), r = Math.max(1, Math.round(diam * 0.012));
    const thick = dilate(erode(fatIn, W, H, r), W, H, r);
    const tc = components(thick, W, H, true), seam = new Uint8Array(N);
    { const minX = new Int32Array(tc.sizes.length).fill(1e9), maxX = new Int32Array(tc.sizes.length).fill(-1), minY = new Int32Array(tc.sizes.length).fill(1e9), maxY = new Int32Array(tc.sizes.length).fill(-1);
      for (let i = 0; i < N; i++) { const l = tc.lab[i]; if (!l) continue; const x = i % W, y = (i / W) | 0;
        if (x < minX[l]) minX[l] = x; if (x > maxX[l]) maxX[l] = x; if (y < minY[l]) minY[l] = y; if (y > maxY[l]) maxY[l] = y; }
      const span = diam * 0.3;
      const keep = tc.sizes.map((s, l) => l && (Math.hypot(maxX[l] - minX[l], maxY[l] - minY[l]) >= span || s >= nc * 0.03));
      for (let i = 0; i < N; i++) if (keep[tc.lab[i]]) seam[i] = 1; }

    // 떡심: 노란 기의 굵은 조직 덩어리
    const yc = components(Uint8Array.from(yel, (v, i) => v && core[i] ? 1 : 0), W, H, true);
    let tteokArea = 0; const tteok = new Uint8Array(N);
    { const keep = yc.sizes.map((s, l) => l && s >= nc * 0.006);
      for (let i = 0; i < N; i++) if (keep[yc.lab[i]]) { tteok[i] = 1; tteokArea++; } }

    // 근육 덩어리: 근간지방을 뺀 영역을 침식해 씨앗을 만들고 다시 키운다
    const muscle = new Uint8Array(N); for (let i = 0; i < N; i++) muscle[i] = core[i] && !seam[i] && !tteok[i] ? 1 : 0;
    // 끊긴 근간지방 띠를 이어 붙여 덩어리 경계를 닫는다
    const cut = dilate(seam, W, H, Math.max(2, Math.round(diam * 0.02)));
    const open = new Uint8Array(N); for (let i = 0; i < N; i++) open[i] = muscle[i] && !cut[i] ? 1 : 0;
    const seedMask = erode(open, W, H, Math.max(1, Math.round(diam * 0.012)));

    // 부분 화소 지방 추정: 화소보다 작은 마블링은 색이 반쯤 섞이므로 z를 0~1 비율로 환산해 합산
    const zs = []; for (let i = 0; i < N; i += 2) if (seam[i] && fat[i]) zs.push(z[i]);
    const zFat = Math.max(Z_FAT + 15, zs.length > 30 ? median(zs) : 45);
    const soft = new Float32Array(N);
    for (let i = 0; i < N; i++) soft[i] = Math.min(1, Math.max(0, (z[i] - Z_LEAN) / (zFat - Z_LEAN)));
    const sc2 = components(seedMask, W, H, false), remap = new Int32Array(sc2.sizes.length); let nb = 0;
    for (let l = 1; l < sc2.sizes.length; l++) if (sc2.sizes[l] >= nc * 0.012) remap[l] = ++nb;
    const seeds = new Int32Array(N); for (let i = 0; i < N; i++) seeds[i] = remap[sc2.lab[i]];
    const blockLab = nb ? growLabels(seeds, muscle, W, H) : new Int32Array(N);

    // 덩어리별 지표
    const blocks = [];
    for (let l = 1; l <= nb; l++) blocks.push({ id: l, area: 0, valid: 0, softSum: 0, sx: 0, sy: 0, sxx: 0, syy: 0, sxy: 0 });
    for (let i = 0; i < N; i++) { const l = blockLab[i]; if (!l) continue; const B = blocks[l - 1], x = i % W, y = (i / W) | 0;
      B.area++; B.sx += x; B.sy += y; B.sxx += x * x; B.syy += y * y; B.sxy += x * y;
      if (!glare[i]) { B.valid++; B.softSum += soft[i]; } }
    for (const B of blocks) {
      B.share = B.area / nc; B.marb = B.valid ? B.softSum / B.valid * 100 : 0;
      B.cx = B.sx / B.area; B.cy = B.sy / B.area;
      const vx = B.sxx / B.area - B.cx ** 2, vy = B.syy / B.area - B.cy ** 2, cxy = B.sxy / B.area - B.cx * B.cy;
      const tr = vx + vy, det = vx * vy - cxy * cxy, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
      B.elong = Math.sqrt((tr / 2 + disc) / Math.max(1e-6, tr / 2 - disc));
      delete B.sx; delete B.sy; delete B.sxx; delete B.syy; delete B.sxy;
    }
    // 너무 작은 덩어리(근육 면적의 3% 미만)는 버린다
    for (let k2 = blocks.length - 1; k2 >= 0; k2--) if (blocks[k2].area < nc * 0.03) {
      const id = blocks[k2].id; for (let i = 0; i < N; i++) if (blockLab[i] === id) blockLab[i] = 0; blocks.splice(k2, 1); }
    for (const B of blocks) B.share = B.area / nc;
    blocks.sort((a, b) => b.area - a.area);

    // 전체 지표 (반사광 제외)
    let nl = 0, nf = 0, ng = 0, nseam = 0, sAll = 0, vAll = 0; const Ls = [0, 0, 0], Fs = [0, 0, 0], fineFat = new Uint8Array(N);
    for (let i = 0; i < N; i++) { if (!core[i]) continue;
      if (seam[i]) nseam++;
      if (!glare[i]) { vAll++; sAll += soft[i]; }
      if (lean[i]) { nl++; Ls[0] += d[i * 4]; Ls[1] += d[i * 4 + 1]; Ls[2] += d[i * 4 + 2]; }
      else if (fat[i] && !tteok[i] && !yel[i]) { nf++; Fs[0] += d[i * 4]; Fs[1] += d[i * 4 + 1]; Fs[2] += d[i * 4 + 2]; if (!seam[i]) fineFat[i] = 1; }
      else if (fat[i] && !seam[i]) fineFat[i] = 1;
      else if (glare[i]) ng++; }
    if (nl + nf < 300) return { img, W, H, error: '분석할 영역이 너무 작습니다. 조금 더 가까이에서 찍어 주세요.' };

    // 마블링 섬세함: 주 근육 안 근내지방 입자
    const eye = blocks[0];
    const eyeFat = new Uint8Array(N); if (eye) for (let i = 0; i < N; i++) if (blockLab[i] === eye.id && fat[i]) eyeFat[i] = 1;
    const fc = components(eye ? eyeFat : fineFat, W, H, true), parts = fc.sizes.slice(1).filter(s => s >= 2);
    const meanPart = parts.length ? parts.reduce((a, b) => a + b, 0) / parts.length : 0;

    const V = new Float32Array(N), S = new Float32Array(N);
    for (let i = 0; i < N; i++) { const r = d[i * 4], g = d[i * 4 + 1], b = d[i * 4 + 2], mx = Math.max(r, g, b), mn = Math.min(r, g, b); V[i] = mx / 255; S[i] = mx ? (mx - mn) / mx : 0; }
    const cal = gradeFeatures({ W, H, L: Lc, A: Ac, V, S }, region);
    return { img, W, H, core, fat, fineFat, seam, tteok, blockLab, blocks, soft, cal,
      ratioAll: vAll ? sAll / vAll * 100 : 0,
      ratioEye: eye ? eye.marb : (vAll ? sAll / vAll * 100 : 0),
      seamShare: nseam / nc * 100, tteokShare: tteokArea / nc * 100,
      parts: parts.length, partFrac: meanPart / nc * 1000,
      leanLab: nl ? toLab(Ls[0] / nl, Ls[1] / nl, Ls[2] / nl) : null,
      fatLab: nf ? toLab(Fs[0] / nf, Fs[1] / nf, Fs[2] / nf) : null,
      glarePct: ng / nc * 100, coverPct: nc / N * 100, brightness };
  }

  window.ScanCore = { analyze, compare, classify, MODEL };
})();
