// Motion-graphics stage: the pipeline as a looping kinetic-type sequence.
//
//   Crawl -> Chunk -> Embed -> Store   (ingestion.py)
//   Search -> Answer Complete          (core.py)
//
// Renders to a fixed full-viewport <canvas>, but only paints inside the
// "stage" rect left of the chat panel (a top band on narrow screens).
//
// Input comes from app.py via hidden elements, picked up by a MutationObserver:
//   .docs-bg-signal[data-config]  index stats shown in the scenes
//   .docs-bg-signal[data-live]    agent progress; holds on Search until done
(() => {
  "use strict";

  // Streamlit re-executes this on every rerun; keep a single instance.
  if (window.DocsBG) { window.DocsBG.ensure(); window.DocsBG.scan(document); return; }

  // Load the companion stylesheet from the same path/version as this script.
  const me = document.currentScript && document.currentScript.src;
  if (me && !document.getElementById("docs-bg-css")) {
    const link = document.createElement("link");
    link.id = "docs-bg-css"; link.rel = "stylesheet";
    link.href = me.replace("background.js", "background.css");
    document.head.appendChild(link);
  }

  // ------------------------------------------------------------ look & feel
  const BG = "#0b0f17";        // app background
  const CUT = "#4f5bff";       // stage fill for "cut" scenes
  const INK = [255, 255, 255];
  const GREEN = [61, 214, 140];
  const HEAD = '"Inter Tight", "Helvetica Neue", Arial, sans-serif';
  const MONO = '"JetBrains Mono", ui-monospace, Menlo, monospace';
  const GLYPHS = "#%&*+=<>/0123456789ABCDEFXZ";

  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
  const expoOut = (t) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));
  const expoIn = (t) => (t <= 0 ? 0 : Math.pow(2, 10 * t - 10));
  const backOut = (t) => 1 + 2.7 * Math.pow(t - 1, 3) + 1.7 * Math.pow(t - 1, 2);
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Stats shown in scenes; defaults are overwritten by data-config.
  const D = {
    source: "docs.langchain.com", depth: 3, pages: 12, chunks: 12,
    size: 4000, overlap: 200, model: "text-embedding-3-small", dims: 1536,
    index: "document-assistant", llm: "gpt-5.2",
    query: "What are deep agents?", topk: 4, sources: 4, seconds: "10.1",
    status: "similarity search",
  };

  let canvas, ctx, W = 0, H = 0, small = false, ART;
  // ST: stage rect. V: portrait stage (art above headline).
  let ST = { x: 0, y: 0, w: 0, h: 0 }, V = false, hidden = false, compact = false;
  let layoutCheck = 0, stageKey = "";
  let scene = 0, t = 0, time = 0, last = 0;
  const live = { hold: null, done: false, label: "" };

  // ------------------------------------------------------------ setup
  function ensure() {
    if (canvas && document.body.contains(canvas)) return;
    canvas = document.createElement("canvas");
    canvas.id = "docs-bg";
    canvas.setAttribute("aria-hidden", "true");
    document.body.prepend(canvas);
    ctx = canvas.getContext("2d");
    resize();
  }
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + "px"; canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    layout();
  }
  // Stage = space between the sidebar and the chat panel. Below 900px it's a
  // top band; background.css offsets the panel to match.
  const GUT = 24, TOP = 64;
  // Outside Streamlit (the GitHub Pages site) an element #docs-stage marks the stage.
  const host = () => document.getElementById("docs-stage");
  function stageRect() {
    const h = host();
    if (h) { const r = h.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; }
    if (W < 900) return { x: 12, y: TOP, w: W - 24, h: Math.round(H * 0.4) };
    let left = 0;
    const sb = document.querySelector('[data-testid="stSidebar"]');
    if (sb) { const r = sb.getBoundingClientRect(); if (r.width > 40 && r.right > 40) left = r.right; }
    const panel = document.querySelector('[data-testid="stMainBlockContainer"]');
    const right = panel ? panel.getBoundingClientRect().left : W * 0.45;
    return { x: Math.round(left + GUT), y: TOP, w: Math.round(right - left - GUT * 2), h: Math.round(H - TOP - GUT) };
  }
  function layout() {
    hidden = ST.w < 300 || ST.h < 220;       // too small to be legible
    V = ST.w / ST.h < 1.15;
    small = ST.w < 520;
    compact = ST.h < 460;                     // short band: drop the stats block
    const pad = Math.max(20, ST.w * 0.07);
    ART = compact
      ? { x: ST.x + pad, y: ST.y + ST.h * 0.17, w: ST.w - pad * 2, h: ST.h * 0.4 }
      : V
      ? { x: ST.x + pad, y: ST.y + ST.h * 0.12, w: ST.w - pad * 2, h: ST.h * 0.4 }
      : { x: ST.x + ST.w * 0.55, y: ST.y + ST.h * 0.12, w: ST.w * 0.4, h: ST.h * 0.62 };
  }
  function configure(data) { Object.assign(D, data || {}); }

  // ------------------------------------------------------------ type helpers
  function spacing(px) { if ("letterSpacing" in ctx) ctx.letterSpacing = px; }

  // Headline lines slide up through a clip mask (expo), with ghost copies as motion blur.
  function headline(lines, dur) {
    let size = clamp(Math.min(ST.w * (V ? 0.2 : 0.115), ST.h * 0.16), 40, 190);
    ctx.font = `800 ${size}px ${HEAD}`;
    spacing(`${-size * 0.045}px`);
    // fit to the available width so it never collides with the art
    const maxW = V ? ST.w * 0.86 : ST.w * 0.47;
    const widest = Math.max(...lines.map((l) => ctx.measureText(l).width));
    if (widest > maxW) {
      size *= maxW / widest;
      ctx.font = `800 ${size}px ${HEAD}`;
      spacing(`${-size * 0.045}px`);
    }
    const lh = size * 0.93;
    const x = ST.x + ST.w * 0.07;
    const base = ST.y + ST.h * (compact ? 0.84 : V ? 0.73 : 0.66) - (lines.length - 1) * lh;
    lines.forEach((line, i) => {
      const y = base + i * lh;
      const inP = reduced ? 1 : expoOut(clamp((t - 0.05 - i * 0.09) / 0.75));
      const outP = reduced ? 0 : expoIn(clamp((t - (dur - 0.55) - i * 0.06) / 0.45));
      const dy = (1 - inP) * lh * 1.1 - outP * lh * 1.1;
      const blur = Math.max(1 - inP, outP);
      ctx.save();
      ctx.beginPath(); ctx.rect(ST.x, y - size * 0.86, ST.w, size * 1.04); ctx.clip();
      if (blur > 0.02) {
        for (let g = 1; g <= 4; g++) {
          ctx.fillStyle = rgba(INK, 0.07);
          ctx.fillText(line, x, y + dy + g * size * 0.07 * blur);
        }
      }
      ctx.fillStyle = rgba(INK, 0.96);
      ctx.fillText(line, x, y + dy);
      ctx.restore();
    });
    spacing("0px");
  }

  // Type-on with a trailing scramble; wipes left-to-right on exit.
  function scramble(text, x, y, start, dur, color, cps = 70) {
    const n = Math.floor((t - start) * cps);
    if (n <= 0) return;
    const wipe = t > dur - 0.45 ? Math.floor((t - (dur - 0.45)) * 140) : 0;
    let out = "";
    for (let j = 0; j < Math.min(n, text.length); j++) {
      if (j < wipe) { out += " "; continue; }
      const settled = reduced || j < n - 5 || text[j] === " ";
      out += settled ? text[j] : GLYPHS[(Math.random() * GLYPHS.length) | 0];
    }
    ctx.fillStyle = color;
    ctx.fillText(out, x, y);
  }

  function stats(rows, dur, cut) {
    if (compact) return;
    const fs = small ? 10 : 12;
    ctx.font = `500 ${fs}px ${MONO}`;
    const x = ST.x + ST.w * 0.07, y0 = ST.y + ST.h * (V ? 0.8 : 0.79), lh = fs * 1.55;
    rows.forEach(([k, v], i) => {
      const line = `${(k.toUpperCase() + ":").padEnd(12)}${v}`;
      scramble(line, x, y0 + i * lh, 0.35 + i * 0.11, dur, rgba(INK, cut ? 0.92 : 0.8));
    });
  }

  // Eased count-up for stat values.
  const tick = (to, start = 0.4, len = 1.3) =>
    Math.round(expoOut(clamp((t - start) / len)) * to);

  // Echo cascade: the value repeated per row, one char shorter each row.
  // Rows pop in unevenly on purpose.
  function cascade(values, x0, y0, colW, size, dur, color) {
    ctx.font = `800 ${size}px ${HEAD}`;
    spacing(`${-size * 0.04}px`);
    values.forEach((v, c) => {
      for (let r = 0; r < v.length; r++) {
        const on = 0.25 + c * 0.12 + r * 0.075;
        const off = dur - 0.5 + (v.length - r) * 0.035;
        if (reduced ? false : t < on || t > off) continue;
        ctx.fillStyle = color;
        ctx.fillText(v.slice(0, v.length - r), x0 + c * colW, y0 + r * size * 0.98);
      }
    });
    spacing("0px");
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }

  // ------------------------------------------------------------ track helpers
  function stadium(cx, cy, w, h) {
    const r = h / 2;
    ctx.beginPath();
    ctx.moveTo(cx - w / 2 + r, cy - r);
    ctx.lineTo(cx + w / 2 - r, cy - r);
    ctx.arc(cx + w / 2 - r, cy, r, -Math.PI / 2, Math.PI / 2);
    ctx.lineTo(cx - w / 2 + r, cy + r);
    ctx.arc(cx - w / 2 + r, cy, r, Math.PI / 2, Math.PI * 1.5);
    ctx.closePath();
  }
  const stadiumLen = (w, h) => 2 * (w - h) + Math.PI * h;
  function stadiumPoint(cx, cy, w, h, u) {
    u = ((u % 1) + 1) % 1;
    const r = h / 2, straight = w - h, half = Math.PI * r;
    let s = u * stadiumLen(w, h);
    if (s < straight) return { x: cx - w / 2 + r + s, y: cy - r };
    s -= straight;
    if (s < half) { const a = -Math.PI / 2 + s / r; return { x: cx + w / 2 - r + Math.cos(a) * r, y: cy + Math.sin(a) * r }; }
    s -= half;
    if (s < straight) return { x: cx + w / 2 - r - s, y: cy + r };
    s -= straight;
    const a = Math.PI / 2 + s / r;
    return { x: cx - w / 2 + r + Math.cos(a) * r, y: cy + Math.sin(a) * r };
  }
  function trackGeom() {
    const w = ART.w, h = Math.min(ART.h * 0.62, ART.w * 0.5);
    return { cx: ART.x + w / 2, cy: ART.y + ART.h * 0.5, w, h, lanes: 7, gap: Math.max(6, h * 0.055) };
  }
  const lane = (g, i) => ({ w: g.w - i * g.gap * 2, h: g.h - i * g.gap * 2 });

  // Draw the concentric track; `p` 0..1 draws each lane on in turn.
  function track(g, p, color, width = 1.4) {
    ctx.strokeStyle = color; ctx.lineWidth = width;
    for (let i = 0; i < g.lanes; i++) {
      const L = lane(g, i);
      const len = stadiumLen(L.w, L.h);
      const lp = reduced ? 1 : expoOut(clamp(p * 1.6 - i * 0.09));
      ctx.setLineDash([len, len]);
      ctx.lineDashOffset = len * (1 - lp);
      stadium(g.cx, g.cy, L.w, L.h); ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  // Vector positions on the track, shared by the Store and Search scenes.
  const STORED = Array.from({ length: 16 }, (_, i) => ({
    lane: 1 + (i % 6), u: (i * 0.137 + (i % 3) * 0.21) % 1, delay: 0.6 + i * 0.07,
  }));

  // ------------------------------------------------------------ scenes
  const SCENES = [
    {
      title: ["Crawl."], lane: "INGESTION · ingestion.py", cut: false, dur: 3.8,
      stats: () => [["source", D.source], ["tool", "Tavily crawl"], ["depth", D.depth], ["pages", tick(D.pages)]],
      art(dur) {
        const cols = 4, rows = 3;
        const cw = Math.min(ART.w / cols - 16, 78), ch = cw * 1.28;
        const gx = ART.x + (ART.w - cols * (cw + 16)) / 2, gy = ART.y + (ART.h - rows * (ch + 16)) / 2;
        const hop = Math.floor(time * 5) % (cols * rows);
        for (let k = 0; k < cols * rows; k++) {
          const inP = reduced ? 1 : backOut(clamp((t - 0.3 - k * 0.06) / 0.45));
          const outP = reduced ? 0 : expoIn(clamp((t - (dur - 0.6) - k * 0.02) / 0.4));
          const s = inP * (1 - outP);
          if (s <= 0.01) continue;
          const x = gx + (k % cols) * (cw + 16) + cw / 2, y = gy + Math.floor(k / cols) * (ch + 16) + ch / 2;
          ctx.save(); ctx.translate(x, y); ctx.scale(s, s);
          ctx.strokeStyle = rgba(k === hop ? GREEN : INK, k === hop ? 0.95 : 0.55);
          ctx.lineWidth = k === hop ? 2 : 1.2;
          ctx.strokeRect(-cw / 2, -ch / 2, cw, ch);
          ctx.fillStyle = rgba(INK, 0.35);
          for (let l = 0; l < 4; l++) ctx.fillRect(-cw / 2 + 8, -ch / 2 + 10 + l * 9, (cw - 16) * (l === 3 ? 0.5 : 1), 2);
          ctx.restore();
        }
        // scan line sweeping across the site
        const sx = gx + ((time * 0.6) % 1) * cols * (cw + 16);
        ctx.fillStyle = rgba(GREEN, 0.5); ctx.fillRect(sx, gy - 10, 1.5, rows * (ch + 16) + 4);
      },
    },
    {
      title: ["Chunk."], lane: "INGESTION · ingestion.py", cut: true, dur: 3.6,
      stats: () => [["splitter", "Recursive"], ["size", D.size], ["overlap", D.overlap], ["chunks", tick(D.chunks)]],
      art(dur) {
        const bw = Math.min(ART.w * 0.5, 280), bh = ART.h * 0.86;
        const bx = ART.x + (ART.w - bw) / 2, by = ART.y + (ART.h - bh) / 2;
        const pieces = 4, ph = bh / pieces;
        const outP = reduced ? 0 : expoIn(clamp((t - (dur - 0.55)) / 0.45));
        for (let i = 0; i < pieces; i++) {
          const inP = reduced ? 1 : expoOut(clamp((t - 0.15) / 0.6));
          const split = reduced ? 1 : expoOut(clamp((t - 1.0 - i * 0.09) / 0.7));
          const dx = split * (i % 2 ? 46 : -46) + outP * W * 0.5;
          const dy = split * (i - 1.5) * 16 + (1 - inP) * H * 0.4;
          const x = bx + dx, y = by + i * ph + dy;
          ctx.fillStyle = rgba(INK, 0.96); ctx.fillRect(x, y, bw, ph - 1);
          ctx.fillStyle = "rgba(79,91,255,0.55)";
          for (let l = 0; l < 4; l++) ctx.fillRect(x + 14, y + 14 + l * (ph / 6), (bw - 28) * (l === 3 ? 0.55 : 1), 3);
          if (split > 0.05) {
            ctx.font = `500 11px ${MONO}`; ctx.fillStyle = rgba(INK, 0.9 * split);
            // narrow stages have no room beside the pieces, so label inside them
            if (small) { ctx.fillStyle = "rgba(79,91,255,0.95)"; ctx.fillText(`chunk_0${i}`, x + bw - 74, y + ph - 10); }
            else ctx.fillText(`chunk_0${i}`, i % 2 ? x + bw + 12 : x - 74, y + ph / 2 + 4);
          }
        }
      },
    },
    {
      title: ["Embed."], lane: "INGESTION · ingestion.py", cut: false, dur: 3.8,
      stats: () => [["model", D.model], ["dims", tick(D.dims, 0.4, 1.6)], ["batch", "concurrent"], ["output", "vectors"]],
      art(dur) {
        const vals = ["0.0381", "-0.127", "0.5520"];
        // both cascades must fit inside ART: 6 rows + gap + 6 small rows ≈ 10 × size
        const size = clamp(Math.min(ART.w / vals.length / 3.6, ART.h / 10), 12, 72);
        cascade(vals, ART.x, ART.y + size, ART.w / vals.length, size, dur, rgba(INK, 0.92));
        cascade(["[1536]"], ART.x, ART.y + size * 6.9, ART.w, size * 0.5, dur, rgba(GREEN, 0.9));
      },
    },
    {
      title: ["Store."], lane: "INGESTION · ingestion.py", cut: false, dur: 3.8,
      stats: () => [["database", "Pinecone"], ["index", D.index], ["metric", "cosine"], ["vectors", tick(D.chunks, 0.6, 1.6)]],
      art(dur) {
        const g = trackGeom();
        const outP = reduced ? 0 : expoIn(clamp((t - (dur - 0.55)) / 0.45));
        ctx.globalAlpha = 1 - outP;
        track(g, clamp((t - 0.1) / 1.2), rgba(INK, 0.7));
        for (const s of STORED) {
          const L = lane(g, s.lane);
          const land = stadiumPoint(g.cx, g.cy, L.w, L.h, s.u + time * 0.015);
          const k = reduced ? 1 : expoOut(clamp((t - s.delay) / 0.7));
          if (k <= 0) continue;
          const from = { x: ST.x + ST.w * 0.3, y: ST.y + ST.h * 0.68 };
          const x = from.x + (land.x - from.x) * k, y = from.y + (land.y - from.y) * k;
          if (k < 0.98) { ctx.strokeStyle = rgba(GREEN, 0.35); ctx.beginPath(); ctx.moveTo(from.x + (land.x - from.x) * Math.max(0, k - 0.25), from.y + (land.y - from.y) * Math.max(0, k - 0.25)); ctx.lineTo(x, y); ctx.stroke(); }
          ctx.fillStyle = rgba(GREEN, 0.95); ctx.beginPath(); ctx.arc(x, y, 3.2, 0, 6.283); ctx.fill();
        }
        ctx.globalAlpha = 1;
      },
    },
    {
      title: ["Search."], lane: "ANSWERING · core.py", cut: true, dur: 4.2,
      stats: () => [["query", `"${D.query}"`.slice(0, 44)], ["tool", "retrieve_context"], ["top_k", D.topk], ["status", D.status]],
      art(dur) {
        const g = trackGeom();
        const outP = reduced ? 0 : expoIn(clamp((t - (dur - 0.55)) / 0.45));
        ctx.globalAlpha = 1 - outP;
        track(g, clamp(t / 0.7), rgba(INK, 0.55));
        const dots = STORED.map((s) => { const L = lane(g, s.lane); return stadiumPoint(g.cx, g.cy, L.w, L.h, s.u); });
        for (const d of dots) { ctx.fillStyle = rgba(INK, 0.7); ctx.beginPath(); ctx.arc(d.x, d.y, 2.6, 0, 6.283); ctx.fill(); }
        // query point laps the outer lane; links to its top-k nearest dots
        const O = lane(g, 0);
        const u = time * 0.28;
        const run = stadiumPoint(g.cx, g.cy, O.w, O.h, u);
        for (let k = 1; k <= 14; k++) {
          const p = stadiumPoint(g.cx, g.cy, O.w, O.h, u - k * 0.006);
          ctx.fillStyle = rgba(INK, 0.5 * (1 - k / 14)); ctx.beginPath(); ctx.arc(p.x, p.y, 4.5 * (1 - k / 18), 0, 6.283); ctx.fill();
        }
        const near = dots.map((d) => ({ d, dist: Math.hypot(d.x - run.x, d.y - run.y) })).sort((a, b) => a.dist - b.dist).slice(0, D.topk);
        const lock = reduced ? 1 : expoOut(clamp((t - 0.9) / 0.5));
        for (const { d } of near) {
          ctx.strokeStyle = rgba(INK, 0.85 * lock); ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.moveTo(run.x, run.y); ctx.lineTo(run.x + (d.x - run.x) * lock, run.y + (d.y - run.y) * lock); ctx.stroke();
          ctx.strokeStyle = rgba(INK, 0.9 * lock); ctx.beginPath(); ctx.arc(d.x, d.y, 7 + Math.sin(time * 8) * 1.5, 0, 6.283); ctx.stroke();
        }
        ctx.fillStyle = rgba(INK, 1); ctx.beginPath(); ctx.arc(run.x, run.y, 6, 0, 6.283); ctx.fill();
        // the question, typed above the track
        ctx.font = `500 ${small ? 12 : 15}px ${MONO}`;
        scramble(`> ${D.query}`, g.cx - g.w / 2, g.cy - g.h / 2 - 26, 0.2, dur, rgba(INK, 0.95), 40);
        ctx.globalAlpha = 1;
      },
    },
    {
      title: ["Answer", "Complete."], lane: "ANSWERING · core.py", cut: false, dur: 4.6,
      stats: () => [["sources", D.sources], ["time", `${D.seconds}s`], ["model", D.llm], ["citations", "yes"]],
      art(dur) {
        const v = `${D.seconds}s`;
        const size = clamp(Math.min(ART.w / 3.6, ART.h / 6.5), 26, 92);
        cascade([v, v], ART.x, ART.y + size, ART.w / 2, size, dur, rgba(INK, 0.92));
        // small track badge, bottom right
        const b = { cx: ST.x + ST.w - 70, cy: ST.y + ST.h - 52, w: 76, h: 36, lanes: 5, gap: 3.5 };
        track(b, clamp((t - 0.6) / 1.0), rgba(INK, 0.8), 1.3);
      },
    },
  ];
  const SEARCH = 4, ANSWER = 5;

  // ------------------------------------------------------------ live hooks
  // step: 1 planning · 2 searching · 3 writing · 4 done
  // Indexing a new site: show the matching ingestion scene and hold on it.
  const INGEST_SCENE = { crawl: 0, documents: 0, chunks: 1, store: 2, batch: 3 };
  function liveIngest(info) {
    if (info.source) D.source = info.source;
    if (info.pages) D.pages = info.pages;
    if (info.chunks) D.chunks = info.chunks;
    live.label = "● LIVE · INDEXING";
    if (info.ingest === "done") { live.hold = null; scene = 3; t = 0; return; }
    const target = INGEST_SCENE[info.ingest];
    if (target === undefined) return;
    if (scene !== target) { scene = target; t = 0; }
    live.hold = target;
  }

  function liveStep(info) {
    if (info && info.ingest) return liveIngest(info);
    const { step, query, sources, seconds } = info || {};
    live.label = "● LIVE · YOUR QUESTION";
    if (query) D.query = query;
    if (step >= 1 && step <= 3) {
      D.status = step === 1 ? "planning" : step === 2 ? "searching pinecone" : "writing answer";
      if (scene !== SEARCH) { scene = SEARCH; t = 0; }
      live.hold = SEARCH;
    } else if (step === 4) {
      if (sources !== undefined) D.sources = sources;
      if (seconds !== undefined) D.seconds = seconds;
      D.status = "done";
      live.hold = null;
      scene = ANSWER; t = 0;
    }
  }

  // ------------------------------------------------------------ chrome
  function chrome(sc) {
    const fs = small ? 9 : 11;
    ctx.font = `500 ${fs}px ${MONO}`;
    const num = String(scene + 1).padStart(2, "0");
    const cx = ST.x + ST.w * 0.07, cy = ST.y + 34;
    scramble(`${num} / 0${SCENES.length}   ${sc.lane}`, cx, cy, 0.1, sc.dur, rgba(INK, sc.cut ? 0.9 : 0.55));
    if (live.hold !== null || D.status === "done" && scene === ANSWER) {
      ctx.fillStyle = rgba(sc.cut ? INK : GREEN, 0.95);
      ctx.fillText(live.label || "● LIVE", cx, cy + fs * 1.9);
    }
    // progress bar along the bottom of the stage: one segment per scene
    const seg = (ST.w - 32) / SCENES.length, by = ST.y + ST.h - 14;
    for (let i = 0; i < SCENES.length; i++) {
      const bx = ST.x + 16 + i * seg;
      ctx.fillStyle = rgba(INK, sc.cut ? 0.25 : 0.1);
      ctx.fillRect(bx + 2, by, seg - 4, 2);
      const p = i < scene ? 1 : i === scene ? clamp(t / sc.dur) : 0;
      ctx.fillStyle = rgba(INK, 0.85);
      ctx.fillRect(bx + 2, by, (seg - 4) * p, 2);
    }
  }

  // ------------------------------------------------------------ main loop
  function frame(now) {
    const dt = Math.min(0.05, (now - (last || now)) / 1000) * (reduced ? 0.6 : 1);
    last = now; time += dt; t += dt;
    layoutCheck -= dt;
    if (layoutCheck <= 0) {                   // poll layout (sidebar toggle, panel resize)
      layoutCheck = host() ? 0 : 0.25;        // a page host can scroll, so track it every frame
      const r = stageRect(), key = `${r.x},${r.y},${r.w},${r.h}`;
      if (key !== stageKey) { stageKey = key; ST = r; layout(); }
    }

    let sc = SCENES[scene];
    if (live.hold === scene && t > sc.dur - 0.6) t = sc.dur - 0.6;   // hold until the agent finishes
    if (t >= sc.dur) { t = 0; scene = (scene + 1) % SCENES.length; sc = SCENES[scene]; if (scene === 0 && D.status === "done") D.status = "similarity search"; }

    // everything outside the stage is flat page background
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, W, H);

    if (!hidden) {
      ctx.save();
      roundRect(ST.x, ST.y, ST.w, ST.h, 20);
      ctx.clip();
      // hard cut between scenes (no crossfade)
      ctx.fillStyle = sc.cut ? CUT : "#0e1320";
      ctx.fillRect(ST.x, ST.y, ST.w, ST.h);
      if (!reduced && t < 0.07) { ctx.fillStyle = "rgba(255,255,255,0.12)"; ctx.fillRect(ST.x, ST.y, ST.w, ST.h); }
      sc.art(sc.dur);
      headline(sc.title, sc.dur);
      stats(sc.stats(), sc.dur, sc.cut);
      chrome(sc);
      ctx.restore();
      roundRect(ST.x + 0.5, ST.y + 0.5, ST.w - 1, ST.h - 1, 20);
      ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 1; ctx.stroke();
    }

    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------ signals from app.py
  function readSignal(el) {
    if (!el.classList || !el.classList.contains("docs-bg-signal") || el.dataset.seen) return;
    el.dataset.seen = "1";
    try {
      if (el.dataset.config) configure(JSON.parse(el.dataset.config));
      if (el.dataset.live) liveStep(JSON.parse(el.dataset.live));
    } catch (e) { /* ignore malformed signal */ }
  }
  function scan(root) {
    if (root.nodeType !== 1 && root !== document) return;
    if (root !== document) readSignal(root);
    root.querySelectorAll(".docs-bg-signal").forEach(readSignal);
  }
  new MutationObserver((muts) => muts.forEach((m) => m.addedNodes.forEach(scan)))
    .observe(document.body, { childList: true, subtree: true });

  window.addEventListener("resize", () => canvas && resize());
  window.DocsBG = { ensure, configure, scan, live: liveStep };
  scan(document);
  ensure();
  // wait for the webfont to avoid a fallback-font first frame
  (document.fonts && document.fonts.load ? document.fonts.load(`800 100px "Inter Tight"`).catch(() => {}) : Promise.resolve())
    .finally(() => requestAnimationFrame(frame));
})();
