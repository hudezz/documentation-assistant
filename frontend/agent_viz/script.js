// Agent visualisation: draws the pipeline and (in loop mode) replays recorded
// agent runs forever. Config comes from components.py via window.VIZ_CONFIG.

const CONFIG = window.VIZ_CONFIG || { mode: "static", step: 0 };

// The four steps the real agent goes through (see stream_llm in core.py).
const ICONS = {
  question: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 2.5-3 4.5"/><circle cx="12" cy="18" r=".6" fill="currentColor"/></svg>',
  plan: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/></svg>',
  search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="6"/><path d="M20 20l-4.3-4.3"/></svg>',
  write: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.5 6.5l4 4"/></svg>',
  check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
};
const STEPS = [
  { icon: "question", label: "Question" },
  { icon: "plan", label: "Agent plans" },
  { icon: "search", label: "Pinecone search" },
  { icon: "write", label: "Write answer" },
];

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---------------------------------------------------------------------------
// Pipeline. `step` = index of the active node; nodes before it are done.
// step === STEPS.length means every node is done.
// ---------------------------------------------------------------------------
function renderPipeline(step) {
  const parts = [];
  STEPS.forEach((s, i) => {
    if (i > 0) {
      // wire i-1 -> i: done if both ends finished, flowing if it leads to the active node
      const state = step > i ? "done" : step === i ? "flow" : "";
      const packets = state === "flow" ? '<span class="packet"></span>'.repeat(3) : "";
      parts.push(`<div class="wire ${state}">${packets}</div>`);
    }
    const state = i < step ? "done" : i === step ? "active" : "";
    const icon = state === "done" ? ICONS.check : ICONS[s.icon];
    parts.push(
      `<div class="node ${state}"><div class="circle">${icon}</div>` +
      `<div class="node-label">${s.label}</div></div>`
    );
  });
  $("pipeline").innerHTML = parts.join("");
}

// ---------------------------------------------------------------------------
// Loop mode helpers. Text is always set with textContent (never innerHTML)
// so recorded answers can't inject markup.
// ---------------------------------------------------------------------------
function log(tag, text) {
  const line = document.createElement("div");
  line.className = "log";
  const t = document.createElement("span");
  t.className = `tag ${tag}`;
  t.textContent = tag;
  line.append(t, document.createTextNode(text));
  const c = $("console");
  c.append(line);
  while (c.children.length > 14) c.firstChild.remove(); // keep the console short
}

async function typeInto(el, text, msPerChar) {
  el.textContent = "";
  const caret = document.createElement("span");
  caret.className = "caret";
  const node = document.createTextNode("");
  el.append(node, caret);
  if (reducedMotion) { node.data = text; caret.remove(); return; }
  for (let i = 0; i < text.length; i += 2) {
    node.data = text.slice(0, i + 2);
    await sleep(msPerChar);
  }
  caret.remove();
}

function addChip(url) {
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = url.replace(/^https?:\/\/(docs|reference)\.langchain\.com/, "") || "/";
  chip.title = url;
  $("chips").append(chip);
}

function reset() {
  $("question").textContent = "";
  $("answer").textContent = "";
  $("chips").textContent = "";
}

// Replay one recorded run. Timings are sped up a little but the CONTENT
// (search queries, sources, answer) is exactly what the real agent did.
async function play(run) {
  reset();
  renderPipeline(0);
  log("agent", "new question received");
  await typeInto($("question"), run.question, 45);
  await sleep(500);

  const searches = run.searches.length ? run.searches : [run.question];
  const perSearch = Math.ceil(run.sources.length / searches.length);

  for (let s = 0; s < searches.length; s++) {
    renderPipeline(1);
    log("agent", s === 0 ? "planning: I should search the docs" : "need more context, searching again");
    await sleep(1300);

    renderPipeline(2);
    log("tool", `retrieve_context("${searches[s]}")`);
    await sleep(1100);

    const batch = run.sources.slice(s * perSearch, (s + 1) * perSearch);
    for (const url of batch) { addChip(url); await sleep(260); }
    log("pinecone", `← ${batch.length} matching pages`);
    await sleep(500);
  }

  renderPipeline(3);
  log("agent", "writing answer with citations");
  await typeInto($("answer"), run.answer, 14);

  renderPipeline(STEPS.length);
  log("done", `✓ answered in ${run.seconds}s · ${run.sources.length} sources`);
  await sleep(5500);
}

async function loopForever(runs) {
  $("run-count").textContent = `${runs.length} questions on repeat`;
  for (let i = 0; ; i = (i + 1) % runs.length) {
    await play(runs[i]);
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
if (CONFIG.mode === "loop" && CONFIG.runs && CONFIG.runs.length) {
  loopForever(CONFIG.runs);
} else {
  $("viz").classList.add("static");
  renderPipeline(CONFIG.step || 0);
}
