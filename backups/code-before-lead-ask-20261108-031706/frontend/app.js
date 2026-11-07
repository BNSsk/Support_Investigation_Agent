/* Support Investigation Agent - minimal UI (vanilla JS, no build step). */
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x == null ? "–" : Math.round(x * 100) + "%");

const S = { graph: null, threadId: null, counts: {}, last: null, waiting: null, error: null, t0: 0,
  busy: false, scenarios: [], scenario: null, step: 0, renderSeq: 0, renderQueued: false };

const LABELS = {
  __start__: "START", __end__: "END", intake: "Intake · open case + memory", retrieve: "Hybrid retrieval (Qdrant)",
  triage: "Triage · classify + pick skills", investigate: "Investigate (LLM decides)",
  tools: "Tools · 7 (incl. ask customer ⏸)", respond: "Respond · resolve or escalate", evaluate: "Evidence check",
  review: "Review ⏸ (trust gate)", finalize: "Finalize · persist + index", update_memory: "Update long-term memory",
};
const COLOR = {
  retrieve: "var(--info)", tools: "var(--info)", triage: "var(--accent)", investigate: "var(--accent)",
  respond: "var(--accent)", evaluate: "var(--accent)", review: "var(--warn)", intake: "var(--ok)",
  finalize: "var(--ok)", update_memory: "var(--live)",
};

mermaid.initialize({
  startOnLoad: false, theme: "base", securityLevel: "loose",
  themeVariables: { darkMode: true, background: "#151b23", primaryColor: "#1b222c", primaryTextColor: "#e6edf3",
    primaryBorderColor: "#3a4452", lineColor: "#566275", fontFamily: "Inter, sans-serif", fontSize: "13px" },
  flowchart: { curve: "basis", nodeSpacing: 24, rankSpacing: 30, htmlLabels: true, padding: 8 },
});

// ------------------------------------------------------------------ API helpers
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

async function stream(url, body) {
  setBusy(true);
  try {
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        if (chunk.startsWith("data: ")) handleEvent(JSON.parse(chunk.slice(6)), true);
      }
    }
  } catch (e) {
    handleEvent({ type: "error", message: String(e) }, true);
  } finally {
    setBusy(false);
    loadCases(); loadMemory(); loadHealth();
  }
}

function setBusy(b) {
  S.busy = b;
  document.querySelectorAll("button.run").forEach((x) => (x.disabled = b));
  $("#submitBtn").disabled = b;
  $("#submitBtn").innerHTML = b ? '<span class="spinner"></span> Agent working…' : "Investigate ticket";
}

// ------------------------------------------------------------------ graph
function buildMermaid() {
  const g = S.graph;
  let s = "flowchart TD\n";
  for (const n of g.nodes) {
    const c = S.counts[n] || 0;
    const label = (LABELS[n] || n) + (c > 1 ? ` ×${c}` : "");
    s += n === "__start__" || n === "__end__" ? `  ${n}(["${label}"])\n` : `  ${n}("${label}")\n`;
  }
  for (const e of g.edges) s += `  ${e.source} ${e.conditional ? "-.->" : "-->"} ${e.target}\n`;
  s += "  classDef idle fill:#1b222c,stroke:#3a4452,color:#c9d1d9\n";
  s += "  classDef visited fill:#173b25,stroke:#3fb950,color:#e6edf3\n";
  s += "  classDef active fill:#1d3a6e,stroke:#58a6ff,stroke-width:3px,color:#ffffff\n";
  s += "  classDef waiting fill:#5a4313,stroke:#e3a72f,stroke-width:3px,color:#ffffff\n";
  s += "  classDef error fill:#5c1d22,stroke:#f85149,stroke-width:3px,color:#ffffff\n";
  const started = Boolean(S.last || S.waiting || Object.keys(S.counts).length);
  const special = (n) => n === S.last || n === S.waiting;
  const visited = g.nodes.filter((n) => !special(n) && (S.counts[n] || (n === "__start__" && started)));
  const idle = g.nodes.filter((n) => !special(n) && !visited.includes(n));
  if (idle.length) s += `  class ${idle.join(",")} idle\n`;
  if (visited.length) s += `  class ${visited.join(",")} visited\n`;
  if (S.last && g.nodes.includes(S.last)) s += `  class ${S.last} ${S.error ? "error" : "active"}\n`;
  if (S.waiting) s += `  class ${S.waiting} waiting\n`;
  return s;
}

function renderGraph() {
  if (!S.graph || S.renderQueued) return;
  S.renderQueued = true;
  requestAnimationFrame(async () => {
    S.renderQueued = false;
    try {
      const { svg } = await mermaid.render("g" + ++S.renderSeq, buildMermaid());
      $("#graph").innerHTML = svg;
    } catch (e) { console.error(e); }
  });
}

// ------------------------------------------------------------------ events
function resetRun(threadId) {
  S.threadId = threadId; S.counts = {}; S.last = null; S.waiting = null; S.error = null; S.done = false; S.t0 = 0;
  S.tokens = 0; S.pendingLLM = []; S.waitItem = null;
  $("#timeline").innerHTML = "";
  $("#actionCard").classList.add("hidden");
  $("#resultCard").classList.add("hidden");
  $("#followupCard").classList.add("hidden");
  $("#threadLabel").textContent = threadId || "";
  renderGraph();
}

function handleEvent(ev, live) {
  if (ev.type === "run_started") {
    if (ev.thread_id !== S.threadId) resetRun(ev.thread_id);
    S.t0 = S.t0 || ev.ts; S.waiting = null; S.error = null;
    $("#actionCard").classList.add("hidden");
    $("#threadLabel").textContent = ev.thread_id;
    addItem({ node: "run", summary: "Run started", color: "var(--muted)", ts: ev.ts });
  } else if (ev.type === "llm_call") {
    // buffered and attached to the node that made the call (its update event follows)
    (S.pendingLLM = S.pendingLLM || []).push(ev);
    S.tokens = (S.tokens || 0) + (ev.counted_tokens || 0);
    $("#threadLabel").textContent = `${S.threadId || ""} · ${S.tokens.toLocaleString()} Groq tokens counted`;
    S.waitItem = null;
  } else if (ev.type === "llm_note") {
    addItem({ node: "llm", summary: `${ev.label}: ${ev.note}`, color: "var(--muted)", ts: ev.ts });
  } else if (ev.type === "rate_wait") {
    const txt = `Pacing, not an error: holding the ${ev.label} call ${ev.wait_s}s so it fits ${ev.model.split("/").pop()}'s per-minute token limit`;
    if (S.waitItem) S.waitItem.querySelector(".ev-sum").textContent = txt;
    else S.waitItem = addItem({ node: "pacing", summary: txt, color: "var(--muted)", ts: ev.ts });
  } else if (ev.type === "node") {
    S.counts[ev.node] = (S.counts[ev.node] || 0) + 1;
    S.last = ev.node;
    const calls = S.pendingLLM || []; S.pendingLLM = []; S.waitItem = null;
    const tok = calls.length ? "  ·  " + calls.map((c) => `${c.model.split("/").pop()}: ${c.input_tokens}→${c.output_tokens} tok${c.cached_tokens ? ` (${c.cached_tokens} cached)` : ""}`).join(", ") : "";
    addItem({ node: ev.node, summary: ev.summary + tok, data: ev.data, ts: ev.ts });
  } else if (ev.type === "interrupt") {
    S.waiting = ev.interrupt?.type === "review" ? "review" : "tools"; // ask_customer runs inside the tool node
    addItem({ node: S.waiting, summary: ev.interrupt?.type === "review" ? "Paused: waiting for support-lead review" : "Paused: waiting for customer clarification", color: "var(--warn)" });
    showAction(ev.interrupt);
  } else if (ev.type === "completed") {
    S.done = true; S.counts.__end__ = 1; S.last = "__end__";
    addItem({ node: "END", summary: `Completed - ${ev.final?.status || ""}`, color: "var(--ok)" });
    showResult(ev.final);
  } else if (ev.type === "error") {
    S.error = ev.message;
    addItem({ node: "error", summary: ev.message, color: "var(--bad)", retry: true });
  }
  if (live) renderGraph();
}

function addItem({ node, summary, data, ts, color, retry }) {
  const li = document.createElement("li");
  li.style.setProperty("--dot", color || COLOR[node] || "var(--text)");
  const t = ts && S.t0 ? `+${(ts - S.t0).toFixed(1)}s` : "";
  li.innerHTML = `<div class="ev-h"><span class="ev-node">${esc(node)}</span><span class="ev-time">${t}</span></div>
    <div class="ev-sum">${esc(summary)}</div>${details(node, data)}${retry && S.threadId ? '<button class="btn tiny run" onclick="retryRun()">retry from last checkpoint</button>' : ""}`;
  const tl = $("#timeline");
  tl.querySelector(".empty")?.remove();
  tl.appendChild(li);
  tl.scrollTop = tl.scrollHeight;
  return li;
}

function details(node, d) {
  if (!d) return "";
  let inner = "";
  const msgs = d.messages || [];
  if (node === "investigate") {
    const m = msgs[msgs.length - 1] || {};
    if (m.tool_calls?.length) inner = m.tool_calls.map((t) => `<div class="toolcall">${esc(t.name)}(${esc(JSON.stringify(t.args))})</div>`).join("");
    else inner = `<pre>${esc(m.content)}</pre>`;
  } else if (node === "tools") {
    inner = msgs.map((m) => { let c = m.content; try { c = JSON.stringify(JSON.parse(c), null, 1); } catch (_) {} return `<div class="toolcall">${esc(m.tool || "tool")}</div><pre>${esc(String(c).slice(0, 3000))}</pre>`; }).join("");
  } else if (d.evidence && node === "retrieve") {
    inner = "<pre>" + Object.values(d.evidence).map((e) => `${e.case_id}  sim=${e.similarity ?? "-"}  ${e.lead_verified ? "[LEAD-VERIFIED] " : e.source === "live" ? "[NEW CASE] " : ""}${e.queue}/${e.type}\n   ${e.subject}`).join("\n") + "</pre>";
    if (d.label_votes) inner += `<pre>label votes: ${esc(JSON.stringify(d.label_votes))}</pre>`;
  } else if (node === "triage") {
    inner = `<pre>${esc(d.classification?.rationale)}\nskills suggested: ${esc((d.skills || []).join(", "))}</pre>`;
  } else {
    const copy = { ...d }; delete copy.messages;
    inner = `<pre>${esc(JSON.stringify(copy, null, 1).slice(0, 4000))}</pre>`;
  }
  return inner ? `<details><summary>details</summary>${inner}</details>` : "";
}

// ------------------------------------------------------------------ interrupts
function showAction(intr) {
  const card = $("#actionCard");
  card.classList.remove("hidden");
  if (intr?.type === "clarification") {
    const preset = S.scenario?.steps?.[S.step]?.clarification_answer || "";
    card.innerHTML = `<h3>⏸ The agent needs more information from the customer</h3>
      <div class="q">“${esc(intr.question)}”</div>
      ${intr.missing_information?.length ? `<p class="muted small">Missing: ${esc(intr.missing_information.join("; "))}</p>` : ""}
      <textarea id="clarAnswer" rows="3" placeholder="Customer's answer…">${esc(preset)}</textarea>
      <div class="btnrow"><button class="btn primary run" style="width:auto" onclick="sendClarification()">Send answer &amp; resume</button></div>`;
  } else {
    const d = intr?.draft || {}; const e = intr?.evaluation || {};
    const g = intr?.lead_guidance || {};
    const st = S.scenario?.steps?.[S.step] || {};
    card.innerHTML = `<h3>⏸ Support-lead review before the case is finalised</h3>
      <p class="muted small">Accepted resolutions become <b>trusted cases</b> indexed for future retrieval. Evidence check: <b class="${e.passed ? "ok" : "no"}">${e.passed ? "passed" : "not passed"}</b> · confidence ${pct(e.confidence)}</p>
      <div class="sec"><h3>${d.kind === "escalation" ? "Escalation - customer holding reply" : "Proposed customer reply"}</h3><div class="reply">${esc(d.resolution)}</div></div>
      <div class="sec"><h3>Analysis</h3><div class="analysis">${esc(d.analysis)}</div></div>
      ${leadFields("review", g.root_cause || st.lead_root_cause, g.solution || st.lead_solution, g.root_cause || g.solution
        ? "Your diagnosis is applied to the reply above. Accept to send it and index the case as <b>lead-verified</b>, or change it and apply again."
        : "Know the real cause or a fix the evidence doesn't have? Enter it: the agent rewrites the reply from it, and after you accept, similar tickets get your fix.")}
      <textarea id="reviewFeedback" rows="2" placeholder="Feedback for the agent (required for reject; saved as a note on accept)…" style="margin-top:10px"></textarea>
      <div class="btnrow"><button class="btn ok run" onclick="sendReview('accept')">Accept</button><button class="btn lead run" onclick="sendReview('amend')">Apply my diagnosis</button><button class="btn bad run" onclick="sendReview('reject')">Reject &amp; re-investigate</button></div>`;
  }
  card.scrollIntoView({ behavior: "smooth", block: "center" });
}

// Root cause + solution inputs (review card and finished-case card share them)
function leadFields(prefix, rootCause, solution, hint) {
  return `<div class="sec leadbox"><h3>Support-lead diagnosis (optional)</h3>
    <p class="muted small">${hint}</p>
    <textarea id="${prefix}RootCause" rows="2" placeholder="Root cause - why it is happening…">${esc(rootCause || "")}</textarea>
    <textarea id="${prefix}Solution" rows="3" placeholder="Solution / fix - what resolves it…">${esc(solution || "")}</textarea></div>`;
}

window.sendClarification = () => stream(`/api/threads/${S.threadId}/resume`, { answer: $("#clarAnswer").value, auto_accept: $("#autoAccept").checked });
window.sendReview = (action) => {
  const fb = $("#reviewFeedback").value;
  const root_cause = $("#reviewRootCause").value.trim(), solution = $("#reviewSolution").value.trim();
  if (action === "reject" && !fb.trim()) { $("#reviewFeedback").focus(); return; }
  if (action === "amend" && !root_cause && !solution) { $("#reviewSolution").focus(); return; }
  stream(`/api/threads/${S.threadId}/resume`, { action, feedback: fb, root_cause, solution, auto_accept: $("#autoAccept").checked });
};
window.sendLeadSolution = () => {
  const root_cause = $("#doneRootCause").value.trim(), solution = $("#doneSolution").value.trim();
  if (!root_cause && !solution) { $("#doneSolution").focus(); return; }
  stream(`/api/threads/${S.threadId}/lead-solution`, { root_cause, solution, note: $("#doneNote").value.trim(), auto_accept: $("#autoAccept").checked });
};
window.retryRun = () => stream(`/api/threads/${S.threadId}/retry`, {});

// ------------------------------------------------------------------ result
function showResult(f) {
  if (!f) return;
  const card = $("#resultCard");
  card.classList.remove("hidden");
  $("#followupCard").classList.remove("hidden");
  const c = f.classification || {}; const ev = f.evaluation || {}; const esc_ = f.escalation || {};
  const conf = f.confidence ?? 0;
  const ringColor = conf >= 0.7 ? "var(--ok)" : conf >= 0.5 ? "var(--warn)" : "var(--bad)";
  const evRows = (f.evidence || []).map((e) => `<tr>
      <td class="id"><a class="link" onclick="showTicket('${esc(e.case_id)}')">${esc(e.case_id)}</a>${srcTag(e)}</td>
      <td>${e.similarity != null ? `<span class="bar"><i style="width:${Math.max(0, Math.min(100, e.similarity * 100))}%"></i></span>${e.similarity.toFixed(2)}` : '<span class="muted">opened</span>'}</td>
      <td><div class="ev-subj">${esc(e.subject || "(no subject)")}</div>
        ${e.issue ? `<div class="ev-issue">${esc(e.issue)}</div>` : ""}
        ${e.how_used ? `<div class="small"><span class="muted">How it was used:</span> ${esc(e.how_used)}</div>` : ""}
        ${e.issue || e.resolution ? `<details><summary>Full description and past resolution</summary>
          ${e.issue ? `<div class="ev-label">Description</div><div class="ev-full">${esc(e.issue)}</div>` : ""}
          ${e.resolution ? `<div class="ev-label">Past resolution</div><div class="ev-full reply">${esc(e.resolution)}</div>` : ""}
        </details>` : ""}</td></tr>`).join("");
  const tk = f.ticket || {};
  const g = f.lead_guidance || {}; const idx = f.index || {}; const rv = f.review || {};
  const leadSec = g.root_cause || g.solution ? `<div class="sec leadbox"><h3>Support-lead diagnosis <span class="src lead">lead-verified</span></h3>
      ${g.root_cause ? `<div class="small"><b>Root cause:</b> ${esc(g.root_cause)}</div>` : ""}
      ${g.solution ? `<div class="small"><b>Solution:</b> ${esc(g.solution)}</div>` : ""}
      ${g.note ? `<div class="small muted">Note: ${esc(g.note)}</div>` : ""}
      <div class="small muted">by ${esc(g.by || "support lead")}${g.at ? " · " + esc(g.at) : ""}</div></div>` : "";
  const idxLine = idx.indexed == null ? "" : `<p class="small ${idx.indexed ? "ok" : "muted"}">${idx.indexed
    ? (idx.lead_verified ? "Indexed as a <b>lead-verified</b> case: similar tickets will retrieve and prefer this fix." : "Indexed as a trusted case for future retrieval.")
    : `Not indexed: ${esc(idx.reason || "")}`}${rv.note ? ` · Review note: ${esc(rv.note)}` : ""}</p>`;
  const leadForm = `<details class="sec leadform"${f.status === "escalated" && !(g.root_cause || g.solution) ? " open" : ""}>
      <summary>${f.status === "escalated" ? "Resolve this escalation with the lead's solution" : "Correct this case with the lead's solution"}</summary>
      ${leadFields("done", g.root_cause, g.solution, "The agent rewrites the customer reply from your root cause and fix, runs the evidence check and asks for your review; once accepted, the case is indexed as <b>lead-verified</b> for similar tickets.")}
      <input id="doneNote" placeholder="Optional note for the agent (e.g. what to tell the customer)…" />
      <div class="btnrow" style="margin-top:8px"><button class="btn lead run" onclick="sendLeadSolution()">Apply the lead's solution</button></div>
    </details>`;
  const checks = Object.entries(ev.checks || {}).map(([k, v]) => `<span class="chip"><span class="${v ? "ok" : "no"}">${v ? "✓" : "✗"}</span> ${esc(k.replaceAll("_", " "))}</span>`).join("");
  card.innerHTML = `
    <div class="result-h">
      <span class="badge ${esc(f.status)}">${esc((f.status || "").toUpperCase())}</span>
      <h3>${esc(f.case_id)}</h3><span class="muted small">customer ${esc(f.customer_id)} · skills: ${esc((f.skills_used || []).join(", "))}</span>
      <div class="conf"><span class="muted small">confidence</span><div class="ring" style="--p:${Math.round(conf * 100)};--ring:${ringColor}"><span>${pct(conf)}</span></div></div>
    </div>
    <div class="chips">
      <span class="chip"><b>queue</b>${esc(c.queue)}</span><span class="chip"><b>type</b>${esc(c.type)}</span><span class="chip"><b>priority</b>${esc(c.priority)}</span>
      ${(c.tags || []).map((t) => `<span class="chip tag">#${esc(t)}</span>`).join("")}
    </div>
    ${tk.subject || tk.body ? `<div class="sec ticket"><h3>Customer ticket</h3>
      <div class="ticket-subj">${esc(tk.subject || "(no subject)")}</div><div class="ticket-body">${esc(tk.body || "")}</div></div>` : ""}
    ${leadSec}${idxLine}
    <div class="grid2">
      <div class="sec"><h3>Resolution (customer reply)</h3><div class="reply">${esc(f.resolution)}</div>
        ${f.internal_actions?.length ? `<h3>Internal actions</h3><ul class="clean">${f.internal_actions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul>` : ""}
        <h3>Escalation</h3><div>${esc_.required ? `<b class="no">Required</b> → ${esc(esc_.target_team || "")} (${esc(esc_.urgency || "")}): ${esc(esc_.reason)}` : '<span class="ok">Not required</span>'}</div>
        ${f.clarifications?.length ? `<h3>Clarifications</h3>${f.clarifications.map((q) => `<div class="small"><b>Q:</b> ${esc(q.question)}<br><b>A:</b> ${esc(q.answer)}</div>`).join("")}` : ""}
      </div>
      <div class="sec"><h3>Analysis</h3><div class="analysis">${esc(f.analysis)}</div>
        <h3>Evidence check</h3>
        <div class="metrics">
          <div class="metric"><div class="v">${pct(ev.groundedness)}</div><div class="k">groundedness</div></div>
          <div class="metric"><div class="v">${pct(ev.addresses_issue)}</div><div class="k">addresses issue</div></div>
          <div class="metric"><div class="v">${pct(ev.retrieval_strength)}</div><div class="k">retrieval strength</div></div>
          <div class="metric"><div class="v">${pct(ev.label_agreement)}</div><div class="k">label agreement</div></div>
        </div><div class="checks">${checks}</div>
        ${ev.unsupported_claims?.length ? `<p class="small no">Unsupported: ${esc(ev.unsupported_claims.join("; "))}</p>` : ""}
      </div>
    </div>
    <div class="sec"><h3>Evidence used (${(f.evidence || []).length})</h3>
      <table class="evidence"><thead><tr><th>Case</th><th>Similarity</th><th>Ticket used: subject, description, how it was used</th></tr></thead><tbody>${evRows || '<tr><td colspan="3" class="muted">none</td></tr>'}</tbody></table>
    </div>
    ${leadForm}`;
}

function srcTag(e) {
  if (e.source === "lead") return '<span class="src lead">support lead</span>';
  if (e.lead_verified) return '<span class="src lead">lead-verified</span>';
  return e.source === "live" ? '<span class="src live">new case</span>' : '<span class="src">historical</span>';
}

window.showTicket = async (id) => {
  try {
    const lead = id.startsWith("LEAD-");  // a support lead's diagnosis: show the case it belongs to
    const t = await api(`/api/tickets/${encodeURIComponent(lead ? id.slice(5) : id)}`);
    const cls = t.classification || t;
    const g = t.lead_guidance || {};
    $("#modalBody").innerHTML = `<h3>${esc(id)} <span class="src ${t.source === "live" ? "live" : ""}">${esc(t.source)}</span>${g.root_cause || g.solution ? '<span class="src lead">lead-verified</span>' : ""}</h3>
      <div class="chips" style="margin:8px 0"><span class="chip"><b>queue</b>${esc(cls.queue)}</span><span class="chip"><b>type</b>${esc(cls.type)}</span><span class="chip"><b>priority</b>${esc(cls.priority)}</span>${(cls.tags || []).map((x) => `<span class="chip tag">#${esc(x)}</span>`).join("")}</div>
      <div class="sec"><h3>Subject</h3>${esc(t.subject)}</div><div class="sec"><h3>Issue</h3><div class="analysis">${esc(t.body)}</div></div>
      ${g.root_cause || g.solution ? `<div class="sec leadbox"><h3>Support-lead diagnosis</h3>${g.root_cause ? `<div><b>Root cause:</b> ${esc(g.root_cause)}</div>` : ""}${g.solution ? `<div><b>Solution:</b> ${esc(g.solution)}</div>` : ""}</div>` : ""}
      <div class="sec"><h3>Resolution</h3><div class="reply">${esc(t.answer || t.resolution)}</div></div>`;
    $("#modal").classList.remove("hidden");
  } catch (e) { alert(e); }
};
$("#modalClose").onclick = () => $("#modal").classList.add("hidden");
$("#modal").onclick = (e) => { if (e.target.id === "modal") $("#modal").classList.add("hidden"); };

// ------------------------------------------------------------------ sidebar
async function loadHealth() {
  try {
    const h = await api("/api/health");
    const gu = h.groq_usage || {};
    const used = (m) => ((gu.today || {})[m] || {}).counted_tokens || 0;
    const k = (n) => (n >= 1000 ? (n / 1000).toFixed(1) + "K" : String(n));
    const models = [...new Set([h.model, h.fast_model])];
    const usage = models.map((m) => `${esc(m.split("/").pop())} <b>${k(used(m))}</b>`).join(" · ");
    const tpm = models.map((m) => ((gu.models || {})[m] || {}).tokens_per_minute).filter(Boolean);
    $("#healthPills").innerHTML = `
      <span class="pill ${h.groq_configured ? "" : "bad"}">Groq <b>${esc(h.model)}</b>${h.groq_configured ? "" : " · key missing"}</span>
      <span class="pill">fast <b>${esc(h.fast_model)}</b></span>
      <span class="pill" title="Groq tokens used today (cached prompt tokens excluded). No client-side limits; plan limit reported by Groq: ${tpm.length ? tpm.join(" / ") + " tokens/min" : "not seen yet"}">tokens today ${usage}</span>
      <span class="pill">Qdrant <b>${(h.vectors.historical || 0).toLocaleString()}</b> historical · <b>${h.vectors.live || 0}</b> new trusted</span>
      <span class="pill">cases <b>${h.db.cases}</b></span>`;
  } catch (e) { $("#healthPills").innerHTML = `<span class="pill bad">backend offline</span>`; }
}

async function loadCases() {
  try {
    const cases = await api("/api/cases");
    $("#casesList").innerHTML = cases.length ? cases.map((c) => `<div class="case" onclick="openThread('${esc(c.thread_id)}')">
        <div class="t">${esc(c.subject || "(no subject)")}</div>
        <div class="m"><span class="st ${esc(c.status)}">${esc(c.status)}</span>${esc(c.customer_id)}${c.confidence != null ? ` · ${pct(c.confidence)}` : ""}</div></div>`).join("")
      : '<div class="muted small">No cases yet.</div>';
  } catch (_) {}
}

async function loadMemory() {
  const cid = $("#customerId").value.trim();
  if (!cid) return;
  try {
    const m = await api(`/api/customers/${encodeURIComponent(cid)}/memory`);
    const facts = m.facts.map((f) => `<div class="fact"><b>${esc(f.key)}</b> ${esc(f.value)}</div>`).join("");
    const cases = m.cases.map((c) => `<div class="small muted">• ${esc(c.case_id)} <span class="st ${esc(c.status)}">${esc(c.status)}</span> ${esc(c.subject)}</div>`).join("");
    $("#memoryBox").innerHTML = (facts || '<div class="muted small">No long-term facts stored for this customer yet.</div>') +
      (cases ? `<div style="margin-top:8px">${cases}</div>` : "");
  } catch (_) {}
}

window.openThread = async (threadId) => {
  const snap = await api(`/api/threads/${threadId}`);
  resetRun(threadId);
  const v = snap.values || {};
  if (v.customer_id) { $("#customerId").value = v.customer_id; loadMemory(); }
  for (const ev of snap.events || []) handleEvent(ev, false);
  if (!snap.events?.length && v.final) showResult(v.final);
  renderGraph();
};

// ------------------------------------------------------------------ scenarios
function fillStep() {
  const st = S.scenario?.steps?.[S.step];
  if (!st) return;
  // fresh customer per scenario run (same suffix across its steps) so earlier runs' memory/history don't leak in
  $("#customerId").value = st.customer_id + S.runSuffix; $("#subject").value = st.subject; $("#body").value = st.body;
  $("#scenarioHint").textContent = `${S.scenario.description}  (step ${S.step + 1}/${S.scenario.steps.length})`;
  loadMemory();
}

$("#scenarioSelect").onchange = (e) => {
  S.scenario = S.scenarios.find((x) => x.id === e.target.value) || null; S.step = 0;
  S.runSuffix = "-" + Math.random().toString(36).slice(2, 6);
  $("#nextStepBtn").classList.add("hidden");
  if (S.scenario) fillStep(); else $("#scenarioHint").textContent = "";
};
$("#nextStepBtn").onclick = () => { S.step += 1; fillStep(); $("#nextStepBtn").classList.add("hidden"); };

$("#submitBtn").onclick = async () => {
  const body = { customer_id: $("#customerId").value.trim() || "anonymous", subject: $("#subject").value, body: $("#body").value, auto_accept: $("#autoAccept").checked };
  if (!body.subject.trim() && !body.body.trim()) return;
  resetRun(null);
  await stream("/api/tickets", body);
  if (S.scenario && S.step + 1 < S.scenario.steps.length && S.done) $("#nextStepBtn").classList.remove("hidden");
};
$("#followupBtn").onclick = () => {
  const m = $("#followupInput").value.trim();
  if (!m || !S.threadId) return;
  $("#followupInput").value = "";
  stream(`/api/threads/${S.threadId}/followup`, { message: m, auto_accept: $("#autoAccept").checked });
};
$("#refreshCasesBtn").onclick = loadCases;
$("#clearMemBtn").onclick = async () => {
  const cid = $("#customerId").value.trim();
  if (cid && confirm(`Delete long-term memory for ${cid}?`)) { await api(`/api/customers/${encodeURIComponent(cid)}/memory`, { method: "DELETE" }); loadMemory(); }
};
let memTimer; $("#customerId").oninput = () => { clearTimeout(memTimer); memTimer = setTimeout(loadMemory, 400); };

// Show "next step" button also after a review/clarification completes a scenario step
const _origShowResult = showResult;
showResult = function (f) { _origShowResult(f); if (S.scenario && S.step + 1 < S.scenario.steps.length) $("#nextStepBtn").classList.remove("hidden"); };

// ------------------------------------------------------------------ init
(async function init() {
  loadHealth(); loadCases(); loadMemory();
  try {
    S.graph = await api("/api/graph");
    renderGraph();
  } catch (e) { $("#graph").innerHTML = `<p class="muted">Could not load graph: ${esc(e)}</p>`; }
  try {
    S.scenarios = await api("/api/samples");
    $("#scenarioSelect").innerHTML += S.scenarios.map((s) => `<option value="${esc(s.id)}">${esc(s.title)}</option>`).join("");
  } catch (_) {}
})();
