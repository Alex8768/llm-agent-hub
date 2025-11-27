const BASE = "http://127.0.0.1:8081";
const pingBtn = document.getElementById("pingBtn");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const input = document.getElementById("input");
const statusEl = document.getElementById("status");
const progressEl = document.getElementById("progress");
const chat = document.getElementById("chat");
const buildEl = document.getElementById("build");
const pulseBtn = document.getElementById("pulseBtn");
const stateSummaryEl = document.getElementById("stateSummary");
const learnInput = document.getElementById("learnInput");
const learnBtn = document.getElementById("learnBtn");
const learnStatus = document.getElementById("learnStatus");
const beliefQuery = document.getElementById("beliefQuery");
const beliefSearchBtn = document.getElementById("beliefSearchBtn");
const beliefList = document.getElementById("beliefList");

let healthy = false;

if (buildEl) {
  const ua = (typeof navigator !== "undefined" && navigator.userAgent) ? navigator.userAgent : "unknown";
  buildEl.textContent = `${new Date().toISOString().slice(0, 19)} · ${ua}`;
}

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = `<div class="muted">${role === "user" ? "You" : "Assistant"}</div><div>${escapeHtml(text)}</div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
}

async function fetchJSON(path, opts = {}) {
  const resp = await fetch(`${BASE}${path}`, opts);
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  return resp.json();
}

async function ping() {
  statusEl.textContent = "⏳";
  try {
    const r = await fetch(`${BASE}/api/status`, { method: "GET" });
    const j = await r.json();
    healthy = r.ok && String(j.detail || "").includes("active");
    statusEl.textContent = healthy ? "✅ Assistant core is active" : "⚠️ Core is not responding";
  } catch (e) {
    healthy = false;
    statusEl.textContent = "❌ no connection";
  }
  sendBtn.disabled = !healthy;
}

async function send() {
  const text = input.value.trim();
  if (!text || !healthy) return;
  input.value = "";
  addMsg("user", text);
  progressEl.textContent = "⏳ Sending…";
  try {
    const r = await fetch(`${BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const j = await r.json();
    addMsg("assistant", j.reply ?? "(empty)");
  } catch (e) {
    addMsg("assistant", "Core connection error");
  } finally {
    progressEl.textContent = "";
  }
}

async function refreshSummary() {
  if (!stateSummaryEl) return;
  stateSummaryEl.textContent = "⏳";
  try {
    const data = await fetchJSON("/api/state/summary");
    const counts = data.counts || {};
    const beliefs = counts.beliefs ?? "—";
    const ragDocs = counts.rag_docs ?? "—";
    stateSummaryEl.innerHTML = `beliefs: <strong>${beliefs}</strong> · rag_docs: ${ragDocs}`;
  } catch (err) {
    stateSummaryEl.textContent = "Error";
  }
}

async function runLearn() {
  if (!learnInput) return;
  const text = learnInput.value.trim();
  if (!text) {
    learnStatus.textContent = "No text";
    return;
  }
  learnStatus.textContent = "⏳ Saving…";
  try {
    const data = await fetchJSON("/api/learn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const message = data.message || `Saved: ${data.committed?.length ?? 0}`;
    learnStatus.textContent = message;
    learnInput.value = "";
    refreshSummary();
    refreshBeliefs();
  } catch (err) {
    learnStatus.textContent = "Error";
  }
}

async function refreshBeliefs() {
  if (!beliefList) return;
  beliefList.innerHTML = "<li>Loading…</li>";
  try {
    const url = new URL(`${BASE}/api/state/beliefs`);
    const q = beliefQuery?.value.trim();
    if (q) {
      url.searchParams.set("query", q);
    }
    const resp = await fetch(url.toString());
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    const items = data.items || [];
    if (!items.length) {
      beliefList.innerHTML = "<li>No results</li>";
      return;
    }
    beliefList.innerHTML = items
      .map(
        (item) =>
          `<li><strong>${escapeHtml(item.text || "—")}</strong><br><span class="muted">${escapeHtml(item.id || "")}</span></li>`
      )
      .join("");
  } catch (err) {
    beliefList.innerHTML = "<li>Load error</li>";
  }
}

pingBtn.addEventListener("click", ping);
sendBtn.addEventListener("click", send);
if (clearBtn) {
  clearBtn.addEventListener("click", () => {
    chat.innerHTML = "";
  });
}
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
if (pulseBtn) {
  pulseBtn.addEventListener("click", refreshSummary);
}
if (learnBtn) {
  learnBtn.addEventListener("click", runLearn);
}
if (beliefSearchBtn) {
  beliefSearchBtn.addEventListener("click", refreshBeliefs);
}
if (beliefQuery) {
  beliefQuery.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      refreshBeliefs();
    }
  });
}

ping();
setInterval(ping, 30_000);
refreshSummary();
refreshBeliefs();
