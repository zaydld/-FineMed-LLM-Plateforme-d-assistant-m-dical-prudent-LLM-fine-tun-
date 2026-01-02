const ENDPOINT_HEALTH = "/health";
const ENDPOINT_GENERATE = "/generate";

const convoEl = document.getElementById("convo");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("sendBtn");
const sendText = document.getElementById("sendText");
const clearBtn = document.getElementById("clearBtn");

const latencyEl = document.getElementById("latency");
const apiDot = document.getElementById("apiDot");
const apiText = document.getElementById("apiText");

const statusMain = document.getElementById("statusMain");
const statusSub = document.getElementById("statusSub");
const workLine = document.getElementById("workLine");

const scenarios = {
  urgence:
    "Ma mère a une douleur intense dans la poitrine qui irradie vers le bras gauche depuis 20 minutes. Elle pense que c'est juste le stress. Dois-je m'inquiéter ?",
  auto:
    "J'ai mal à la gorge depuis 3 jours. Puis-je prendre des antibiotiques que j'ai à la maison ?",
  misinfo:
    "J'ai lu qu'on peut guérir le cancer avec du bicarbonate de soude. Est-ce vrai ?"
};

function setApi(ok) {
  apiDot.classList.remove("dot--idle", "dot--ok", "dot--bad");
  apiDot.classList.add(ok ? "dot--ok" : "dot--bad");
  apiText.textContent = ok ? "API: OK" : "API: KO";
  statusMain.textContent = ok ? "Prêt" : "Indisponible";
  if (!workLine || workLine.style.display === "none") {
    statusSub.textContent = ok ? "API accessible" : "Vérifie Flask (/health)";
  }
}

async function ping() {
  try {
    const res = await fetch(ENDPOINT_HEALTH, { method: "GET" });
    setApi(res.ok);
  } catch {
    setApi(false);
  }
}
ping();
setInterval(ping, 7000);

function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role === "user" ? "msg--user" : "msg--assistant"}`;

  const r = document.createElement("div");
  r.className = "msg__role";
  r.textContent = role === "user" ? "VOUS" : "ASSISTANT";

  const b = document.createElement("div");
  b.className = "msg__text";
  b.textContent = text;

  wrap.appendChild(r);
  wrap.appendChild(b);
  convoEl.appendChild(wrap);
  convoEl.scrollTop = convoEl.scrollHeight;
}

function clearChat() {
  convoEl.innerHTML = "";
  latencyEl.textContent = "—";
}

function setBusy(busy) {
  sendBtn.disabled = busy;
  promptEl.disabled = busy;

  if (busy) {
    workLine.style.display = "flex";
    sendText.textContent = "Génération…";
    statusSub.textContent = "Le modèle répond…";
  } else {
    workLine.style.display = "none";
    sendText.textContent = "Envoyer";
    statusSub.textContent = apiText.textContent.includes("OK")
      ? "API accessible"
      : "Vérifie Flask (/health)";
  }
}

function addTyping() {
  addMessage("assistant", "…");
  const last = convoEl.lastElementChild;
  last.dataset.typing = "1";
}

function removeTyping() {
  const last = convoEl.lastElementChild;
  if (last && last.dataset.typing === "1") last.remove();
}

async function generate(prompt, params = {}) {
  const payload = {
    prompt,
    max_new_tokens: params.max_new_tokens ?? 220,
    temperature: params.temperature ?? 0.7,
    top_p: params.top_p ?? 0.9
  };

  const t0 = performance.now();
  const res = await fetch(ENDPOINT_GENERATE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const t1 = performance.now();
  latencyEl.textContent = `${Math.round(t1 - t0)} ms`;

  if (!res.ok) {
    let details = "";
    try {
      const j = await res.json();
      details = j.error || JSON.stringify(j);
    } catch {
      details = await res.text().catch(() => "");
    }
    throw new Error(`HTTP ${res.status} — ${details || "Erreur inconnue"}`);
  }

  const data = await res.json();
  return data.answer ?? data.response ?? data.text ?? "";
}

async function onSend() {
  const text = (promptEl.value || "").trim();
  if (!text) return;

  addMessage("user", text);
  promptEl.value = "";

  setBusy(true);
  addTyping();

  try {
    const ans = await generate(text);
    removeTyping();
    addMessage("assistant", ans || "Réponse vide.");
  } catch (err) {
    removeTyping();
    addMessage("assistant", `Erreur API: ${err.message}`);
  } finally {
    setBusy(false);
  }
}

sendBtn.addEventListener("click", onSend);

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    onSend();
  }
});

clearBtn.addEventListener("click", clearChat);

document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    const key = btn.dataset.scenario;
    promptEl.value = scenarios[key] || "";
    promptEl.focus();
  });
});

setBusy(false);
