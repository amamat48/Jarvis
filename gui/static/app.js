const state = {
  mode: "demo",
  conversationId: null,
  epoch: null,
  cursor: 0,
  eventsInitialized: false,
  tasks: [],
  selected: null,
  lastStates: new Map(),
  toolEvents: new Map(),
  mathNotebookId: null,
};

const $ = (id) => document.getElementById(id);
const terminal = new Set(["completed", "failed", "cancelled"]);
const busy = new Set(["queued", "running", "waiting_for_user_input", "waiting_for_approval", "paused"]);

async function api(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || data.error || "Request failed");
  return data;
}

function toast(text) {
  const node = $("toast");
  node.textContent = text;
  node.classList.add("show");
  setTimeout(() => node.classList.remove("show"), 3200);
}

function notify(title, text) {
  toast(text);
  if ("Notification" in window && Notification.permission === "granted") {
    new Notification(title, { body: text });
  }
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[character]));
}

function setMode(mode) {
  state.mode = mode;
  $("mode").value = mode;
  $("mode-badge").textContent = mode.toUpperCase();
  $("mode-badge").className = `badge${mode === "real" ? " real" : ""}`;
  state.conversationId = null;
  state.epoch = null;
  state.cursor = 0;
  state.eventsInitialized = false;
  state.tasks = [];
  state.selected = null;
  state.lastStates.clear();
  state.toolEvents.clear();
  $("chat").innerHTML = "";
  initialize();
}

async function initialize() {
  try {
    const conversation = await api(`/api/conversation?mode=${state.mode}`);
    state.conversationId = conversation.conversation_id;
    await refresh(true);
  } catch (error) {
    toast(error.message);
  }
}

function addChat(role, text, meta = "") {
  const chat = $("chat");
  const welcome = chat.querySelector(".welcome");
  if (welcome) welcome.remove();
  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.innerHTML = (meta ? `<div class="message-meta">${escapeHtml(meta)}</div>` : "") + escapeHtml(text);
  chat.appendChild(item);
  chat.scrollTop = chat.scrollHeight;
}

function paintChat() {
  const chat = $("chat");
  chat.innerHTML = "";
  const items = state.tasks
    .filter((task) => task.conversation_id === state.conversationId)
    .sort((a, b) => a.created_at.localeCompare(b.created_at));
  if (!items.length) {
    chat.innerHTML = '<div class="welcome"><div class="orb">J</div><h2>How can I help?</h2><p>Ask a question or start a task. Demo mode runs a deterministic workflow without loading a model.</p></div>';
    return;
  }
  for (const task of items) {
    addChat("user", task.objective, "You");
    let reply = task.result || (task.state === "failed" ? task.error : "Working on this task…");
    if (task.state === "waiting_for_approval") {
      reply = "This task is waiting for your approval. Review the request in the task panel.";
    } else if (task.state === "waiting_for_user_input") {
      reply = task.pending_question?.prompt || "This task is waiting for your input.";
    }
    addChat("jarvis", reply, task.state.replaceAll("_", " ").toUpperCase());
  }
}

function render() {
  const tasks = [...state.tasks].sort((a, b) => b.created_at.localeCompare(a.created_at));
  $("task-count").textContent = tasks.filter((task) => busy.has(task.state)).length;
  const list = $("task-list");
  list.innerHTML = "";
  for (const task of tasks) {
    const previous = state.lastStates.get(task.task_id);
    if (terminal.has(task.state) && previous && !terminal.has(previous)) {
      notify("JARVIS task finished", `${task.title}: ${task.state}`);
    }
    state.lastStates.set(task.task_id, task.state);

    const button = document.createElement("button");
    button.className = `task-card${task.task_id === state.selected ? " selected" : ""}`;
    button.innerHTML = `<strong>${escapeHtml(task.title)}</strong><small>${escapeHtml(task.current_activity)} · ${escapeHtml(task.state.replaceAll("_", " "))}</small><div class="mini-track"><i style="width:${task.progress}%"></i></div>`;
    button.onclick = () => selectTask(task.task_id);
    list.appendChild(button);
  }

  const selected = tasks.find((task) => task.task_id === state.selected);
  $("detail-empty").classList.toggle("hidden", Boolean(selected));
  $("detail-content").classList.toggle("hidden", !selected);
  if (selected) renderTaskDetails(selected);
  paintChat();
}

function renderTaskDetails(task) {
  $("detail-title").textContent = task.title;
  $("detail-state").textContent = task.state.replaceAll("_", " ");
  $("detail-activity").textContent = task.current_activity;
  $("detail-percent").textContent = `${task.progress}%`;
  $("detail-progress").style.width = `${task.progress}%`;
  $("pause-task").disabled = task.state !== "running" && task.state !== "queued";
  $("resume-task").disabled = task.state !== "paused";
  $("cancel-task").disabled = terminal.has(task.state);
  $("detail-steps").innerHTML = (task.plan || []).map((step) =>
    `<li class="${escapeHtml(step.state)}">${escapeHtml(step.title)}</li>`
  ).join("") || "<li>Work steps will appear here.</li>";
  $("detail-timeline").innerHTML = (task.timeline || []).slice(-8).reverse().map((entry) =>
    `<div class="timeline-item">${escapeHtml(entry.label)}<small>${escapeHtml(entry.timestamp)}</small></div>`
  ).join("") || "<div class='timeline-item'>No activity yet.</div>";

  const approval = task.pending_approval;
  $("approval-box").classList.toggle("hidden", !approval);
  if (approval) $("approval-summary").textContent = approval.summary;
  const question = task.pending_question;
  $("question-box").classList.toggle("hidden", !question);
  if (question) $("question-prompt").textContent = question.prompt;
  $("result-box").classList.toggle("hidden", !task.result);
  if (task.result) $("result-text").textContent = task.result;
  if ($("activity-dialog").open) renderActivity(task);
}

async function refresh(initial = false) {
  try {
    const snapshot = await api(`/api/snapshot?mode=${state.mode}`);
    state.epoch = snapshot.manager_epoch;
    if (!state.eventsInitialized) {
      state.cursor = snapshot.last_sequence;
      state.eventsInitialized = true;
    }
    state.tasks = snapshot.tasks || [];
    if (!state.selected) {
      const focused = state.tasks.find((task) => task.task_id === snapshot.focused_task_id);
      const active = focused || state.tasks.find((task) => busy.has(task.state));
      if (active) state.selected = active.task_id;
    }
    render();
    if (initial) return;

    const feed = await api(`/api/events?mode=${state.mode}&after=${state.cursor}&epoch=${encodeURIComponent(state.epoch)}`);
    if (feed.resync_required) {
      state.cursor = feed.current_sequence;
    } else {
      for (const event of feed.events || []) {
        state.cursor = Math.max(state.cursor, event.sequence || 0);
        if (event.event_type === "tool_started" || event.event_type === "tool_finished") {
          const name = event.payload.tool_name || "a tool";
          const history = state.toolEvents.get(event.task_id) || [];
          history.unshift(`${event.event_type === "tool_started" ? "Started" : "Completed"} ${name}`);
          state.toolEvents.set(event.task_id, history.slice(0, 8));
          if (event.event_type === "tool_started") toast(`Running ${name}`);
        }
      }
    }
  } catch (error) {
    toast(error.message);
  }
}

function selectTask(taskId) {
  state.selected = taskId;
  api(`/api/tasks/${taskId}/focus`, { method: "POST", body: JSON.stringify({ mode: state.mode }) }).catch(() => {});
  render();
  const task = state.tasks.find((item) => item.task_id === taskId);
  if (task && busy.has(task.state) && task.objective.length > 75) openActivity(task);
}

async function control(action, payload = {}) {
  if (!state.selected) return;
  try {
    const result = await api(`/api/tasks/${state.selected}/${action}`, {
      method: "POST",
      body: JSON.stringify({ mode: state.mode, ...payload }),
    });
    toast(result.message || `${action} requested.`);
    await refresh();
  } catch (error) {
    toast(error.message);
  }
}

function renderActivity(task) {
  $("dialog-title").textContent = task.title;
  const toolActivity = state.toolEvents.get(task.task_id) || [];
  const stateLine = task.state.replaceAll("_", " ").toUpperCase();
  const waiting = task.pending_approval
    ? `<p>Waiting for approval: ${escapeHtml(task.pending_approval.summary)}</p>`
    : task.pending_question
      ? `<p>Waiting for input: ${escapeHtml(task.pending_question.prompt)}</p>`
      : "";
  const outcome = task.state === "failed"
    ? `<p class="error-text">${escapeHtml(task.error || "Task failed.")}</p>`
    : task.result ? `<p>${escapeHtml(task.result)}</p>` : "";
  $("dialog-body").innerHTML = `<p><strong>${escapeHtml(stateLine)}</strong> · ${escapeHtml(task.current_activity)} · ${task.progress}%</p><ol>${(task.plan || []).map((step) => `<li>${escapeHtml(step.state)} — ${escapeHtml(step.title)}</li>`).join("")}</ol>${waiting}<h3>Tool activity</h3><ul>${toolActivity.map((item) => `<li>${escapeHtml(item)}</li>`).join("") || "<li>No tool activity yet.</li>"}</ul>${outcome}<div class="button-row"><button id="dialog-pause" class="secondary">Pause</button><button id="dialog-resume" class="secondary">Resume</button><button id="dialog-cancel" class="danger">Cancel</button></div><p>JARVIS reports observable actions and does not expose private reasoning traces.</p>`;
  $("dialog-pause").disabled = task.state !== "running" && task.state !== "queued";
  $("dialog-resume").disabled = task.state !== "paused";
  $("dialog-cancel").disabled = terminal.has(task.state);
  $("dialog-pause").onclick = () => control("pause");
  $("dialog-resume").onclick = () => control("resume");
  $("dialog-cancel").onclick = () => control("cancel");
}

function openActivity(task) {
  renderActivity(task);
  $("activity-dialog").show();
}

$("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = $("message").value.trim();
  if (!text) return;
  $("message").value = "";
  addChat("user", text, "You");
  try {
    const result = await api("/api/messages", {
      method: "POST",
      body: JSON.stringify({ mode: state.mode, conversation_id: state.conversationId, text }),
    });
    if (result.status_answer) {
      addChat("jarvis", result.message, "JARVIS status");
      return;
    }
    if (!result.accepted) throw new Error(result.message);
    state.lastStates.set(result.task_id, "queued");
    if (result.side_task) {
      state.conversationId = result.conversation_id;
      toast("Started this side question in a separate context; the earlier task continues.");
    } else {
      addChat("jarvis", `I’ve started task ${result.task_id.slice(0, 8)}. You can continue chatting while it runs.`, "JARVIS");
    }
    state.selected = result.task_id;
    await refresh();
    const task = state.tasks.find((item) => item.task_id === result.task_id);
    if (task && /analy[sz]e|debug|research|project|simulation|flight/i.test(text)) openActivity(task);
  } catch (error) {
    addChat("jarvis", error.message, "JARVIS");
  }
});

$("message").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});

$("new-chat").onclick = async () => {
  try {
    const result = await api("/api/conversations", {
      method: "POST", body: JSON.stringify({ mode: state.mode }),
    });
    state.conversationId = result.conversation_id;
    $("conversation-title").textContent = "New conversation";
    paintChat();
  } catch (error) {
    toast(error.message);
  }
};

$("mode").onchange = (event) => setMode(event.target.value);
$("pause-task").onclick = () => control("pause");
$("resume-task").onclick = () => control("resume");
$("cancel-task").onclick = () => control("cancel");
$("approve-task").onclick = () => control("approve");
$("reject-task").onclick = () => control("reject");
$("answer-form").onsubmit = (event) => {
  event.preventDefault();
  const task = state.tasks.find((item) => item.task_id === state.selected);
  if (task?.pending_question) {
    control("answer", {
      question_id: task.pending_question.question_id,
      answer: $("answer-input").value,
    });
  }
  $("answer-input").value = "";
};
$("status-task").onclick = async () => {
  try {
    const result = await api(`/api/status?mode=${state.mode}&task_id=${state.selected}`);
    addChat("jarvis", result.status, "JARVIS status");
  } catch (error) {
    toast(error.message);
  }
};
$("close-detail").onclick = () => $("detail-panel").classList.toggle("mobile-open");
$("dialog-close").onclick = () => $("activity-dialog").close();

if ("Notification" in window && Notification.permission === "default") {
  Notification.requestPermission().catch(() => {});
}
initialize();
setInterval(() => refresh(), 600);

/* MathNotebook UI */
const MN = {
  notebookId: null,
  cells: [],

  async new() {
    try {
      const result = await api("/api/math-notebooks", { method: "POST", body: JSON.stringify({ mode: state.mode }) });
      MN.notebookId = result.notebook_id;
      MN.cells = [];
      MN.render();
      MN.updateNotebookSelect();
      toast("Created new notebook: " + MN.notebookId);
    } catch (error) {
      toast(error.message);
    }
  },

  async load(notebookId) {
    try {
      const result = await api(`/api/math-notebooks/${notebookId}/cells`, { method: "GET", body: JSON.stringify({ mode: state.mode }) });
      MN.notebookId = result.notebook_id;
      MN.cells = result.cells || [];
      MN.render();
      MN.updateNotebookSelect();
      toast("Loaded notebook: " + MN.notebookId);
    } catch (error) {
      toast(error.message);
    }
  },

  updateNotebookSelect() {
    const select = $("mn-notebook-select");
    // Keep the "New Notebook" option
    while (select.options.length > 1) select.remove(1);
    if (MN.notebookId) {
      const opt = document.createElement("option");
      opt.value = MN.notebookId;
      opt.textContent = MN.notebookId;
      opt.selected = true;
      select.appendChild(opt);
    }
  },

  async addCell() {
    if (!MN.notebookId) {
      await MN.new();
    }
    const source = $("mn-cell-source").value;
    const cellType = $("mn-cell-type").value;
    if (!source.trim() && cellType === "code") return;
    try {
      const result = await api(`/api/math-notebooks/${MN.notebookId}/cells`, {
        method: "POST",
        body: JSON.stringify({ mode: state.mode, cell_type: cellType, source, index: null }),
      });
      toast("Added cell: " + result.cell_id);
      $("mn-cell-source").value = "";
      await MN.refresh();
    } catch (error) {
      toast(error.message);
    }
  },

  async refresh() {
    if (!MN.notebookId) return;
    try {
      const result = await api(`/api/math-notebooks/${MN.notebookId}/cells`, { method: "GET", body: JSON.stringify({ mode: state.mode }) });
      MN.cells = result.cells || [];
      MN.render();
    } catch (error) {
      toast(error.message);
    }
  },

  async evalCell(cellId) {
    if (!MN.notebookId) return;
    try {
      const result = await api(`/api/math-notebooks/${MN.notebookId}/eval`, {
        method: "POST",
        body: JSON.stringify({ mode: state.mode, cell_id: cellId }),
      });
      await MN.refresh();
      toast("Evaluated: " + result.cell_id);
    } catch (error) {
      toast(error.message);
    }
  },

  async evalAll() {
    if (!MN.notebookId) return;
    try {
      const result = await api(`/api/math-notebooks/${MN.notebookId}/eval-all`, {
        method: "POST",
        body: JSON.stringify({ mode: state.mode }),
      });
      await MN.refresh();
      toast("Evaluated all cells");
    } catch (error) {
      toast(error.message);
    }
  },

  async save() {
    if (!MN.notebookId) return;
    const path = prompt("Enter path to save (e.g., notebook.mathnb):");
    if (!path) return;
    try {
      const result = await api(`/api/math-notebooks/${MN.notebookId}/save`, {
        method: "POST",
        body: JSON.stringify({ mode: state.mode, path }),
      });
      toast("Saved to " + result.path);
    } catch (error) {
      toast(error.message);
    }
  },

  async loadFromFile() {
    const notebookId = prompt("Enter notebook ID to load into:");
    const path = prompt("Enter path to load from (e.g., notebook.mathnb):");
    if (!notebookId || !path) return;
    try {
      const result = await api(`/api/math-notebooks/${notebookId}/load`, {
        method: "POST",
        body: JSON.stringify({ mode: state.mode, path }),
      });
      MN.notebookId = result.notebook_id;
      await MN.refresh();
      MN.updateNotebookSelect();
      toast("Loaded from " + path);
    } catch (error) {
      toast(error.message);
    }
  },

  async deleteCell(cellId) {
    if (!MN.notebookId) return;
    try {
      await api(`/api/math-notebooks/${MN.notebookId}/cells/${cellId}`, {
        method: "DELETE",
        body: JSON.stringify({ mode: state.mode }),
      });
      await MN.refresh();
      toast("Deleted cell");
    } catch (error) {
      toast(error.message);
    }
  },

  render() {
    const container = $("mn-cells");
    if (!MN.notebookId) {
      container.innerHTML = '<div class="empty-state"><div class="empty-icon">📓</div><p>Create or load a notebook to begin.</p></div>';
      return;
    }
    if (!MN.cells.length) {
      container.innerHTML = '<div class="empty-state"><div class="empty-icon">📄</div><p>No cells yet. Add a code or markdown cell below.</p></div>';
      return;
    }
    container.innerHTML = MN.cells.map((cell) => `
      <div class="mn-cell" data-cell-id="${escapeHtml(cell.id)}">
        <div class="mn-cell-header">
          <span class="mn-cell-id">${escapeHtml(cell.id)}</span>
          <span class="mn-cell-type">${escapeHtml(cell.cell_type)}</span>
          <span class="mn-cell-status ${cell.status === "error" ? "error" : ""} ${cell.status === "running" ? "running" : ""}">${escapeHtml(cell.status)}</span>
          <span class="mn-cell-exec">${cell.execution_count != null ? "#" + cell.execution_count : ""}</span>
        </div>
        <pre class="mn-cell-source">${escapeHtml(cell.source)}</pre>
        ${cell.outputs && cell.outputs.length > 0 ? `
        <div class="mn-cell-outputs">
          ${cell.outputs.map((out) => `
            <div class="mn-output ${out.output_type === "error" ? "error" : ""}">${escapeHtml(out.data["text/plain"] || "")}</div>
          `).join("")}
        </div>
        ` : ""}
        <div class="mn-cell-actions" style="padding:8px 12px;border-top:1px solid var(--line);display:flex;gap:6px;">
          ${cell.cell_type === "code" && cell.status !== "running" ? `<button class="secondary mn-eval-btn" data-cell-id="${escapeHtml(cell.id)}" style="font-size:11px;padding:4px 8px;">▶ Run</button>` : ""}
          <button class="secondary mn-del-btn" data-cell-id="${escapeHtml(cell.id)}" style="font-size:11px;padding:4px 8px;">Delete</button>
        </div>
      </div>
    `).join("");

    // Attach event listeners
    container.querySelectorAll(".mn-eval-btn").forEach((btn) => {
      btn.onclick = () => MN.evalCell(btn.dataset.cellId);
    });
    container.querySelectorAll(".mn-del-btn").forEach((btn) => {
      btn.onclick = () => MN.deleteCell(btn.dataset.cellId);
    });
  },
};

/* Tab switching */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tab").forEach((t) => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    const panelId = "panel-" + tab.dataset.tab;
    $(panelId).classList.add("active");
    if (tab.dataset.tab === "mathnotebook") MN.render();
  };
});

/* MathNotebook event handlers */
$("mn-new").onclick = () => MN.new();
$("mn-save").onclick = () => MN.save();
$("mn-load").onclick = () => MN.loadFromFile();
$("mn-eval-all").onclick = () => MN.evalAll();
$("mn-add-cell").onclick = () => MN.addCell();
$("mn-notebook-select").onchange = (event) => {
  const notebookId = event.target.value;
  if (notebookId) MN.load(notebookId);
};
