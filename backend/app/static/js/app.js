// Plain fetch()-based client: no build step, no framework. Talks to the FastAPI backend on
// the same origin (so no CORS setup needed) and keeps the JWT in localStorage.
(function () {
  const TOKEN_KEY = "mmrag_token";
  let currentSessionId = null;
  let sessions = [];
  let documents = [];
  let pollTimer = null;

  const $ = (id) => document.getElementById(id);

  function getToken() {
    return localStorage.getItem(TOKEN_KEY);
  }
  function setToken(token) {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  }

  // Thin fetch wrapper: attaches the bearer token, picks a sensible Content-Type, and
  // turns non-2xx responses into thrown Errors with a readable message.
  async function api(path, options = {}) {
    const headers = Object.assign({}, options.headers);
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
    // Leave Content-Type unset for URLSearchParams (login) and FormData (file upload) —
    // the browser sets the correct header (incl. multipart boundary) for those itself.
    if (
      options.body &&
      !(options.body instanceof URLSearchParams) &&
      !(options.body instanceof FormData) &&
      !headers["Content-Type"]
    ) {
      headers["Content-Type"] = "application/json";
    }

    const res = await fetch(path, Object.assign({}, options, { headers }));

    if (res.status === 401) {
      setToken(null);
      showAuthView();
      throw new Error("Session expired — please log in again.");
    }
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const data = await res.json();
        if (data.detail) {
          detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
        }
      } catch (_) {
        /* response had no JSON body */
      }
      throw new Error(detail);
    }
    if (res.status === 204) return null;
    return res.json();
  }

  // ---------- View toggling ----------
  function showAuthView() {
    $("auth-view").classList.remove("hidden");
    $("chat-view").classList.add("hidden");
  }
  function showChatView() {
    $("auth-view").classList.add("hidden");
    $("chat-view").classList.remove("hidden");
  }

  // ---------- Auth ----------
  function switchTab(which) {
    const isLogin = which === "login";
    $("tab-login").classList.toggle("active", isLogin);
    $("tab-register").classList.toggle("active", !isLogin);
    $("login-form").classList.toggle("hidden", !isLogin);
    $("register-form").classList.toggle("hidden", isLogin);
  }
  $("tab-login").addEventListener("click", () => switchTab("login"));
  $("tab-register").addEventListener("click", () => switchTab("register"));

  // fastapi-users' JWT login route is an OAuth2 password flow: form-encoded
  // `username` (the email) + `password`, returning {access_token, token_type}.
  async function passwordLogin(email, password) {
    const body = new URLSearchParams();
    body.set("username", email);
    body.set("password", password);
    const data = await api("/api/auth/jwt/login", { method: "POST", body });
    setToken(data.access_token);
  }

  $("login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("login-error").textContent = "";
    try {
      await passwordLogin($("login-email").value, $("login-password").value);
      await enterApp();
    } catch (err) {
      $("login-error").textContent = err.message;
    }
  });

  $("register-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("register-error").textContent = "";
    const email = $("register-email").value;
    const password = $("register-password").value;
    try {
      await api("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      await passwordLogin(email, password);
      await enterApp();
    } catch (err) {
      $("register-error").textContent = err.message;
    }
  });

  $("logout-btn").addEventListener("click", () => {
    setToken(null);
    currentSessionId = null;
    sessions = [];
    documents = [];
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    $("session-list").innerHTML = "";
    $("message-list").innerHTML = "";
    $("document-list").innerHTML = "";
    showAuthView();
  });

  // ---------- Sessions ----------
  async function loadSessions() {
    sessions = await api("/api/sessions");
    if (!sessions.length) {
      await createSession();
      return;
    }
    renderSessionList();
    await selectSession(sessions[0].id);
  }

  function renderSessionList() {
    const list = $("session-list");
    list.innerHTML = "";
    sessions.forEach((s) => {
      const li = document.createElement("li");
      li.textContent = s.title || "New conversation";
      li.className = "session-item" + (s.id === currentSessionId ? " active" : "");
      li.addEventListener("click", () => selectSession(s.id));
      list.appendChild(li);
    });
  }

  $("new-chat-btn").addEventListener("click", () => createSession());

  async function createSession() {
    const session = await api("/api/sessions", { method: "POST" });
    sessions.unshift(session);
    await selectSession(session.id);
  }

  async function selectSession(id) {
    currentSessionId = id;
    renderSessionList();
    const messages = await api(`/api/sessions/${id}/messages`);
    renderMessages(messages);
  }

  // ---------- Documents ----------
  const STATUS_LABELS = {
    queued: "Queued",
    processing: "Processing…",
    ready: "Ready",
    failed: "Failed",
  };

  function renderDocumentList() {
    const list = $("document-list");
    list.innerHTML = "";
    documents.forEach((doc) => {
      const li = document.createElement("li");
      li.className = "document-item";

      const name = document.createElement("span");
      name.className = "document-name";
      name.textContent = doc.filename;
      name.title = doc.filename;

      const badge = document.createElement("span");
      badge.className = `status-badge status-${doc.status}`;
      badge.textContent = STATUS_LABELS[doc.status] || doc.status;
      if (doc.status === "failed" && doc.error_message) badge.title = doc.error_message;

      const actions = document.createElement("span");
      actions.className = "doc-actions";

      if (doc.status === "failed") {
        const retryBtn = document.createElement("button");
        retryBtn.className = "doc-retry-btn";
        retryBtn.title = "Retry processing";
        retryBtn.textContent = "↺";
        retryBtn.addEventListener("click", async () => {
          try {
            const updated = await api(`/api/documents/${doc.id}/reprocess`, { method: "PATCH" });
            const idx = documents.findIndex((d) => d.id === doc.id);
            if (idx !== -1) documents[idx] = updated;
            renderDocumentList();
            await loadDocuments();
          } catch (err) {
            $("upload-error").textContent = err.message;
          }
        });
        actions.appendChild(retryBtn);
      }

      const delBtn = document.createElement("button");
      delBtn.className = "doc-delete-btn";
      delBtn.title = "Delete document";
      delBtn.textContent = "×";
      delBtn.addEventListener("click", async () => {
        try {
          await api(`/api/documents/${doc.id}`, { method: "DELETE" });
          documents = documents.filter((d) => d.id !== doc.id);
          renderDocumentList();
          if (!documents.some((d) => d.status === "queued" || d.status === "processing") && pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
          }
        } catch (err) {
          $("upload-error").textContent = err.message;
        }
      });
      actions.appendChild(delBtn);

      li.appendChild(name);
      li.appendChild(badge);
      li.appendChild(actions);
      list.appendChild(li);
    });
  }

  async function loadDocuments() {
    documents = await api("/api/documents");
    renderDocumentList();

    // While anything is still queued/processing, poll for status updates so the badge
    // flips to "Ready" without the user having to refresh the page.
    const stillWorking = documents.some((d) => d.status === "queued" || d.status === "processing");
    if (stillWorking && !pollTimer) {
      pollTimer = setInterval(async () => {
        documents = await api("/api/documents");
        renderDocumentList();
        if (!documents.some((d) => d.status === "queued" || d.status === "processing")) {
          clearInterval(pollTimer);
          pollTimer = null;
        }
      }, 4000);
    }
  }

  $("document-upload").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    $("upload-error").textContent = "";

    const body = new FormData();
    body.append("file", file);
    try {
      const doc = await api("/api/documents", { method: "POST", body });
      documents.unshift(doc);
      renderDocumentList();
      await loadDocuments(); // kicks off polling if needed
    } catch (err) {
      $("upload-error").textContent = err.message;
    } finally {
      e.target.value = "";
    }
  });

  // ---------- Messages ----------
  function renderMessages(messages) {
    const list = $("message-list");
    list.innerHTML = "";
    messages.forEach((m) => appendMessage(m.role, m.content, m.citations));
    scrollToBottom();
  }

  function appendMessage(role, content, citations) {
    const el = document.createElement("div");
    el.className = `message message-${role}`;

    const body = document.createElement("div");
    body.className = "message-content";
    if (role === "assistant") {
      body.classList.add("markdown-body");
      body.innerHTML = marked.parse(content);
    } else {
      body.textContent = content;
    }
    el.appendChild(body);

    if (citations && citations.length) {
      const chips = document.createElement("div");
      chips.className = "citation-chips";
      citations.forEach((c) => {
        const chip = document.createElement("span");
        chip.className = "citation-chip";
        const typeLabel = c.chunk_type === "vision" ? "Figure" : c.chunk_type === "table" ? "Table" : null;
        chip.textContent = typeLabel ? `${typeLabel} — ${c.filename} p. ${c.page_number}` : `${c.filename} — p. ${c.page_number}`;
        chips.appendChild(chip);
      });
      el.appendChild(chips);
    }

    $("message-list").appendChild(el);
    // Animate in on the next frame so the CSS transition actually fires.
    requestAnimationFrame(() => el.classList.add("message-visible"));
    return el;
  }

  // Replaces a placeholder message's text and (optionally) renders citation chips —
  // used once the real assistant reply comes back from /api/chat.
  function fillMessage(el, content, citations) {
    const bodyEl = el.querySelector(".message-content");
    bodyEl.classList.add("markdown-body");
    bodyEl.innerHTML = marked.parse(content);
    if (citations && citations.length) {
      const chips = document.createElement("div");
      chips.className = "citation-chips";
      citations.forEach((c) => {
        const chip = document.createElement("span");
        chip.className = "citation-chip";
        const typeLabel = c.chunk_type === "vision" ? "Figure" : c.chunk_type === "table" ? "Table" : null;
        chip.textContent = typeLabel ? `${typeLabel} — ${c.filename} p. ${c.page_number}` : `${c.filename} — p. ${c.page_number}`;
        chips.appendChild(chip);
      });
      el.appendChild(chips);
    }
  }

  function scrollToBottom() {
    const list = $("message-list");
    list.scrollTo({ top: list.scrollHeight, behavior: "smooth" });
  }

  $("message-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("message-input");
    const content = input.value.trim();
    if (!content || !currentSessionId) return;

    input.value = "";
    input.disabled = true;
    appendMessage("user", content);
    const pending = appendMessage("assistant", "Thinking…");
    pending.classList.add("message-pending");
    scrollToBottom();

    // Auto-title the session from the first message the user sends
    const sess = sessions.find((s) => s.id === currentSessionId);
    if (sess && (!sess.title || sess.title === "New conversation")) {
      sess.title = content.length > 45 ? content.substring(0, 45) + "…" : content;
      renderSessionList();
    }

    try {
      const data = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({ session_id: currentSessionId, content }),
      });
      fillMessage(pending, data.assistant_message.content, data.assistant_message.citations);
      pending.classList.remove("message-pending");
    } catch (err) {
      fillMessage(pending, `Error: ${err.message}`);
      pending.classList.remove("message-pending");
      pending.classList.add("message-error");
    } finally {
      scrollToBottom();
      input.disabled = false;
      input.focus();
    }
  });

  // ---------- Bootstrap ----------
  async function enterApp() {
    showChatView();
    currentSessionId = null;
    await Promise.all([loadSessions(), loadDocuments()]);
  }

  if (getToken()) {
    enterApp().catch(() => showAuthView());
  } else {
    showAuthView();
  }
})();
