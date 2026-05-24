// Skein content script — shared core
//
// All cross-site behavior lives here: the floating badge, the transient
// toast, recall plumbing, relevance-marker parsing, the submit-intercept
// state machine, and bootstrap. Each site-specific content script
// (content_claude.js, content_chatgpt.js, content_gemini.js) declares
// only its DOM selectors and calls __SkeinCommon.init(siteAdapter).
//
// Why a shared module: iter 32's relevance-marker patch touched the
// recall response handling. Without this split, iter 33's port to
// chatgpt.com + gemini.google.com would require copy-pasting the same
// fix into three files. With the split, every future hot-path change
// lands in one place and three sites inherit it.
//
// Content scripts listed under the same manifest entry share an
// isolated world per tab, so this file just attaches __SkeinCommon to
// globalThis and the per-site files reference it on script-load.

(() => {
  const LOG = (...args) => console.info("[skein]", ...args);
  const WARN = (...args) => console.warn("[skein]", ...args);

  // ---- runtime state cached from background ---------------------------

  let cached = { enabled: true, activeScope: null, daemonUrl: null, bearerToken: null };

  async function refreshState() {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: "getState" }, (r) => {
        if (r && r.ok) cached = r.state;
        resolve(cached);
      });
    });
  }

  // ---- floating badge -------------------------------------------------

  let badge;
  function ensureBadge() {
    if (badge) return badge;
    badge = document.createElement("div");
    badge.className = "skein-badge";
    badge.innerHTML = `
      <span class="skein-dot"></span>
      <span class="skein-label">Skein</span>
      <span class="skein-detail">…</span>
    `;
    badge.title = "Click to open the Skein extension popup";
    badge.addEventListener("click", () => {
      // Can't open the popup programmatically (Chrome MV3 limitation);
      // just surface a hint in console and flash the badge.
      LOG("click the toolbar icon to open settings");
      badge.classList.add("skein-flash");
      setTimeout(() => badge.classList.remove("skein-flash"), 400);
    });
    document.documentElement.appendChild(badge);
    return badge;
  }

  function setBadge({ kind = "ok", label = "Skein", detail = "" } = {}) {
    const b = ensureBadge();
    b.classList.remove("skein-ok", "skein-warn", "skein-err");
    b.classList.add(`skein-${kind}`);
    b.querySelector(".skein-label").textContent = label;
    b.querySelector(".skein-detail").textContent = detail;
  }

  // ---- transient toast above the prompt -------------------------------

  function flashToast(msg, ms = 1500) {
    const t = document.createElement("div");
    t.className = "skein-toast";
    t.textContent = msg;
    document.documentElement.appendChild(t);
    setTimeout(() => t.remove(), ms);
  }

  // ---- text manipulation on a contenteditable (default impl) ---------

  // Works for ProseMirror, Quill, Lexical, and most other rich-text
  // editors that listen to InputEvent. Site adapters can override this
  // via siteAdapter.setPromptText when the editor has stricter input
  // semantics.
  function defaultGetPromptText(el) {
    return el ? el.innerText || el.textContent || "" : "";
  }

  function defaultSetPromptText(el, text) {
    el.focus();
    document.execCommand("selectAll", false, null);
    document.execCommand("delete", false, null);
    document.execCommand("insertText", false, text);
  }

  // ---- recall plumbing -----------------------------------------------

  function callRecall(query) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { type: "recall", query, limit: 5 },
        (r) => resolve(r),
      );
    });
  }

  // Iter 32 token-waste fix: the daemon prepends a machine-readable
  // marker on line 1: `[skein:relevance=high|medium|low|none]`. Parse it
  // and refuse to inject when relevance is `low` or `none`. Strip the
  // marker line either way so the LLM never sees it. Older daemons
  // without the marker fall through as "medium" for backwards compat.
  function parseRelevanceAndStrip(rawText) {
    const text = (rawText || "").trim();
    const firstNewline = text.indexOf("\n");
    const firstLine = firstNewline >= 0 ? text.slice(0, firstNewline) : text;
    const m = /^\[skein:relevance=(high|medium|low|none)\]\s*$/.exec(firstLine.trim());
    if (!m) {
      return { relevance: "medium", body: text };
    }
    const body = firstNewline >= 0 ? text.slice(firstNewline + 1).trimStart() : "";
    return { relevance: m[1], body };
  }

  function formatContextBlock(recallResult, originalQuery) {
    if (!recallResult || !recallResult.content || !recallResult.content[0]) {
      return { block: null, relevance: "none" };
    }
    const text = recallResult.content[0].text;
    if (!text || !text.trim()) return { block: null, relevance: "none" };
    const { relevance, body } = parseRelevanceAndStrip(text);
    if (relevance === "low" || relevance === "none") {
      LOG("relevance=" + relevance + " — skipping injection (saves tokens)");
      return { block: null, relevance };
    }
    if (!body || !body.trim()) return { block: null, relevance };
    if (/^no relevant context found/i.test(body.trim())) return { block: null, relevance: "none" };
    if (/^no fragment in skein matches/i.test(body.trim())) return { block: null, relevance: "none" };
    return {
      relevance,
      block: [
        `[Skein context — auto-injected by browser extension for query: "${originalQuery.slice(0, 80)}", relevance=${relevance}]`,
        body.trim(),
        `[/Skein context]`,
        "",
        "",
      ].join("\n"),
    };
  }

  // ---- submit interception (the work) ---------------------------------

  function makeInterceptor(siteAdapter) {
    const findPromptElement = siteAdapter.findPromptElement;
    const findSendButton = siteAdapter.findSendButton;
    const getPromptText = siteAdapter.getPromptText || defaultGetPromptText;
    const setPromptText = siteAdapter.setPromptText || defaultSetPromptText;

    let interceptingNow = false;

    function retriggerSubmit(promptEl) {
      const send = findSendButton();
      if (send && !send.disabled) {
        send.click();
        return;
      }
      // Fallback: synthetic Enter keydown on the prompt.
      const ev = new KeyboardEvent("keydown", {
        key: "Enter", code: "Enter", keyCode: 13, which: 13,
        bubbles: true, cancelable: true,
      });
      promptEl.dispatchEvent(ev);
    }

    async function handleSubmitAttempt(originalEvent, source) {
      if (interceptingNow) return; // re-entry guard
      if (!cached.enabled) {
        LOG("disabled — passing through");
        return;
      }
      if (!cached.activeScope) {
        WARN("no active scope set in popup; passing through");
        flashToast("Skein: open the toolbar icon and pick a scope");
        return;
      }
      const promptEl = findPromptElement();
      if (!promptEl) {
        WARN("prompt element gone; passing through");
        return;
      }
      const original = getPromptText(promptEl).trim();
      if (!original || original.length < 4) {
        LOG("prompt too short; passing through");
        return;
      }
      // Already has a Skein block (e.g. user re-submitted after we
      // injected); don't double-inject.
      if (original.startsWith("[Skein context")) {
        LOG("already has Skein block; passing through");
        return;
      }

      LOG(`intercept (${source}) for query:`, original.slice(0, 80));
      originalEvent.preventDefault();
      originalEvent.stopImmediatePropagation();
      interceptingNow = true;
      flashToast("Skein → recalling context…");

      try {
        const r = await callRecall(original);
        if (!r || !r.ok) {
          WARN("recall failed; passing through:", r && r.error);
          flashToast(`Skein: recall failed (${r && r.error || "unknown"})`);
          retriggerSubmit(promptEl);
          return;
        }
        const { block, relevance } = formatContextBlock(r.result, original);
        if (!block) {
          LOG(`relevance=${relevance} — passing through without injection`);
          if (relevance === "low" || relevance === "none") {
            flashToast(`Skein: no relevant context (saved tokens)`);
            setBadge({
              kind: "warn",
              detail: `${cached.activeScope || ""} · no relevant match`,
            });
          } else {
            flashToast("Skein: no relevant context found");
          }
          retriggerSubmit(promptEl);
          return;
        }
        LOG("injecting", block.length, "chars of context, relevance=" + relevance);
        setPromptText(promptEl, block + original);
        flashToast(`Skein → injected (relevance=${relevance})`);
        // Wait a tick for the editor to re-render, then submit.
        setTimeout(() => retriggerSubmit(promptEl), 80);
      } catch (err) {
        WARN("error during intercept:", err);
        flashToast(`Skein error: ${err.message || err}`);
        retriggerSubmit(promptEl);
      } finally {
        // Release the re-entry guard a moment after the resubmit so it
        // doesn't catch our own simulated event.
        setTimeout(() => { interceptingNow = false; }, 300);
      }
    }

    return { handleSubmitAttempt };
  }

  // ---- Save to Skein button on assistant turns (iter 35) -------------
  //
  // Per-turn "Save to Skein" button. Default behavior: extract the turn's
  // plain text (innerText), pass to background → MCP note() → daemon
  // classifies type/tags/value automatically. Same shape on all three
  // sites — only the assistant-turn selector differs per adapter.
  //
  // A WeakSet tracks elements we've already decorated so the
  // MutationObserver doesn't double-inject when React re-renders.

  const decorated = new WeakSet();
  const saved = new WeakSet();

  function defaultGetAssistantText(el) {
    return el ? (el.innerText || el.textContent || "").trim() : "";
  }

  function makeSaveButton(turnEl, getAssistantText) {
    const btn = document.createElement("button");
    btn.className = "skein-save-btn";
    btn.type = "button";
    btn.textContent = "Save to Skein";
    btn.title = "Save this assistant turn as a Skein note (auto-classified)";
    btn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      if (saved.has(turnEl) || btn.disabled) return;
      if (!cached.activeScope) {
        flashToast("Skein: pick a scope in the toolbar popup first");
        return;
      }
      const text = getAssistantText(turnEl);
      if (!text || text.length < 8) {
        flashToast("Skein: nothing to save (turn is empty)");
        return;
      }
      btn.disabled = true;
      btn.textContent = "Saving…";
      try {
        const r = await new Promise((resolve) => {
          chrome.runtime.sendMessage({ type: "note", content: text }, resolve);
        });
        if (!r || !r.ok) {
          btn.textContent = "Save failed";
          btn.classList.add("skein-save-err");
          WARN("note failed:", r && r.error);
          flashToast(`Skein: save failed (${r && r.error || "unknown"})`);
          setTimeout(() => {
            btn.disabled = false;
            btn.classList.remove("skein-save-err");
            btn.textContent = "Save to Skein";
          }, 2500);
          return;
        }
        saved.add(turnEl);
        btn.textContent = "✓ Saved";
        btn.classList.add("skein-save-ok");
        flashToast("Skein → noted");
      } catch (err) {
        WARN("save error:", err);
        btn.textContent = "Save failed";
        btn.classList.add("skein-save-err");
        setTimeout(() => {
          btn.disabled = false;
          btn.classList.remove("skein-save-err");
          btn.textContent = "Save to Skein";
        }, 2500);
      }
    }, true);
    return btn;
  }

  function decorateAssistantTurns(siteAdapter) {
    const findTurns = siteAdapter.findAssistantTurns;
    if (typeof findTurns !== "function") return;
    const getAssistantText = siteAdapter.getAssistantTurnText || defaultGetAssistantText;
    const turns = findTurns() || [];
    for (const turn of turns) {
      if (!turn || decorated.has(turn)) continue;
      const text = getAssistantText(turn);
      if (!text || text.length < 8) continue; // streaming or empty turn
      decorated.add(turn);
      const btn = makeSaveButton(turn, getAssistantText);
      // Position button inside the turn, top-right. Site CSS shouldn't
      // need to know about us — the button is absolute-positioned and the
      // turn becomes relative via the .skein-host class.
      turn.classList.add("skein-host");
      turn.appendChild(btn);
    }
  }

  function installSaveButtons(siteAdapter) {
    if (typeof siteAdapter.findAssistantTurns !== "function") return;
    // Initial pass.
    setTimeout(() => decorateAssistantTurns(siteAdapter), 1200);
    // Watch for new turns and React re-renders. Throttled to one pass
    // per animation frame; the page may emit dozens of mutations per
    // streaming token.
    let pending = false;
    const obs = new MutationObserver(() => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        try {
          decorateAssistantTurns(siteAdapter);
        } catch (err) {
          WARN("decorate failed:", err);
        }
      });
    });
    obs.observe(document.documentElement, {
      childList: true, subtree: true,
    });
  }

  // ---- bootstrap ------------------------------------------------------

  async function bootstrap(siteAdapter) {
    setBadge({ kind: "warn", detail: "initialising" });
    await refreshState();

    if (!cached.bearerToken) {
      LOG("no token yet; requesting pair");
      await new Promise((resolve) => chrome.runtime.sendMessage({ type: "pair" }, resolve));
      await refreshState();
    }

    chrome.runtime.sendMessage({ type: "projectBriefing" }, (r) => {
      if (r && r.ok) {
        const txt = (r.result && r.result.content && r.result.content[0] && r.result.content[0].text) || "";
        const m = txt.match(/(\d+)\s+fragments?/i);
        const count = m ? `${m[1]} fragments` : "ready";
        setBadge({ kind: "ok", detail: `${cached.activeScope || "no scope"} · ${count}` });
      } else if (!cached.activeScope) {
        setBadge({ kind: "warn", detail: "pick a scope ▸ toolbar" });
      } else {
        setBadge({ kind: "err", detail: `daemon unreachable — ${r && r.error || "?"}` });
      }
    });
  }

  // ---- public entry point --------------------------------------------

  function init(siteAdapter) {
    if (!siteAdapter || typeof siteAdapter.findPromptElement !== "function" ||
        typeof siteAdapter.findSendButton !== "function") {
      WARN("init() called without a valid siteAdapter — abort");
      return;
    }

    const { handleSubmitAttempt } = makeInterceptor(siteAdapter);

    // Capture-phase so we run BEFORE the page's own React/Lit/Vue
    // handlers.
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" || e.shiftKey || e.metaKey || e.ctrlKey || e.altKey) return;
      const promptEl = siteAdapter.findPromptElement();
      if (!promptEl || !promptEl.contains(e.target)) return;
      handleSubmitAttempt(e, "keydown:Enter");
    }, true);

    document.addEventListener("click", (e) => {
      const btn = e.target.closest && e.target.closest("button");
      if (!btn) return;
      const send = siteAdapter.findSendButton();
      if (btn !== send) return;
      handleSubmitAttempt(e, "click:sendButton");
    }, true);

    // React-heavy SPAs may finish rendering long after document_idle;
    // give them a couple seconds.
    setTimeout(() => bootstrap(siteAdapter), 800);

    // Iter 35: install Save-to-Skein buttons on assistant turns (no-op
    // for adapters that don't expose findAssistantTurns).
    installSaveButtons(siteAdapter);

    chrome.storage.onChanged.addListener((changes) => {
      LOG("state changed:", Object.keys(changes));
      refreshState();
    });

    LOG(`content script loaded for ${siteAdapter.siteName || location.host}`);
  }

  globalThis.__SkeinCommon = { init };
})();
