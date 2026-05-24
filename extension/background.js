// Skein browser extension — service worker
//
// Role: the only context in the extension that can fetch from 127.0.0.1.
// Content scripts run under the page's origin (https://claude.ai) and can't
// hit the daemon directly. They send `chrome.runtime.sendMessage` requests
// here; this worker holds the bearer token, talks to the daemon, and ships
// results back to whichever content script asked.
//
// State (in chrome.storage.local):
//   daemonUrl    — base URL of the Skein daemon (default 8766 for the
//                  experiment branch; production extension will default to
//                  8765)
//   bearerToken  — string returned by /v1/pair-browser; null until paired
//   activeScope  — last-used scope handle (sticks across sessions)
//   enabled      — global on/off; defaults true

// Iter 35: extension graduated — defaults to the production daemon (8765).
// Users running the iter-30..34 experiment can change the URL via the
// toolbar popup if they still want to talk to a side daemon on 8766.
const DEFAULT_DAEMON_URL = "http://127.0.0.1:8765";

// ---- storage helpers ---------------------------------------------------

async function getState() {
  const s = await chrome.storage.local.get([
    "daemonUrl", "bearerToken", "activeScope", "enabled",
  ]);
  return {
    daemonUrl: s.daemonUrl || DEFAULT_DAEMON_URL,
    bearerToken: s.bearerToken || null,
    activeScope: s.activeScope || null,
    enabled: s.enabled !== false, // default on
  };
}

async function setState(patch) {
  await chrome.storage.local.set(patch);
}

// ---- pairing -----------------------------------------------------------

// Pair with the local daemon — one call, idempotent. The daemon's
// /v1/pair-browser endpoint returns the same bearer token every time
// (it's just `cfg.bearer_token`), so re-pairing after the extension
// reloads or the user uninstalls + reinstalls just works.
async function pair() {
  const { daemonUrl } = await getState();
  console.info("[skein] pairing with", daemonUrl);
  const r = await fetch(`${daemonUrl}/v1/pair-browser`, {
    method: "POST",
    // The browser auto-fills Origin to `chrome-extension://<our id>` for
    // any extension-initiated fetch; the daemon checks that header.
  });
  if (!r.ok) {
    console.error("[skein] pairing failed", r.status, await r.text());
    throw new Error(`pair-browser returned ${r.status}`);
  }
  const data = await r.json();
  await setState({
    bearerToken: data.bearer_token,
    daemonUrl: data.daemon_url,
  });
  console.info("[skein] paired ✓ daemon=", data.daemon_url);
  return data;
}

// Ensure we have a token. Re-pair if not, or if the existing token has
// gone stale (daemon rebuilt with a new token between sessions).
async function ensurePaired() {
  const s = await getState();
  if (s.bearerToken) return s;
  await pair();
  return getState();
}

// ---- MCP relay ---------------------------------------------------------

async function mcpCall(method, params) {
  const s = await ensurePaired();
  const r = await fetch(`${s.daemonUrl}/mcp`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${s.bearerToken}`,
      "Content-Type": "application/json",
      "Accept": "application/json, text/event-stream",
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: Math.floor(Math.random() * 1e9),
      method,
      params,
    }),
  });
  if (r.status === 401) {
    // Token rotated under us. Drop it and re-pair next call.
    await setState({ bearerToken: null });
    throw new Error("token rejected; will re-pair");
  }
  if (!r.ok) {
    throw new Error(`mcp ${method} returned ${r.status}`);
  }
  const body = await r.json();
  if (body.error) {
    throw new Error(`mcp ${method} error: ${body.error.message}`);
  }
  return body.result;
}

// Convenience: list scopes available on the daemon. Used by the popup
// scope picker. The /v1/scopes endpoint returns all scopes the bearer
// token has access to (i.e. everything on this single-user daemon).
async function listScopes() {
  const s = await ensurePaired();
  const r = await fetch(`${s.daemonUrl}/v1/scopes`, {
    headers: { "Authorization": `Bearer ${s.bearerToken}` },
  });
  if (!r.ok) throw new Error(`/v1/scopes returned ${r.status}`);
  return r.json();
}

// ---- message router ----------------------------------------------------

// Content scripts (and the popup) send messages to this worker via
// chrome.runtime.sendMessage(...). Each message has a `type` field;
// we dispatch and return the result via the sendResponse callback.
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      switch (msg.type) {
        case "ping":
          sendResponse({ ok: true });
          return;

        case "getState":
          sendResponse({ ok: true, state: await getState() });
          return;

        case "setState":
          await setState(msg.patch || {});
          sendResponse({ ok: true, state: await getState() });
          return;

        case "pair":
          sendResponse({ ok: true, data: await pair() });
          return;

        case "recall": {
          // msg.query (string), msg.scope (string, optional), msg.limit (int, optional)
          const state = await getState();
          const scope = msg.scope || state.activeScope;
          if (!scope) {
            sendResponse({ ok: false, error: "no active scope set" });
            return;
          }
          const result = await mcpCall("tools/call", {
            name: "recall",
            arguments: {
              query: msg.query,
              scope,
              limit: msg.limit || 5,
            },
          });
          sendResponse({ ok: true, result });
          return;
        }

        case "listScopes":
          sendResponse({ ok: true, scopes: await listScopes() });
          return;

        case "projectBriefing": {
          const state = await getState();
          const scope = msg.scope || state.activeScope;
          if (!scope) {
            sendResponse({ ok: false, error: "no active scope" });
            return;
          }
          const result = await mcpCall("tools/call", {
            name: "project_briefing",
            arguments: { scope },
          });
          sendResponse({ ok: true, result });
          return;
        }

        case "note": {
          // Iter 35: "Save to Skein" button on assistant turns. Pass the
          // turn text straight to MCP note() — daemon classifies type +
          // tags + value automatically. msg.content (string, required);
          // msg.fromRecall (string, optional) for outcome linking.
          const state = await getState();
          const scope = msg.scope || state.activeScope;
          if (!scope) {
            sendResponse({ ok: false, error: "no active scope set" });
            return;
          }
          if (!msg.content || typeof msg.content !== "string") {
            sendResponse({ ok: false, error: "missing content" });
            return;
          }
          const args = { content: msg.content, scope };
          if (msg.fromRecall) args.from_recall = msg.fromRecall;
          const result = await mcpCall("tools/call", {
            name: "note",
            arguments: args,
          });
          sendResponse({ ok: true, result });
          return;
        }

        default:
          sendResponse({ ok: false, error: `unknown message type: ${msg.type}` });
      }
    } catch (err) {
      console.error("[skein] message handler error", msg.type, err);
      sendResponse({ ok: false, error: String(err && err.message || err) });
    }
  })();
  // returning true keeps the message channel open for the async response
  return true;
});

// ---- pair on install ---------------------------------------------------

chrome.runtime.onInstalled.addListener(async () => {
  console.info("[skein] extension installed; attempting pair");
  try {
    await pair();
  } catch (err) {
    console.warn("[skein] initial pair failed; will retry on first use:", err.message);
  }
});
