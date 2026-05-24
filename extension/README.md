# Skein browser extension

Inject local Skein context into prompts on **claude.ai**, **chatgpt.com**,
and **gemini.google.com** — and save assistant turns back to Skein
with one click.

The extension is a thin client of your existing `skein up` daemon. It
runs entirely against `127.0.0.1`. Prompts never leave the machine.

## What it does

1. **Context injection on send.** When you hit Enter, the extension
   asks Skein for context relevant to your prompt and prepends a
   `[Skein context — auto-injected ...]` block before submitting.
   If Skein has no high-signal match (daemon says `relevance=low|none`),
   the extension skips injection so it doesn't waste tokens.

2. **Save to Skein (iter 35).** Every assistant turn renders a small
   "Save to Skein" button in the top-right corner. Click to save the
   turn as a note — the daemon classifies type / tags / value
   automatically. Useful for capturing one-off insights mid-conversation
   that would otherwise evaporate when the session ends.

3. **Floating badge.** Bottom-right corner shows daemon health, active
   scope, and fragment count. Green = ready, yellow = no scope picked,
   red = daemon unreachable.

## Install (5 minutes)

### 1. Make sure your daemon is at v0.2.0 or newer

```bash
skein --version          # should print 0.2.0 (or higher)
skein update             # if not, bump and restart
```

The extension's pairing endpoint (`/v1/pair-browser`) lands in v0.2.0.
Older daemons will refuse to pair.

### 2. Load the extension in a Chromium browser

Works in Chrome, Brave, Arc, Edge — anything Chromium-based.

1. Open `chrome://extensions` (or `brave://extensions`, etc.).
2. Toggle **Developer mode** on (top-right corner).
3. Click **Load unpacked**.
4. Pick this directory: `~/Documents/company-brain/extension/`.
5. Chrome should list "Skein for browser LLMs · 0.2.0" with a small icon.

Firefox support is on the roadmap (MV3 differences are small); Safari
is deferred (Xcode + Apple Developer Program required).

### 3. Pair and pick a scope

1. Click the Skein icon in the toolbar — opens the popup.
2. **Status** should turn green: `✓ paired · N scope(s) available`.
   If red, your daemon isn't running — `skein up` first.
3. **Scope** dropdown — pick your project (e.g. `project:myapp`).
4. **Inject context on send** is on by default. Leave it.
5. **Query** field + **Test recall** — confirm end-to-end works.

### 4. Use it

Open any supported site:
- `https://claude.ai/new`
- `https://chatgpt.com/`
- `https://gemini.google.com/app`

Type a real question. Hit Enter. You should see:

- A small blue toast: `Skein → recalling context…`
- A `[Skein context — auto-injected ...]` block prepended to your prompt.
- The message submits with context attached.
- The assistant references project facts it couldn't have known otherwise.

After the assistant finishes streaming, hover any turn to reveal the
**Save to Skein** button in its top-right. Click to capture.

## Verifying it's working

Three independent signals:

**Console (`Cmd-Opt-I` → Console, filter `[skein]`):**
```
[skein] content script loaded for claude.ai
[skein] paired ✓ daemon= http://127.0.0.1:8765
[skein] intercept (keydown:Enter) for query: how does auth work…
[skein] injecting 412 chars of context, relevance=high
```

**Daemon log:**
```bash
skein tail   # or tail -f ~/.config/skein/logs/daemon.log
```
You'll see `POST /v1/pair-browser` once, then `POST /mcp` per submit.

**Assistant quality.** Ask something only Skein could know — e.g.
*"what did we decide about the auth flow?"*. Without the extension,
the LLM has no context. With it on, the answer is specific.

## Turning it off

- **Per-message:** hold Shift+Enter — the interceptor only fires on
  bare Enter. (Plus the popup toggle.)
- **Per-session:** toolbar icon → uncheck **Inject context on send**.
- **Entirely:** `chrome://extensions` → toggle Skein off, or **Remove**.

## What's safe

- Every request goes to `127.0.0.1` only. The extension makes zero
  outbound calls to anything except your local daemon.
- The extension reads your prompt text (it has to, to know what to
  recall) but never sends it anywhere except the daemon.
- The bearer token is stored in `chrome.storage.local`, sandboxed to
  this extension. Other extensions and websites can't read it.
- The daemon's `/v1/pair-browser` endpoint validates the request
  `Origin` against `^chrome-extension://[a-z0-9]+$` so a malicious
  webpage can't pair itself.

## Known limits

- **DOM selector drift.** ChatGPT and Gemini ship UI changes
  frequently — selectors are written defensively (multiple fallbacks
  per site) but a hard breaking change will need a selector update.
  Check the console for `[skein] prompt element gone` if injection
  silently stops.
- **No streaming-aware feedback.** The "injected" toast disappears
  after 1.5 s; for long responses the console logs are the audit trail.
- **Firefox / Safari not yet supported.** Chromium MV3 only.
