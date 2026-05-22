// Skein content script — chatgpt.com / chat.openai.com
//
// Site-specific selectors only. Shared core in content_common.js.
//
// ChatGPT's composer has changed shape twice in two years:
//   1. Legacy:  <textarea id="prompt-textarea">  (still present on some
//               experimental rollouts and the legacy chat.openai.com
//               domain).
//   2. Current: <div id="prompt-textarea" contenteditable="true">
//               wrapping a ProseMirror instance.
//
// We probe in priority order: contenteditable first (the dominant
// shape in 2026), textarea last (legacy fallback). setPromptText
// dispatches the right event family for whichever it found.

(() => {
  function findPromptElement() {
    return (
      document.querySelector('div#prompt-textarea[contenteditable="true"]') ||
      document.querySelector('div[contenteditable="true"].ProseMirror') ||
      document.querySelector('div[contenteditable="true"][data-id]') ||
      document.querySelector('textarea#prompt-textarea') ||
      document.querySelector('div[contenteditable="true"]')
    );
  }

  function findSendButton() {
    return (
      document.querySelector('button[data-testid="composer-send-button"]') ||
      document.querySelector('button[data-testid="send-button"]') ||
      document.querySelector('button[aria-label*="Send prompt" i]') ||
      document.querySelector('button[aria-label*="Send" i]') ||
      document.querySelector('button[type="submit"]')
    );
  }

  // Smart setter: handles both the legacy <textarea> path (needs the
  // React-aware value setter + InputEvent) and the modern contenteditable
  // path (execCommand insertText, same as the default in
  // content_common.js).
  function setPromptText(el, text) {
    if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
      const proto = Object.getPrototypeOf(el);
      const desc = Object.getOwnPropertyDescriptor(proto, "value");
      if (desc && desc.set) {
        desc.set.call(el, text);
      } else {
        el.value = text;
      }
      el.dispatchEvent(new Event("input", { bubbles: true }));
      return;
    }
    el.focus();
    document.execCommand("selectAll", false, null);
    document.execCommand("delete", false, null);
    document.execCommand("insertText", false, text);
  }

  if (!globalThis.__SkeinCommon) {
    console.warn("[skein] content_common.js not loaded; aborting chatgpt.com script");
    return;
  }

  globalThis.__SkeinCommon.init({
    siteName: "chatgpt.com",
    findPromptElement,
    findSendButton,
    setPromptText,
  });
})();
