// Skein content script — gemini.google.com
//
// Site-specific selectors only. Shared core in content_common.js.
//
// Gemini's composer is a `<rich-textarea>` Angular/Lit custom element
// that wraps a Quill `.ql-editor` contenteditable internally. Light-DOM
// querySelector reaches it because rich-textarea exposes its content
// as light children, not shadow DOM.
//
// Send button is an Angular Material icon button with aria-label
// "Send message". On some experimental rollouts it lacks the
// mat-icon-button selector, so we fall back to aria-label probing.

(() => {
  function findPromptElement() {
    return (
      document.querySelector('rich-textarea [contenteditable="true"]') ||
      document.querySelector('div.ql-editor[contenteditable="true"]') ||
      document.querySelector('.ql-editor') ||
      document.querySelector('[contenteditable="true"][role="textbox"]') ||
      document.querySelector('div[contenteditable="true"]')
    );
  }

  function findSendButton() {
    // The visible send arrow. aria-label is consistently "Send message"
    // across Gemini variants; data-mat-icon-name="send" exists on the
    // inner icon when Material is in use.
    return (
      document.querySelector('button[aria-label="Send message"]') ||
      document.querySelector('button[aria-label*="Send message" i]') ||
      document.querySelector('button[aria-label*="Send" i]:not([disabled])') ||
      document.querySelector('.send-button') ||
      document.querySelector('button[mat-icon-button][aria-label*="Send" i]')
    );
  }

  // Iter 35: assistant turns for Save-to-Skein. Gemini renders the
  // model response inside `<message-content>` Angular components; some
  // rollouts use `.model-response-text` as the primary class on the
  // text container. We probe in priority order.
  function findAssistantTurns() {
    const nodes = document.querySelectorAll(
      'message-content.model-response-text, ' +
      'message-content[id*="model-response"], ' +
      '.model-response-text',
    );
    return Array.from(nodes);
  }

  if (!globalThis.__SkeinCommon) {
    console.warn("[skein] content_common.js not loaded; aborting gemini.google.com script");
    return;
  }

  globalThis.__SkeinCommon.init({
    siteName: "gemini.google.com",
    findPromptElement,
    findSendButton,
    findAssistantTurns,
  });
})();
