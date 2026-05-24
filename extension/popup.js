// Skein popup — scope picker + daemon URL + on/off + manual test.
//
// All real work is in background.js; this file just renders state and
// fires message-router requests when the user changes a setting.

const $status = document.getElementById("status");
const $daemonUrl = document.getElementById("daemonUrl");
const $scope = document.getElementById("scope");
const $enabled = document.getElementById("enabled");
const $repair = document.getElementById("repair");
const $test = document.getElementById("test");
const $testQuery = document.getElementById("testQuery");
const $testResult = document.getElementById("test-result");

function setStatus(text, kind = "ok") {
  $status.className = `status ${kind}`;
  $status.textContent = text;
}

async function send(msg) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(msg, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        resolve(response);
      }
    });
  });
}

async function refresh() {
  const { ok, state, error } = await send({ type: "getState" });
  if (!ok) {
    setStatus(`state error: ${error}`, "err");
    return;
  }
  $daemonUrl.value = state.daemonUrl;
  $enabled.checked = state.enabled;

  // Try to fetch scopes — if it works, daemon + token are healthy.
  try {
    const r = await send({ type: "listScopes" });
    if (!r.ok) throw new Error(r.error);
    const scopes = (r.scopes || []).filter((s) => s.type !== "personal" || true);
    $scope.innerHTML = '<option value="">— pick a scope —</option>';
    for (const s of scopes) {
      const opt = document.createElement("option");
      opt.value = s.handle;
      opt.textContent = `${s.handle}  (${s.name || ""})`;
      if (s.handle === state.activeScope) opt.selected = true;
      $scope.appendChild(opt);
    }
    setStatus(`✓ paired · ${scopes.length} scope(s) available`, "ok");
  } catch (err) {
    setStatus(`daemon unreachable — ${err.message}`, "err");
  }
}

$daemonUrl.addEventListener("change", async () => {
  await send({ type: "setState", patch: { daemonUrl: $daemonUrl.value.trim(), bearerToken: null } });
  setStatus("daemon URL changed — re-pairing…", "warn");
  await send({ type: "pair" });
  refresh();
});

$scope.addEventListener("change", async () => {
  await send({ type: "setState", patch: { activeScope: $scope.value || null } });
  setStatus(`scope → ${$scope.value || "(none)"}`, "ok");
});

$enabled.addEventListener("change", async () => {
  await send({ type: "setState", patch: { enabled: $enabled.checked } });
  setStatus($enabled.checked ? "injection on" : "injection off", $enabled.checked ? "ok" : "warn");
});

$repair.addEventListener("click", async () => {
  setStatus("re-pairing…", "warn");
  try {
    await send({ type: "setState", patch: { bearerToken: null } });
    await send({ type: "pair" });
    refresh();
  } catch (err) {
    setStatus(`pair failed: ${err.message}`, "err");
  }
});

$test.addEventListener("click", async () => {
  const query = $testQuery.value.trim();
  if (!query) {
    $testResult.style.display = "block";
    $testResult.textContent = "enter a query to test (e.g. 'auth flow', 'what we shipped this week')";
    return;
  }
  $testResult.style.display = "block";
  $testResult.textContent = `calling recall(${JSON.stringify(query)})…`;
  try {
    const r = await send({ type: "recall", query, limit: 3 });
    if (!r.ok) {
      $testResult.textContent = `error: ${r.error}`;
      return;
    }
    const text = (r.result && r.result.content && r.result.content[0] && r.result.content[0].text) || JSON.stringify(r.result, null, 2);
    $testResult.textContent = text.slice(0, 1000);
  } catch (err) {
    $testResult.textContent = `error: ${err.message}`;
  }
});

// Enter inside the query field triggers test.
$testQuery.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    $test.click();
  }
});

refresh();
