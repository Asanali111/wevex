#!/usr/bin/env sh
# Skein installer — one-time bootstrap.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ameliomar/skein/main/bin/install.sh | sh
# or, when run from inside the cloned repo:
#   ./bin/install.sh
#
# What it does:
#   1. Verify Python 3.9+
#   2. Create a venv at ~/.skein/venv
#   3. pip install Skein into it (from the repo we're invoked from, or git-clone)
#   4. Symlink the `skein` binary onto your PATH
#       (prefers /usr/local/bin, falls back to ~/.local/bin if not writable)
#   5. Print "now run `skein up`"
#
# Idempotent — safe to re-run; updates an existing install in place.

set -eu

REPO_URL="${SKEIN_REPO:-https://github.com/ameliomar/skein.git}"
SKEIN_HOME="${SKEIN_HOME:-$HOME/.skein}"
SOURCE_DIR="$SKEIN_HOME/source"
VENV_DIR="$SKEIN_HOME/venv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
say()  { printf '%s\n' "$*"; }
ok()   { printf '\033[32m✓\033[0m  %s\n' "$*"; }
warn() { printf '\033[33m⚠\033[0m  %s\n' "$*"; }
die()  { printf '\033[31m✗\033[0m  %s\n' "$*" >&2; exit 1; }

need() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not on PATH."
}

# ---------------------------------------------------------------------------
# 1. Check Python 3.9+
# ---------------------------------------------------------------------------
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  fi
done
[ -n "$PYTHON" ] || die "Python 3.9+ is required. Install via 'brew install python' or apt."
ok "Found $($PYTHON --version) at $(command -v "$PYTHON")"

# ---------------------------------------------------------------------------
# 2. Locate or fetch source
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
if [ -f "$PROJECT_ROOT/pyproject.toml" ] && [ -d "$PROJECT_ROOT/skein" ]; then
  # We're being run from inside a checkout — install from there
  SOURCE_DIR="$PROJECT_ROOT"
  ok "Installing from local checkout: $SOURCE_DIR"
else
  # curl|sh path: clone or update into ~/.skein/source
  need git
  if [ -d "$SOURCE_DIR/.git" ]; then
    say "  Updating $SOURCE_DIR …"
    git -C "$SOURCE_DIR" pull --ff-only --quiet || warn "git pull failed; continuing with existing source"
  else
    say "  Cloning $REPO_URL → $SOURCE_DIR …"
    mkdir -p "$SKEIN_HOME"
    git clone --quiet "$REPO_URL" "$SOURCE_DIR"
  fi
  ok "Source ready at $SOURCE_DIR"
fi

# ---------------------------------------------------------------------------
# 3. Create venv (if missing) and install
# ---------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  say "  Creating venv at $VENV_DIR …"
  "$PYTHON" -m venv "$VENV_DIR"
  ok "Created venv"
fi

VENV_PIP="$VENV_DIR/bin/pip"
VENV_SKEIN="$VENV_DIR/bin/skein"

say "  Installing Skein …"
"$VENV_PIP" install --quiet --upgrade pip
"$VENV_PIP" install --quiet -e "$SOURCE_DIR"
ok "Skein installed in $VENV_DIR"

[ -x "$VENV_SKEIN" ] || die "Install completed but $VENV_SKEIN is not executable."

# ---------------------------------------------------------------------------
# 4. Symlink onto PATH
# ---------------------------------------------------------------------------
LINK_TARGET=""
case ":$PATH:" in
  *":/usr/local/bin:"*)
    if [ -w /usr/local/bin ] || [ "$(id -u)" = "0" ]; then
      LINK_TARGET=/usr/local/bin/skein
    fi ;;
esac
if [ -z "$LINK_TARGET" ]; then
  case ":$PATH:" in
    *":$HOME/.local/bin:"*)
      LINK_TARGET="$HOME/.local/bin/skein"
      mkdir -p "$HOME/.local/bin" ;;
  esac
fi
if [ -z "$LINK_TARGET" ]; then
  # PATH doesn't contain a writable dir we like — fall back to ~/.local/bin
  # and warn the user to add it
  LINK_TARGET="$HOME/.local/bin/skein"
  mkdir -p "$HOME/.local/bin"
  warn "Adding ~/.local/bin to your PATH is recommended:"
  warn "    echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc"
fi

ln -sf "$VENV_SKEIN" "$LINK_TARGET"
ok "Symlinked $VENV_SKEIN → $LINK_TARGET"

# ---------------------------------------------------------------------------
# 5. Friendly summary
# ---------------------------------------------------------------------------
INSTALLED_VERSION="$("$VENV_SKEIN" --version 2>/dev/null | awk '{print $NF}' || true)"
say ""
ok "Skein ${INSTALLED_VERSION:-installed}.  Now run:"
say ""
say "    cd ~/Documents/your-project"
say "    skein up"
say ""
say "That's it. Every connected LLM (Claude Code, Cursor, Codex, Gemini CLI,"
say "Antigravity, …) will share the same context for that project."
