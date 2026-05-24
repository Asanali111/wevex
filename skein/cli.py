"""Skein CLI — all user-facing commands.

After ADR-002 (iter 26), the visible surface is ten commands. The bulk of
what used to be top-level CLI is now either (a) daemon background work,
(b) MCP tools the agent calls, or (c) sections folded into the diagnostic
commands below. The deletion-candidate commands are still wired up but
hidden=True so the next session can verify nothing relies on them before
removing the code.

Visible commands:
  up         Start the daemon, register the cwd, connect detected clients.
  down       Stop everything cleanly.
  restart    Restart the daemon.
  status     One-screen health: daemon, clients, fragment + chunk counts.
  doctor     Deep diagnostic; --clean and --reingest for cleanup.
  tail       Live event stream.
  briefing   Project state. With --since, becomes the cross-tool diff feed.
  tui        Interactive control panel.
  config     View or set runtime configuration.
  connect    Wire installed LLM tools through Skein (--remove to disconnect).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

# Heavier rich imports (Panel, Table, `rich.print`) are loaded lazily inside the
# handlers that need them — they cost ~22 ms cumulatively at module import time
# and aren't used by hot commands like `skein --help`, `--version`, or status.
# See `_panel()`, `_table()`, `_rprint()` helpers below.

console = Console()
err_console = Console(stderr=True)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_config():
    from .config import get_config
    return get_config()


def _client(base_url: Optional[str] = None, token: Optional[str] = None):
    """Return an httpx.Client pointed at the daemon."""
    import httpx
    cfg = _get_config()
    url = base_url or cfg.base_url
    tok = token or cfg.bearer_token
    return httpx.Client(
        base_url=url,
        headers={"Authorization": f"Bearer {tok}"},
        timeout=30.0,
    )


def _require_running(client) -> bool:
    """Check daemon is up; print error and exit if not."""
    try:
        resp = client.get("/health")
        resp.raise_for_status()
        return True
    except Exception:
        err_console.print(
            "[bold red]✗[/bold red] Skein daemon is not running. "
            "Start it with [bold]skein serve[/bold]."
        )
        sys.exit(1)


def _default_scope() -> str:
    cfg = _get_config()
    return cfg.default_scope


def _resolve_self_bin() -> str:
    """Best-effort path to the `skein` executable for embedding in hooks/launchd.

    Prefer ``/usr/local/bin/skein`` (the symlink the installer creates) so the
    written hooks survive the venv being moved or rebuilt.  Fall back to the
    venv binary, then to plain ``"skein"``.
    """
    import shutil
    candidates = [
        "/usr/local/bin/skein",
        str(Path.home() / ".local" / "bin" / "skein"),
    ]
    for c in candidates:
        if Path(c).is_file():
            return c
    found = shutil.which("skein")
    return found or "skein"


def _resolve_scope(cli_scope: Optional[str]) -> str:
    """Resolve the active scope handle for any CLI command.

    Honors (in order): --scope flag, SKEIN_SCOPE env, .skein/scope pin, config default.
    Prints a one-line note to stderr the first time we fall through to a non-CLI source
    so the user can see when they're getting an inherited pin.
    """
    from .scope_resolver import resolve_scope
    cfg = _get_config()
    scope, source = resolve_scope(cli_scope, config_default=cfg.default_scope)
    # Soft hint: when the user didn't pass --scope but a .skein/scope pin took effect,
    # show a dim line so it's not surprising. Skip for noisy commands (recall/search).
    if source == "pin" and cli_scope is None and not os.environ.get("SKEIN_QUIET_PIN"):
        err_console.print(
            f"[dim]using scope [cyan]{scope}[/cyan] from .skein/scope[/dim]"
        )
    return scope


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(package_name="skein", prog_name="skein")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Skein — local MCP context bus for coding LLMs.

    \b
    Quick start:
        skein up              # start daemon, connect clients, watch this repo
        skein status          # see what's wired up
        skein doctor          # deep diagnostic
        skein briefing        # what's the state of this project?
        skein down            # stop everything

    \b
    Day-to-day, you don't need a CLI. The MCP tools (recall / remember /
    note_decision / boost / bury / archaeology / supersede) live inside your
    LLM — Claude Code, Cursor, Codex, etc. — and the daemon takes care of
    sync, gc, and inbox approval automatically.

    Full docs: https://github.com/ameliomar/skein
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# up — the one-command bootstrap
# ---------------------------------------------------------------------------

@main.command()
@click.argument("path", required=False, default=".",
                type=click.Path(exists=True, file_okay=False))
@click.option("--scope", default=None,
              help="Override the auto-detected scope handle.")
@click.option("--no-persist", is_flag=True, default=False,
              help="Don't install launchd / systemd unit; daemon dies on logout.")
@click.option("--no-ingest", is_flag=True, default=False,
              help="Skip codebase ingestion (faster).")
@click.option("--no-sync", is_flag=True, default=False,
              help="Skip writing per-client MCP configs.")
@click.option("--no-hooks", is_flag=True, default=False,
              help="Skip installing autonomous hooks.")
@click.option("--global", "user_global", is_flag=True, default=False,
              help="Also install hooks user-globally (~/.claude/settings.json).")
def up(
    path: str,
    scope: Optional[str],
    no_persist: bool,
    no_ingest: bool,
    no_sync: bool,
    no_hooks: bool,
    user_global: bool,
) -> None:
    """One-command bootstrap: init + persistent daemon + hooks + sync + ingest.

    \b
    Idempotent — safe to run repeatedly. After this, every connected LLM
    (Claude Code, Cursor, Codex, Gemini CLI, Antigravity, …) automatically
    has shared context for this project.

    \b
    Run from any project directory:
        cd ~/Documents/your-app
        skein up

    \b
    What it does:
      1. Initialise config + bearer token (if needed)
      2. Start the daemon as a background service that survives reboot
         (launchd on macOS, systemd-user on Linux, nohup elsewhere)
      3. Auto-detect a scope from the git remote or directory name
      4. Install autonomous hooks (.claude/settings.json + .cursor/rules + .skein/scope)
      5. Sync MCP configs to all installed LLM clients
      6. Ingest the codebase for RAG
    """
    from .agents_md import render_agents_md
    from .auth import generate_token
    from .config import SkeinConfig, _default_config_path, load_config
    from .daemon import ensure_running
    from .embeddings import get_provider as _get_emb
    from .hooks_install import install_hooks
    from .ingest import ingest_directory
    from .models import IdentityCreate, ScopeCreate
    from .scope_resolver import auto_detect_scope
    from .storage import Storage
    from .sync import sync_all
    # Imported once at the top so every reference below is bound. Python
    # turns `ui` into a function-local because of the later `from . import
    # ui` statements; if those run lazily the early references (ingest
    # progress line, scanner block) raise UnboundLocalError. Hoisting fixes
    # a pre-existing latent crash on the warm-ingest path.
    from . import ui

    repo_path = Path(path).resolve()

    # ---- 1. init (if missing) ----
    cfg_path = _default_config_path()
    if not cfg_path.exists():
        from .embeddings import best_available_provider_name
        token = generate_token()
        # Default new installs to local fastembed (no API key, ~130 MB
        # one-time model download). If the fastembed library somehow isn't
        # importable, fall back to bm25 (FTS5-only) so init still succeeds.
        embedding_provider = best_available_provider_name()
        if embedding_provider == "fastembed":
            try:
                import importlib
                importlib.import_module("fastembed")
            except ImportError:
                embedding_provider = "bm25"
        cfg = SkeinConfig({
            "bearer_token": token,
            "embedding_provider": embedding_provider,
            "default_scope": "project:default",
        })
        cfg.save(cfg_path)
        console.print(f"[green]✓[/green] Initialised config at [dim]{cfg_path}[/dim]")
        console.print(
            f"[dim]Embedding provider: [bold]{embedding_provider}[/bold]"
            + (" (fastembed not installed — run "
               "[bold]pip install fastembed[/bold] to enable semantic search)"
               if embedding_provider == "bm25" else "")
            + "[/dim]"
        )
    cfg = load_config()

    # ---- 2. resolve scope ----
    scope_handle = scope or auto_detect_scope(repo_path)
    console.print(f"[bold]Project scope:[/bold] [cyan]{scope_handle}[/cyan]")

    # ---- 2.5. Safety guards run BEFORE the daemon start.
    # Iter 27 reordering: guards used to live inside the post-daemon try-
    # block, which on Windows CI meant a slow / failing daemon-start
    # short-circuited and the user never saw the actual cause (e.g.
    # "no .git folder"). They are also semantically pointless after we've
    # already paid the cost of spawning the daemon. Moving them here is a
    # DX improvement on every platform.
    from .ingest import _refuse_root
    from . import ui as _guard_ui

    refusal = _refuse_root(repo_path)
    if refusal and not no_ingest:
        err_console.print(
            f"  {_guard_ui.mark('err')} Refusing to ingest {refusal}."
        )
        _guard_ui.hint(
            "Run [bold]skein up[/bold] from a real project directory "
            "(e.g. one with a [bold].git[/bold]/ folder)."
        )
        sys.exit(1)

    # `.git` is required unless --no-ingest is set or the escape hatch is on.
    if (
        not no_ingest
        and not (repo_path / ".git").exists()
        and os.environ.get("SKEIN_ALLOW_NO_GIT") != "1"
    ):
        err_console.print(
            f"  {_guard_ui.mark('err')} No [bold].git[/bold] folder in "
            f"{repo_path} — refusing to ingest."
        )
        _guard_ui.hint(
            "Run [bold]skein up[/bold] from inside a git repo, OR "
            "use [bold]skein up --no-ingest[/bold] to skip indexing, OR "
            "set [bold]SKEIN_ALLOW_NO_GIT=1[/bold] if you really mean it."
        )
        sys.exit(1)

    # ---- 3. start daemon (persistent by default) ----
    # On macOS, launchd-launched processes can't read files under ~/Documents,
    # ~/Desktop, ~/Downloads, etc. (TCC). If our venv lives in one of those,
    # auto-relocate to ~/.skein/venv before starting the service.
    skein_bin_for_daemon = _resolve_self_bin()
    if not no_persist:
        from .daemon import is_tcc_protected_path, relocate_venv_to_skein_home
        if is_tcc_protected_path(Path(skein_bin_for_daemon)):
            console.print(
                "[yellow]⚠[/yellow] Your skein install is inside a macOS-protected "
                f"folder ([dim]{Path(skein_bin_for_daemon).parent.parent}[/dim]).\n"
                "    Relocating to [bold]~/.skein/venv[/bold] so launchd can run it…"
            )
            with console.status("[dim]Building TCC-safe venv at ~/.skein/venv…[/dim]",
                                spinner="dots"):
                try:
                    new_bin = relocate_venv_to_skein_home()
                except Exception as e:
                    err_console.print(
                        f"[red]✗[/red] Relocation failed: {e}\n"
                        f"    Falling back to non-persistent (nohup) daemon. "
                        f"Run skein up again later or move the venv manually."
                    )
                    no_persist = True
                else:
                    skein_bin_for_daemon = str(new_bin)
                    console.print(f"[green]✓[/green] Relocated to [dim]{new_bin}[/dim]")

    # Iter 28: capture whether the daemon was already healthy at entry. The
    # warm path (daemon up + project already registered) can skip the
    # MCP-client resync (idempotent, costly per-client subprocess fan-out)
    # and the file-walking ingest_directory call (the watcher subprocess
    # already covers incremental re-ingest). Saves ~5–8 s on every warm
    # `skein up`.
    from .daemon import _check_health as _probe_health
    was_already_healthy = _probe_health(cfg.base_url)

    method_label = "background service" if not no_persist else "foreground (this terminal)"
    with console.status(f"[dim]Ensuring daemon is running ({method_label})…[/dim]",
                        spinner="dots"):
        try:
            status = ensure_running(
                persist=not no_persist,
                base_url=cfg.base_url,
                skein_bin=skein_bin_for_daemon,
            )
        except RuntimeError as e:
            err_console.print(f"[red]✗[/red] Daemon start failed: {e}")
            sys.exit(1)
    if status.healthy:
        console.print(
            f"[green]✓[/green] Daemon up via [bold]{status.method}[/bold]"
            f"{' pid='+str(status.pid) if status.pid else ''} "
            f"at [dim]{status.base_url}[/dim]"
        )
    else:
        err_console.print(
            f"[red]✗[/red] Daemon did not become healthy "
            f"(method={status.method}). See ~/.config/skein/logs/daemon.err"
        )
        sys.exit(1)

    # ---- 4. ensure scope row exists in DB ----
    storage = Storage(cfg.db_path)
    try:
        scope_obj = storage.get_scope(scope_handle)
        if not scope_obj:
            owner = storage.get_or_create_identity(IdentityCreate(
                handle=f"user:{cfg.bearer_token[:8] if cfg.bearer_token else 'cli'}",
                type="user", name="local-user",
            ))
            stype = scope_handle.split(":", 1)[0]
            if stype not in {"public", "org", "team", "project", "personal"}:
                stype = "project"
            scope_obj = storage.create_scope(ScopeCreate(
                handle=scope_handle,
                type=stype,
                name=scope_handle.split(":", 1)[-1],
                owner_id=owner.id,
            ))
            console.print(f"[green]✓[/green] Created scope [cyan]{scope_handle}[/cyan]")

        # ---- 5. hooks ----
        if not no_hooks:
            skein_bin = _resolve_self_bin()
            report = install_hooks(
                repo_path=repo_path, scope_handle=scope_handle,
                skein_bin=skein_bin, user_global=user_global,
            )
            for w in report.written:
                console.print(f"[green]✓[/green] {w}")
            if report.errors:
                for e in report.errors:
                    err_console.print(f"[red]✗[/red] {e}")

        # ---- 6. sync MCP configs ----
        # Iter 28: on the warm path (daemon already healthy at entry) the
        # per-client MCP config fan-out is skipped — each client.connect()
        # spawns a subprocess (e.g. `claude mcp add`) and the configs are
        # idempotent and rarely changed. The daemon's `_agents_md_sync_loop`
        # owns AGENTS.md regen so it still picks up new fragments within
        # `agents_md_sync_interval` (60 s default). `skein connect` is the
        # explicit path when the user actually adds a new client.
        if not no_sync and not was_already_healthy:
            from . import connections as conns
            connected = conns.get_connected_ids()
            if not connected:
                console.print(
                    "[yellow]⚠[/yellow] No LLM clients connected yet. "
                    "Run [bold]skein connect[/bold] to pick which tools should "
                    "share context, then re-run [bold]skein up[/bold]."
                )
            else:
                agents_md_content = render_agents_md(
                    scope_handle, storage,
                    daemon_url=cfg.base_url,
                    existing_content=(repo_path / "AGENTS.md").read_text()
                        if (repo_path / "AGENTS.md").exists() else None,
                )
                sync_result = sync_all(
                    daemon_url=cfg.base_url,
                    bearer_token=cfg.bearer_token,
                    scope_handle=scope_handle,
                    repo_path=repo_path,
                    agents_md_content=agents_md_content,
                    client_ids=connected,
                )
                written_count = len(sync_result.written)
                skipped_count = len(sync_result.skipped)
                console.print(
                    f"[green]✓[/green] Synced [bold]{written_count}[/bold] LLM client config(s)"
                    + (f" ({skipped_count} skipped)" if skipped_count else "")
                )
                if sync_result.errors:
                    for e in sync_result.errors:
                        err_console.print(f"[red]✗[/red] {e}")

        # ---- 6.5. Safety guards (_refuse_root + .git check) ran earlier,
        # before daemon start. See section 2.5 above (moved in iter 27 so
        # we don't pay the daemon-start cost for refused directories).

        # ---- 6.6. register project + spawn detached watcher (live re-ingest)
        # `--no-persist` callers (notably the test suite, or anyone trying a
        # one-shot ingest) skip both: we don't want a background watcher
        # outliving a transient invocation, and we don't want test runs to
        # pollute the user's real ~/.config/skein/projects.json.
        if not no_persist:
            from .projects import ProjectEntry, upsert_project
            from . import watcher_manager
            entry = ProjectEntry(
                scope=scope_handle,
                root=str(repo_path),
                source_root=repo_path.name,
            )
            upsert_project(entry)

            if watcher_manager.is_running(entry):
                console.print(
                    f"[dim]Watcher already running for [cyan]{scope_handle}[/cyan].[/dim]"
                )
            else:
                try:
                    pid = watcher_manager.spawn(
                        entry, skein_bin=_resolve_self_bin(),
                    )
                    if pid:
                        console.print(
                            f"[green]✓[/green] Auto-reingest watcher spawned (pid {pid}) — "
                            f"file changes will appear in search within ~2s"
                        )
                except Exception as e:
                    err_console.print(
                        f"[yellow]⚠[/yellow] Could not spawn watcher: {e}\n"
                        "    Manual `skein ingest` will still work."
                    )

        # ---- 7. ingest ----
        # Iter 28: on the warm path the watcher subprocess is already
        # running incremental re-ingest, so the explicit walk-and-chunk
        # below adds nothing but latency. The daemon's passive_scan loop
        # owns the package-manifest + docs path. Skip entirely when the
        # daemon was healthy on entry.
        run_ingest = not no_ingest and not was_already_healthy
        shared_provider = None
        shared_provider_err: Optional[str] = None
        if run_ingest:
            try:
                shared_provider = _get_emb(cfg.embedding_provider)
            except Exception as e:
                shared_provider_err = str(e)

        if run_ingest:
            from .ingest import count_ingestable_files

            if shared_provider is None:
                err_console.print(
                    f"  {ui.mark('warn')} Embedding provider unavailable: "
                    f"{shared_provider_err}"
                )
                ui.hint("Continuing with keyword-only ingest.")
            provider = shared_provider

            # Pre-walk so we know what we're getting into.
            try:
                total_files = count_ingestable_files(repo_path)
            except Exception:
                total_files = 0

            HUGE_INGEST_THRESHOLD = 5000
            if total_files > HUGE_INGEST_THRESHOLD:
                err_console.print(
                    f"  {ui.mark('warn')} {total_files:,} files would be "
                    f"indexed under [cyan]{scope_handle}[/cyan] — "
                    f"that's larger than expected for a project."
                )
                ui.hint(
                    "Re-run with [bold]--no-ingest[/bold], or move into a "
                    "smaller subdirectory, or pass [bold]--scope[/bold] "
                    "with a more specific handle."
                )
                if not click.confirm(
                    f"  Index {total_files:,} files anyway?", default=False,
                ):
                    sys.exit(1)

            from rich.progress import (
                Progress, BarColumn, TextColumn, TimeElapsedColumn,
                MofNCompleteColumn,
            )
            progress = Progress(
                TextColumn("  [dim]→[/dim] [bold]Indexing[/bold]"),
                BarColumn(bar_width=24),
                MofNCompleteColumn(),
                TextColumn("[dim]·[/dim]"),
                TimeElapsedColumn(),
                TextColumn("[dim]{task.description}[/dim]"),
                console=console,
                transient=True,
            )

            def progress_cb(rel_path, st):
                # Truncate very long paths so the line doesn't wrap
                disp = rel_path if len(rel_path) <= 50 else "…" + rel_path[-49:]
                progress.update(task_id, advance=1, description=disp)

            with progress:
                task_id = progress.add_task(
                    "starting…", total=total_files or None,
                )
                stats = ingest_directory(
                    repo_path, storage, provider,
                    scope_id=scope_obj.id,
                    source_root=repo_path.name,
                    prune_missing=True,
                    progress_cb=progress_cb,
                )
            kb = stats.bytes_processed / 1024
            ui.step(
                f"Indexed [bold]{stats.files_ingested}[/bold] files",
                detail=(
                    f"{stats.chunks_inserted} new · "
                    f"{stats.chunks_unchanged} unchanged · "
                    f"{stats.chunks_updated} updated · "
                    f"{stats.chunks_pruned} pruned · {kb:.0f} KB"
                ),
                state="ok",
            )
            if stats.embedding_degraded:
                err_console.print(
                    f"  {ui.mark('warn')} Embedding provider degraded mid-run — "
                    "some chunks have zero vectors."
                )
                ui.hint(
                    "Keyword search still works. Run "
                    "[bold]skein ingest .[/bold] later (or set "
                    "[bold]embedding_provider=hash[/bold] to skip embeddings)."
                )
            if stats.errors:
                err_console.print(
                    f"  {ui.mark('warn')} {len(stats.errors)} ingest errors "
                    "(first 3):"
                )
                for err in stats.errors[:3]:
                    err_console.print(f"      [dim]· {err}[/dim]")
    finally:
        storage.close()

    # ---- 7b/7c. Passive code + docs scans now live in the daemon ----
    # Iter 28: the synchronous scan_project / scan_docs blocks moved into
    # `skein/server.py::_passive_scan_loop` so `skein up` returns as soon
    # as the daemon is healthy. The daemon picks up new package manifests
    # and READMEs within `passive_scan_interval` (default 300 s) which is
    # well inside the user's "open editor" window.

    # ---- 8. friendly summary ----
    ui.header("Skein is ready", state="ok")
    home = str(Path.home())
    ui.fields([
        ("Project", f"[cyan]{str(repo_path).replace(home, '~', 1)}[/cyan]"),
        ("Scope",   f"[cyan]{scope_handle}[/cyan]"),
        ("Daemon",  f"[dim]{status.base_url}[/dim]  ·  via {status.method}"),
        ("MCP",     f"[dim]{status.base_url}/mcp[/dim]"),
    ])
    ui.blank()
    ui.hint(
        "Try [bold]skein recall \"<query>\"[/bold] · "
        "[bold]skein search \"<query>\"[/bold] · "
        "[bold]skein down[/bold] to stop"
    )


# ---------------------------------------------------------------------------
# down / restart / daemon
# ---------------------------------------------------------------------------

@main.command()
@click.option("--no-uninstall-hooks", is_flag=True, default=False,
              help="Leave the autonomous hooks in place; only stop the daemon.")
@click.option("--keep-registered", is_flag=True, default=False,
              help="Keep the project registered (daemon will still watch it on restart).")
@click.option("--repo", default=None, type=click.Path(file_okay=False),
              help="Project root for hook removal (default: cwd).")
def down(no_uninstall_hooks: bool, keep_registered: bool, repo: Optional[str]) -> None:
    """Stop the daemon (any backend), kill the watcher, unregister the project, and remove hooks."""
    from . import ui
    from .daemon import stop
    from .hooks_install import uninstall_hooks
    from . import watcher_manager
    from .projects import remove_project, list_projects

    ui.section("Skein down")
    ui.blank()

    # Kill any watcher subprocesses that target the current repo
    repo_path_for_kill = Path(repo) if repo else Path.cwd()
    repo_root_resolved = str(repo_path_for_kill.resolve())
    for entry in list_projects():
        if entry.root == repo_root_resolved and watcher_manager.is_running(entry):
            if watcher_manager.kill(entry):
                ui.step(f"Watcher stopped", detail=entry.scope, state="ok")

    status = stop()
    if status.running:
        ui.step("Daemon still running", detail=f"via {status.method}", state="warn")
    else:
        ui.step("Daemon stopped", state="ok")

    repo_path = Path(repo) if repo else Path.cwd()

    if not keep_registered:
        if remove_project(str(repo_path.resolve())):
            ui.step("Project unregistered",
                    detail=str(repo_path).replace(str(Path.home()), "~", 1),
                    state="ok")

    if not no_uninstall_hooks:
        report = uninstall_hooks(repo_path)
        for w in report.written:
            ui.step("Removed hook", detail=w, state="ok")
    ui.blank()


@main.command()
def restart() -> None:
    """Restart the daemon."""
    from . import ui
    from .daemon import restart as do_restart
    status = do_restart()
    if status.healthy:
        ui.blank()
        ui.step(f"Daemon restarted", detail=f"via {status.method}", state="ok")
        ui.blank()
    else:
        err_console.print(
            f"  {ui.mark('err')} Restart failed (method={status.method}). "
            f"See ~/.config/skein/logs/daemon.err"
        )
        sys.exit(1)


@main.command()
@click.option("--check", "check_only", is_flag=True, default=False,
              help="Only check for an update; don't install anything.")
def update(check_only: bool) -> None:
    """Upgrade Skein to the latest version and restart the daemon.

    Detects how Skein was installed (pipx / uv tool / pip) and runs the
    appropriate upgrade command, then restarts the daemon.
    """
    from . import ui
    from .version_check import check_for_update, _current_version, _fetch_latest

    current = _current_version()
    ui.blank()
    console.print(f"  Current version: [bold]{current}[/bold]")

    latest = _fetch_latest()
    if latest is None:
        err_console.print(
            "  [red]✗[/red] Could not reach PyPI. Check your network connection."
        )
        sys.exit(1)

    console.print(f"  Latest version:  [bold]{latest}[/bold]")

    from .version_check import _version_tuple
    if _version_tuple(latest) <= _version_tuple(current):
        ui.blank()
        console.print("  [green]Already up to date.[/green]")
        ui.blank()
        return

    if check_only:
        ui.blank()
        console.print(
            f"  [yellow]⬆ Update available:[/yellow] {current} → {latest}  "
            f"[dim](run: skein update)[/dim]"
        )
        ui.blank()
        return

    # Detect install method from the binary path and environment.
    import shutil
    skein_bin = shutil.which("skein") or sys.executable
    skein_path = Path(skein_bin).resolve()

    # Check for editable / dev install (source tree).
    is_editable = False
    try:
        import importlib.metadata as _meta
        dist = _meta.distribution("skn")
        direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
        is_editable = direct_url.get("dir_info", {}).get("editable", False)
    except Exception:
        pass

    if is_editable:
        ui.blank()
        console.print(
            "  [yellow]⚠[/yellow] Skein is installed in editable/dev mode.\n"
            "  Pull the latest changes and restart the daemon:\n"
            "\n"
            "    [bold]git pull && skein restart[/bold]\n"
            "\n"
            "  If you installed via the TCC workaround (macOS) also run:\n"
            "    [bold]skein down && rm -rf ~/.skein/source && skein up[/bold]"
        )
        ui.blank()
        return

    # Detect pipx: binary lives inside a pipx venv
    #   ~/.local/pipx/venvs/skn/bin/skein  or  ~/Library/Application Support/pipx/…
    is_pipx = "pipx" in str(skein_path) or "pipx" in os.environ.get("PIPX_HOME", "")
    if not is_pipx:
        # Secondary check: pipx creates a link under ~/.local/bin
        try:
            resolved = skein_path.resolve()
            is_pipx = "pipx" in str(resolved)
        except Exception:
            pass

    # Detect uv tool: binary lives inside ~/.local/share/uv/tools/skn/
    is_uv_tool = "uv" in str(skein_path) and "tools" in str(skein_path)
    if not is_uv_tool:
        try:
            is_uv_tool = "uv" in str(skein_path.resolve()) and "tools" in str(skein_path.resolve())
        except Exception:
            pass

    if is_pipx:
        upgrade_cmd = ["pipx", "upgrade", "skn"]
    elif is_uv_tool:
        upgrade_cmd = ["uv", "tool", "upgrade", "skn"]
    else:
        # Fallback: pip install --upgrade in the same Python that's running
        upgrade_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "skn"]

    ui.blank()
    ui.step(f"Upgrading skn → {latest}", detail=" ".join(upgrade_cmd), state="ok")
    ui.blank()

    try:
        result = subprocess.run(upgrade_cmd, capture_output=False, text=True)
        if result.returncode != 0:
            err_console.print(
                f"  [red]✗[/red] Upgrade command exited {result.returncode}."
            )
            sys.exit(result.returncode)
    except FileNotFoundError as exc:
        err_console.print(f"  [red]✗[/red] Command not found: {exc}")
        sys.exit(1)

    # Invalidate the update-check cache so next `skein status` shows up to date.
    try:
        from .version_check import _save_cache
        _save_cache(latest)
    except Exception:
        pass

    ui.blank()
    ui.step(f"Upgraded to {latest}", state="ok")

    # Restart the daemon so it picks up the new code.
    from .daemon import restart as do_restart
    restart_status = do_restart()
    if restart_status.healthy:
        ui.step("Daemon restarted", detail=f"via {restart_status.method}", state="ok")
    else:
        ui.step(
            "Daemon restart failed",
            detail="run `skein restart` manually",
            state="warn",
        )
    ui.blank()


# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -
# Helpers used by visible commands (hoisted out of ADR-002 deletion
# spans during iter 33 phase B).
# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -# -

def _render_clients_table(connected_ids: set) -> tuple:
    """Build (rows, detected_clients) where each row is suitable for the
    interactive picker."""
    from . import clients as clients_mod
    detected = clients_mod.detect_all()
    rows = []
    for entry in detected:
        if not entry["detected"]:
            continue
        rows.append({
            "id": entry["id"],
            "name": entry["display_name"],
            "note": entry["note"],
            "connected": entry["id"] in connected_ids,
        })
    return rows, detected


def _disconnect_impl(client_id: Optional[str], all_connected: bool) -> None:
    """Remove Skein from one or all connected LLM tools.

    \b
    Forms:
      skein disconnect cursor    surgically remove skein from cursor configs
      skein disconnect --all     disconnect everything
    """
    from . import clients as clients_mod
    from . import connections as conns
    from .sync import disconnect_client

    from . import ui

    if not client_id and not all_connected:
        err_console.print(
            f"  {ui.mark('err')} Pass a client id (e.g. [cyan]cursor[/cyan]) "
            "or [bold]--all[/bold]."
        )
        connected = conns.get_connected_ids()
        if connected:
            ui.hint(f"Currently connected: {', '.join(connected)}")
        sys.exit(1)

    if all_connected:
        targets = conns.get_connected_ids()
        if not targets:
            ui.hint("No clients are currently connected.")
            return
    else:
        if clients_mod.get_client(client_id) is None:
            err_console.print(
                f"  {ui.mark('err')} Unknown client id: [cyan]{client_id}[/cyan]"
            )
            ui.hint(f"Known ids: {', '.join(clients_mod.all_ids())}")
            sys.exit(1)
        if not conns.is_connected(client_id):
            ui.hint(f"[cyan]{client_id}[/cyan] is not currently connected.")
            return
        targets = [client_id]

    ui.blank()
    for cid in targets:
        try:
            modified = disconnect_client(cid)
            ui.step(f"Disconnected [cyan]{cid}[/cyan]", state="ok")
            for p in modified:
                ui.bullet(f"[dim]{ui.home_relative(p)}[/dim]",
                          indent=6, mark_str="└─")
        except Exception as e:
            err_console.print(f"  {ui.mark('err')} {cid}: {e}")
    ui.blank()


def _since_impl(
    since_arg: str,
    scope: Optional[str],
    types: tuple,
    exclude_tool: Optional[str],
    limit: int,
    output_json: bool,
) -> None:
    """List fragments created after a timestamp — cross-tool "what changed?" feed.

    \b
    Examples:
        skein since 1h
        skein since 2d --exclude-tool claude_code
        skein since 2026-05-12 --type decision --limit 20
    """
    iso = _parse_since(since_arg)
    scope_handle = _resolve_scope(scope)

    params: dict = {
        "scope": scope_handle,
        "since": iso,
        "limit": limit,
    }
    if exclude_tool:
        params["exclude_tool"] = exclude_tool
    if types:
        # The GET /v1/fragments endpoint takes a single `type` filter; if the
        # user passes multiple, query each and merge in display order. Rare
        # path so the extra roundtrip is fine.
        all_rows: list = []
        seen_ids: set = set()
        with _client() as client:
            _require_running(client)
            for t in types:
                p = dict(params)
                p["type"] = t
                resp = client.get("/v1/fragments", params=p)
                if resp.status_code != 200:
                    err_console.print(f"[red]✗[/red] Error {resp.status_code}: {resp.text}")
                    sys.exit(1)
                for row in resp.json():
                    if row["id"] not in seen_ids:
                        seen_ids.add(row["id"])
                        all_rows.append(row)
        # Re-sort merged set by created_at DESC and trim.
        all_rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        rows = all_rows[:limit]
    else:
        with _client() as client:
            _require_running(client)
            resp = client.get("/v1/fragments", params=params)
        if resp.status_code != 200:
            err_console.print(f"[red]✗[/red] Error {resp.status_code}: {resp.text}")
            sys.exit(1)
        rows = resp.json()

    if output_json:
        print(json.dumps(rows, indent=2))
        return

    from . import ui
    label = f"since {since_arg} → {iso}"
    ui.section(label)
    ui.blank()
    if not rows:
        suffix = f" (excluding {exclude_tool})" if exclude_tool else ""
        ui.bullet(f"No new fragments in [cyan]{scope_handle}[/cyan]{suffix}.")
        ui.blank()
        return

    for f in rows:
        meta_parts = [f"[yellow]{f['type']}[/yellow]"]
        tool = f.get("created_by_tool") or "?"
        meta_parts.append(f"[cyan]{tool}[/cyan]")
        if f.get("territory"):
            meta_parts.append(f"[dim]{f['territory']}[/dim]")
        if f.get("tags"):
            meta_parts.append(f"[dim]#{' #'.join(f['tags'])}[/dim]")
        console.print("  " + "  ".join(meta_parts))
        # Show first line of content + truncated remainder hint.
        content = f.get("content", "")
        first_line = content.split("\n", 1)[0]
        if len(first_line) > 100:
            first_line = first_line[:97] + "..."
        console.print(f"      {first_line}")
        console.print(
            f"      [dim]{f['id'][:8]} · {f['created_at'][:19]}[/dim]"
        )
        ui.blank()
    console.print(
        f"  [dim]{len(rows)} fragment{'s' if len(rows) != 1 else ''} "
        f"in [cyan]{scope_handle}[/cyan][/dim]"
    )
    ui.blank()


def _doctor_clean() -> None:
    """ADR-002: replace `skein gc` invocation. Delegates to the existing
    gc handler so the cleanup heuristics live in one place — when `gc`
    is eventually deleted, only this helper moves with it.
    """
    _gc_impl(yes=False, dry_run=False)


def _doctor_reingest() -> None:
    """ADR-002: replace `skein ingest .` invocation. Delegates to the
    existing ingest handler with the conservative defaults a re-ingest
    needs (no --reset, no --prune)."""
    _ingest_impl(path=".", scope=None, source_root=None,
                 chunk_lines=80, overlap_lines=10, include_exts=None,
                 extra_excludes=(), max_bytes=None,
                 prune=False, reset=False, dry_run=False, quiet=False)


def _doctor_reindex_embeddings() -> None:
    """Iter 32: re-embed every fragment under the active provider.

    Legacy 768-dim fragments from the gemini-era return cos=0 from vector
    search under the new fastembed 384-dim provider (they only reach
    `recall` via BM25). This walks every non-stale fragment, recomputes the
    embedding, and writes it back. Runs against the local DB directly — no
    daemon roundtrip — so it's safe under launchd's TCC restrictions.
    """
    from . import ui
    from .config import get_config
    from .embeddings import bytes_to_vec, get_provider, vec_to_bytes
    from .storage import Storage

    cfg = get_config()
    storage = Storage(cfg.db_path)
    try:
        provider = get_provider(cfg.embedding_provider)
        target_dim = getattr(provider, "dimension", 0)
        ui.section(f"Reindex embeddings → {cfg.embedding_provider} (dim={target_dim})")
        ui.blank()

        rows = storage._conn.execute(
            "SELECT id, content, embedding FROM fragments "
            "WHERE is_stale = 0 AND content IS NOT NULL AND content != ''",
        ).fetchall()
        total = len(rows)
        if total == 0:
            ui.step("Nothing to reindex", state="ok", detail="0 live fragments.")
            return

        ui.step(f"Reindexing {total} fragments", state="info")
        reindexed = 0
        already_ok = 0
        failed = 0
        mismatched = 0
        for row in rows:
            existing_dim = None
            if row["embedding"]:
                try:
                    existing_dim = len(bytes_to_vec(row["embedding"]))
                except Exception:
                    existing_dim = None
            if existing_dim == target_dim and existing_dim is not None:
                already_ok += 1
                continue
            if existing_dim is not None and existing_dim != target_dim:
                mismatched += 1
            try:
                vec = provider.embed_one(row["content"])
                blob = vec_to_bytes(vec)
                storage._conn.execute(
                    "UPDATE fragments SET embedding = ? WHERE id = ?",
                    (blob, row["id"]),
                )
                reindexed += 1
            except Exception as e:
                failed += 1
                ui.hint(f"Failed {row['id'][:8]}…: {e}")
        storage._conn.commit()
        ui.blank()
        ui.step(
            f"Reindexed {reindexed}", state="ok",
            detail=f"already aligned: {already_ok}, dimension-mismatched: "
                   f"{mismatched}, failed: {failed}",
        )
        ui.hint("Restart the daemon (skein restart) so cached query "
                "embeddings flush.")
    finally:
        try:
            storage.close()
        except Exception:
            pass


def _gc_impl(yes: bool, dry_run: bool) -> None:
    """Find and remove junk scopes and useless fragments.

    \b
    What counts as junk:
      • Scopes with 0 fragments AND 0 chunks (empty leftovers)
      • Personal scopes named like the user's home dir (project:<homename>)
      • Observation fragments that match `Edit on /path` / `Write on /path`
        patterns (the iteration-11 noise pattern)
      • Conversation fragments older than 30 days

    The user reviews the proposed deletes before anything happens.
    """
    from . import ui
    from .storage import Storage
    cfg = _get_config()
    storage = Storage(cfg.db_path)
    try:
        # 1. Empty scopes
        empty_scopes = []
        for scope in storage.list_scopes(limit=1000):
            n_frag = storage.count_fragments_in_scope(scope.id, include_stale=True)
            n_chunk = storage.count_chunks(scope.id)
            if n_frag == 0 and n_chunk == 0:
                empty_scopes.append(scope)

        # 2. $HOME-named scope (the project:ameliomar leak)
        home_scope_handle = f"project:{Path.home().name.lower()}"
        home_scope = storage.get_scope(home_scope_handle)

        # 3. Bare-tool-event observations (the iteration 11 leak pattern)
        import re as _re
        noise_pattern = _re.compile(
            r"^(Edit|Write|MultiEdit|NotebookEdit) on `?[^`]+`?$"
        )
        noise_frags = []
        for scope in storage.list_scopes(limit=1000):
            for f in storage.list_fragments(scope_id=scope.id,
                                            type_filter="observation",
                                            include_stale=True, limit=10000):
                if noise_pattern.match(f.content.strip()):
                    noise_frags.append(f)

        # 4. Conversation fragments older than 30 days
        from datetime import datetime as _dt, timedelta, timezone as _tz
        cutoff = (_dt.now(_tz.utc) - timedelta(days=30)).isoformat()
        old_convo_frags = []
        for scope in storage.list_scopes(limit=1000):
            for f in storage.list_fragments(scope_id=scope.id,
                                            type_filter="conversation",
                                            include_stale=True, limit=10000):
                if f.created_at < cutoff:
                    old_convo_frags.append(f)

        # ---- Report ----
        ui.section("Skein garbage collection")
        ui.blank()
        ui.fields([
            ("Empty scopes", str(len(empty_scopes))),
            ("$HOME-name scope", "1" if home_scope else "0"),
            ("Bare-tool observations", str(len(noise_frags))),
            ("Old conversations", str(len(old_convo_frags))),
        ], label_width=24)

        total = (
            len(empty_scopes) + (1 if home_scope else 0)
            + len(noise_frags) + len(old_convo_frags)
        )
        if total == 0:
            ui.blank()
            ui.bullet("Nothing to clean. Database looks healthy.")
            return

        ui.blank()
        if empty_scopes:
            console.print("  [bold]Empty scopes:[/bold]")
            for s in empty_scopes:
                console.print(f"    [dim]·[/dim] [cyan]{s.handle}[/cyan]")
        if home_scope:
            console.print(
                f"  [bold]$HOME scope:[/bold] "
                f"[cyan]{home_scope.handle}[/cyan]"
            )
        if noise_frags:
            console.print(
                f"  [bold]Bare-tool observations:[/bold] {len(noise_frags)}"
            )
        if old_convo_frags:
            console.print(
                f"  [bold]Old conversations:[/bold] {len(old_convo_frags)}"
            )

        ui.blank()
        if dry_run:
            ui.hint("--dry-run: nothing deleted.")
            return

        if not yes and not click.confirm(
            f"  Delete all {total} items?", default=False,
        ):
            ui.hint("Cancelled.")
            return

        # ---- Delete ----
        ui.blank()
        deleted_frags = 0
        deleted_scopes = 0
        for f in noise_frags:
            storage._conn.execute("DELETE FROM fragments WHERE id = ?", (f.id,))
            deleted_frags += 1
        for f in old_convo_frags:
            storage._conn.execute("DELETE FROM fragments WHERE id = ?", (f.id,))
            deleted_frags += 1
        if home_scope:
            # Wipe its fragments first so the scope is truly empty
            for f in storage.list_fragments(
                scope_id=home_scope.id, include_stale=True, limit=100000,
            ):
                storage._conn.execute("DELETE FROM fragments WHERE id = ?", (f.id,))
                deleted_frags += 1
            storage._conn.execute("DELETE FROM scopes WHERE id = ?", (home_scope.id,))
            deleted_scopes += 1
        for s in empty_scopes:
            storage._conn.execute("DELETE FROM scopes WHERE id = ?", (s.id,))
            deleted_scopes += 1
        storage._conn.commit()

        ui.step(f"Deleted {deleted_scopes} scopes, {deleted_frags} fragments",
                state="ok")
        ui.hint(
            "Run [bold]skein chunks delete-scope[/bold] for any scope "
            "that had stale chunks; this command only touches metadata."
        )
    finally:
        storage.close()


def _ingest_impl(
    path: str,
    scope: Optional[str],
    source_root: Optional[str],
    chunk_lines: int,
    overlap_lines: int,
    include_exts: Optional[str],
    extra_excludes: tuple,
    max_bytes: Optional[int],
    prune: bool,
    reset: bool,
    dry_run: bool,
    quiet: bool,
) -> None:
    """Ingest a directory of code/docs into the chunks index for RAG.

    \b
    Examples:
        skein ingest ~/Documents/myapp
        skein ingest ./src --include .py,.md --scope project:myapp
        skein ingest . --reset --prune        # full re-index
        skein ingest . --dry-run              # see what would be done

    \b
    The chunks index is searched by:
        skein search "query"
        and the MCP `search_code` tool (called by Claude Code, Cursor, etc.).
    """
    from .config import get_config
    from .embeddings import get_provider as _get_emb
    from .ingest import (
        MAX_FILE_BYTES,
        ingest_directory,
    )
    from .models import IdentityCreate, ScopeCreate
    from .storage import Storage

    cfg = get_config()
    scope_handle = _resolve_scope(scope)
    repo_path = Path(path).resolve()
    root_label = source_root or repo_path.name

    storage = Storage(cfg.db_path)
    try:
        # Auto-create scope if missing (CLI runs without daemon)
        scope_obj = storage.get_scope(scope_handle)
        if not scope_obj:
            owner = storage.get_or_create_identity(IdentityCreate(
                handle=f"user:{cfg.bearer_token[:8] if cfg.bearer_token else 'cli'}",
                type="user", name="local-user",
            ))
            scope_type = scope_handle.split(":", 1)[0] if ":" in scope_handle else "project"
            if scope_type not in {"public", "org", "team", "project", "personal"}:
                scope_type = "project"
            scope_obj = storage.create_scope(ScopeCreate(
                handle=scope_handle, type=scope_type,
                name=scope_handle.split(":", 1)[-1], owner_id=owner.id,
            ))
            console.print(f"[dim]Auto-created scope {scope_handle}[/dim]")

        if reset and not dry_run:
            n = storage.delete_chunks_by_root(scope_obj.id, root_label)
            if n:
                console.print(f"[dim]Reset: deleted {n} existing chunks under '{root_label}'[/dim]")

        # Embedding provider (best-effort; ingest still works without)
        try:
            provider = _get_emb(cfg.embedding_provider)
        except Exception as e:
            console.print(f"[yellow]⚠ Embedding provider unavailable ({e}); ingesting keyword-only.[/yellow]")
            provider = None

        # Parse include extensions
        include_set = None
        if include_exts:
            include_set = {
                ext if ext.startswith(".") else f".{ext}"
                for ext in include_exts.split(",")
            }

        max_b = max_bytes if max_bytes is not None else MAX_FILE_BYTES

        progress_cb = None
        if not quiet:
            def progress_cb(rel_path: str, stats):
                console.print(
                    f"  [dim]{stats.files_ingested:>4}[/dim] {rel_path}",
                    highlight=False,
                )

        console.print(
            f"[bold]Ingesting[/bold] {repo_path}\n"
            f"  scope:  [cyan]{scope_handle}[/cyan]\n"
            f"  root:   [cyan]{root_label}[/cyan]\n"
            f"  embed:  {cfg.embedding_provider}{' (skipped — no provider)' if provider is None else ''}\n"
            f"  chunks: {chunk_lines} lines / {overlap_lines} overlap\n"
        )

        stats = ingest_directory(
            repo_path,
            storage,
            provider,
            scope_id=scope_obj.id,
            source_root=root_label,
            chunk_lines=chunk_lines,
            overlap_lines=overlap_lines,
            include_exts=include_set,
            extra_excludes=tuple(extra_excludes),
            max_file_bytes=max_b,
            prune_missing=prune,
            dry_run=dry_run,
            progress_cb=progress_cb,
        )

        # Summary
        kb = stats.bytes_processed / 1024
        body = (
            f"  Files seen:       [bold]{stats.files_seen}[/bold]\n"
            f"  Files ingested:   [bold]{stats.files_ingested}[/bold]\n"
            f"  Files skipped:    {stats.files_skipped}\n"
            f"  Chunks inserted:  [bold green]{stats.chunks_inserted}[/bold green]\n"
            f"  Chunks updated:   [yellow]{stats.chunks_updated}[/yellow]\n"
            f"  Chunks unchanged: {stats.chunks_unchanged}\n"
            f"  Chunks pruned:    {stats.chunks_pruned}\n"
            f"  Bytes processed:  {kb:.1f} KB"
        )
        if stats.errors:
            body += f"\n\n[red]Errors ({len(stats.errors)}):[/red]"
            for err in stats.errors[:10]:
                body += f"\n  • {err}"
            if len(stats.errors) > 10:
                body += f"\n  … and {len(stats.errors) - 10} more"
        from rich.panel import Panel
        console.print(Panel(body, title="Ingest summary", expand=False))

        if stats.skipped_paths and not quiet:
            console.print(
                f"[dim]Skipped {len(stats.skipped_paths)} paths "
                f"(too large / non-utf8). First 5:[/dim]"
            )
            for s in stats.skipped_paths[:5]:
                console.print(f"  [dim]• {s}[/dim]")
    finally:
        storage.close()





@main.command(hidden=True)
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--scope", required=True, help="Scope handle to attribute chunks to.")
@click.option("--source-root", default=None,
              help="Stable source-root label. Defaults to PATH's basename.")
@click.option("--polling", is_flag=True, default=False,
              help="Force the polling backend instead of watchdog.")
def watch(path: str, scope: str, source_root: Optional[str], polling: bool) -> None:
    """Run a foreground filesystem watcher for one project.

    \b
    Internal use: ``skein up`` spawns this as a detached subprocess.
    Direct invocation is fine for debugging.
    """
    import signal as _signal
    from .config import get_config
    from .embeddings import get_provider as _get_emb
    from .storage import Storage
    from .watcher import make_watcher

    cfg = get_config()
    repo_path = Path(path).resolve()
    label = source_root or repo_path.name

    storage = Storage(cfg.db_path)
    try:
        scope_obj = storage.get_scope(scope)
        if scope_obj is None:
            err_console.print(f"[red]✗[/red] Scope '{scope}' not found.")
            sys.exit(1)

        try:
            provider = _get_emb(cfg.embedding_provider)
        except Exception as e:
            err_console.print(
                f"[yellow]⚠[/yellow] Embedding provider unavailable ({e}); "
                "watcher will skip embeddings."
            )
            provider = None

        w = make_watcher(
            root=repo_path, scope_id=scope_obj.id, source_root=label,
            storage=storage, provider=provider, force_polling=polling,
        )

        stop_now = {"flag": False}
        def _handle_signal(*_a):
            stop_now["flag"] = True
            w.stop()
        _signal.signal(_signal.SIGTERM, _handle_signal)
        _signal.signal(_signal.SIGINT, _handle_signal)

        console.print(
            f"[bold green]▶[/bold green] watching [cyan]{repo_path}[/cyan] "
            f"(scope=[cyan]{scope}[/cyan]) — Ctrl+C to stop"
        )
        w.start()

        # Block until signal
        import time as _time
        try:
            while not stop_now["flag"]:
                _time.sleep(0.5)
        finally:
            w.stop()
            console.print(
                f"[dim]watcher stopped — files reingested: "
                f"{w.stats.files_reingested}, deleted: {w.stats.files_deleted}, "
                f"errors: {w.stats.errors}[/dim]"
            )
    finally:
        storage.close()


# ---------------------------------------------------------------------------
# projects — registry of active project roots the daemon watches
# ---------------------------------------------------------------------------


@main.command(hidden=True)
@click.option("--db-path", default=None, help="Path to the SQLite database file.")
@click.option("--port", default=8765, type=int, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--embedding-provider", default=None,
              type=click.Choice(["fastembed", "openai", "bm25", "hash"]),
              help="Embedding provider. Default: fastembed (local, 384-dim, "
                   "no API key, ~130 MB one-time model download).")
@click.option("--scope", "default_scope", default="project:default",
              show_default=True,
              help="Default scope handle for this installation.")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite existing config (regenerates token).")
def init(
    db_path: Optional[str],
    port: int,
    host: str,
    embedding_provider: str,
    default_scope: str,
    force: bool,
) -> None:
    """First-time setup: generate bearer token, create config, seed default scope.

    \b
    After init:
      1. Run `skein serve` to start the daemon.
      2. Run `skein sync` in your project directory to configure LLM clients.
    """
    from .auth import generate_token
    from .config import SkeinConfig, _default_config_path

    config_path = _default_config_path()

    if config_path.exists() and not force:
        console.print(
            f"[yellow]Config already exists at {config_path}[/yellow]\n"
            f"Use [bold]--force[/bold] to regenerate."
        )
        return

    token = generate_token()
    from . import paths as _skein_paths
    effective_db = db_path or str(_skein_paths.default_db_path())

    # Default is fastembed (local, no API key). Fall back to bm25 only if
    # the library isn't importable. Never auto-pick 'hash' or 'openai' —
    # 'hash' is tests-only, 'openai' requires explicit opt-in.
    auto_upgraded = False
    if embedding_provider is None:
        from .embeddings import best_available_provider_name
        embedding_provider = best_available_provider_name()
        if embedding_provider == "fastembed":
            try:
                import importlib
                importlib.import_module("fastembed")
                auto_upgraded = True
            except ImportError:
                embedding_provider = "bm25"

    cfg = SkeinConfig({
        "port": port,
        "host": host,
        "db_path": effective_db,
        "embedding_provider": embedding_provider,
        "bearer_token": token,
        "default_scope": default_scope,
    })
    cfg.save(config_path)

    embed_note = ""
    if auto_upgraded:
        embed_note = (
            "\n[bold green]✓[/bold green] Using fastembed (local BGE-small, "
            "384-dim) — semantic search, no API key."
        )
    elif embedding_provider == "hash":
        embed_note = (
            "\n[yellow]⚠[/yellow]  Using offline 'hash' embeddings (no semantic quality).\n"
            "    Install [bold]fastembed[/bold] and run "
            "[bold]skein config set embedding_provider fastembed[/bold]."
        )

    # Seed the default scope in the DB (deferred until serve, but record it now)
    from rich.panel import Panel
    console.print(Panel.fit(
        f"[bold green]✓ Skein initialised[/bold green]\n\n"
        f"Config:  [dim]{config_path}[/dim]\n"
        f"DB:      [dim]{effective_db}[/dim]\n"
        f"Token:   [dim]{token[:16]}…[/dim]  ← keep this secret\n"
        f"Scope:   [dim]{default_scope}[/dim]\n"
        f"Embed:   [dim]{embedding_provider}[/dim]"
        f"{embed_note}\n\n"
        f"Next steps:\n"
        f"  1. [bold]skein serve[/bold]   — start the daemon\n"
        f"  2. [bold]skein sync[/bold]    — configure LLM clients\n"
        f"  3. [bold]skein hooks install[/bold]  — turn on autonomous mode",
        title="Skein Init",
    ))


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@main.command(hidden=True)
@click.option("--host", default=None, help="Override host from config.")
@click.option("--port", default=None, type=int, help="Override port from config.")
@click.option("--reload", is_flag=True, default=False, help="Auto-reload on code changes (dev mode).")
@click.option("--log-level", default=None,
              type=click.Choice(["debug", "info", "warning", "error"]))
def serve(
    host: Optional[str],
    port: Optional[int],
    reload: bool,
    log_level: Optional[str],
) -> None:
    """Start the Skein daemon (FastAPI + MCP on 127.0.0.1:8765).

    \b
    Keep this running in a dedicated terminal (or daemonize it with systemd/launchd).
    The daemon exposes:
      REST API  http://127.0.0.1:8765/v1/...
      MCP       http://127.0.0.1:8765/mcp
      Docs      http://127.0.0.1:8765/docs
    """
    import uvicorn
    from .config import get_config

    cfg = get_config()
    effective_host = host or cfg.host
    effective_port = port or cfg.port
    effective_log = log_level or cfg.log_level

    if not cfg.bearer_token:
        err_console.print(
            "[bold red]✗[/bold red] No bearer token found. "
            "Run [bold]skein init[/bold] first."
        )
        sys.exit(1)

    # Iter 35: refuse to start if another daemon already holds the
    # single-instance lock. Catches the case where a second `skein serve
    # --port N` would otherwise happily run in parallel (how the iter-30
    # 8766 leftover ran alongside 8765 for three days). Exits code 0 on
    # contention so launchd's KeepAlive=true doesn't loop.
    from . import paths as _skein_paths
    from . import single_instance as _single_instance
    _lock_handle = _single_instance.acquire_or_exit(
        _skein_paths.daemon_lock_file(), stderr=sys.stderr,
    )

    # Iter 28 Windows port: when `skein serve` is launched by a Windows
    # Scheduled Task (`schtasks /Run`), there is no console attached and
    # nothing redirects stdout/stderr. macOS' launchd plist sets
    # ``StandardOutPath`` / ``StandardErrorPath`` and systemd's unit does
    # ``StandardOutput=append:…`` — schtasks has no such knob. Replicate
    # those by tee-ing the streams to ``daemon.out`` / ``daemon.err`` under
    # ``skein_home() / "logs"`` whenever we detect we're running headless
    # (no controlling TTY). The same redirection is harmless on launchd /
    # systemd because those backends already redirect at the OS level —
    # the second redirect just appends an extra copy nobody reads.
    import sys as _sys
    if not _sys.stdout.isatty():
        from . import paths as _skein_paths
        log_dir = _skein_paths.daemon_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            out_f = open(log_dir / "daemon.out", "ab", buffering=0)
            err_f = open(log_dir / "daemon.err", "ab", buffering=0)
            os.dup2(out_f.fileno(), _sys.stdout.fileno())
            os.dup2(err_f.fileno(), _sys.stderr.fileno())
        except (OSError, ValueError):
            # Headless on a platform where dup2 of fd 1/2 doesn't work
            # (e.g. embedded interpreter). Continue without redirect.
            pass

    console.print(
        f"[bold green]▶ Starting Skein daemon[/bold green] "
        f"on [bold]http://{effective_host}:{effective_port}[/bold]"
    )
    console.print(
        f"  MCP endpoint:  [dim]http://{effective_host}:{effective_port}/mcp[/dim]\n"
        f"  API docs:      [dim]http://{effective_host}:{effective_port}/docs[/dim]\n"
        f"  Press Ctrl+C to stop."
    )

    uvicorn.run(
        "skein.server:create_app",
        host=effective_host,
        port=effective_port,
        reload=reload,
        log_level=effective_log,
        factory=True,
    )


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------




@main.command()
@click.argument("client_id", required=False)
@click.option("--all", "all_detected", is_flag=True, default=False,
              help="Connect every detected client without prompting.")
@click.option("--no-sync", is_flag=True, default=False,
              help="Update the registry but don't write configs yet.")
@click.option("--remove", "do_remove", is_flag=True, default=False,
              help="Disconnect CLIENT_ID instead of connecting it. With "
                   "--all, disconnects every currently-connected client. "
                   "Replaces `skein disconnect` per ADR-002.")
def connect(
    client_id: Optional[str],
    all_detected: bool,
    no_sync: bool,
    do_remove: bool,
) -> None:
    """Pick which installed LLM tools should share context via Skein.

    \b
    Forms:
      skein connect              interactive checklist of detected tools
      skein connect cursor       connect a single client by id
      skein connect --all        connect every detected client (CI-friendly)
      skein connect cursor --remove  disconnect a single client
      skein connect --all --remove   disconnect every connected client
    """
    # --remove dispatches into the existing disconnect handler so the
    # uninstall path stays consistent. disconnect() will be marked hidden
    # in this iter and deleted in a follow-up after a week of dogfooding.
    if do_remove:
        _disconnect_impl(client_id=client_id, all_connected=all_detected)
        return
    from . import clients as clients_mod
    from . import connections as conns
    from .config import get_config
    from .sync import sync_all
    from .agents_md import render_agents_md
    from .storage import Storage

    cfg = get_config()
    connected_ids = set(conns.get_connected_ids())

    from . import ui

    # ---- direct, non-interactive forms ----
    if client_id:
        client = clients_mod.get_client(client_id)
        if client is None:
            err_console.print(
                f"  {ui.mark('err')} Unknown client id: [cyan]{client_id}[/cyan]"
            )
            ui.hint(f"Known ids: {', '.join(clients_mod.all_ids())}")
            sys.exit(1)
        ok, note = client.detect()
        if not ok:
            err_console.print(
                f"  {ui.mark('err')} {client.display_name} does not appear to "
                f"be installed."
            )
            ui.hint(f"{note}. Install the client first, then re-run.")
            sys.exit(1)
        targets = [client_id]
    elif all_detected:
        targets = [r["id"] for r in _render_clients_table(connected_ids)[0]]
        if not targets:
            err_console.print(
                f"  {ui.mark('err')} No supported clients detected on this machine."
            )
            sys.exit(1)
    else:
        # ---- interactive checklist ----
        from . import ui
        rows, detected = _render_clients_table(connected_ids)
        not_installed = [d for d in detected if not d["detected"]]

        ui.section("Pick clients to share context")

        if not rows:
            ui.blank()
            ui.bullet("None of the supported clients were detected.")
            if not_installed:
                ui.bullet(
                    "Install one of: "
                    + ", ".join(d["display_name"] for d in not_installed),
                )
            ui.hint("Then re-run [bold]skein connect[/bold].")
            return

        ui.blank()
        # Two-column line: "  ✓ 1  Name        note"
        id_w = max(len(r["name"]) for r in rows)
        for i, row in enumerate(rows, 1):
            mk = ui.mark("ok") if row["connected"] else "[dim] [/dim]"
            console.print(
                f"  {mk} [bold]{i:>2}[/bold]  "
                f"{row['name']:<{id_w}}  [dim]{row['note']}[/dim]"
            )

        if not_installed:
            ui.blank()
            console.print(
                f"  [dim]─ Not installed: "
                f"{', '.join(d['display_name'] for d in not_installed)}[/dim]"
            )

        ui.blank()
        try:
            raw = click.prompt(
                "  Numbers (e.g. 1,3), 'all', 'none', or Enter to keep",
                default="all" if not connected_ids else "",
                show_default=False,
            )
        except (click.exceptions.Abort, EOFError, KeyboardInterrupt):
            ui.hint("Aborted.")
            return

        raw = raw.strip().lower()
        if raw == "none" or raw == "":
            ui.hint("No changes.")
            return
        if raw == "all":
            targets = [r["id"] for r in rows]
        else:
            targets = []
            for tok in raw.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    idx = int(tok)
                except ValueError:
                    err_console.print(f"  {ui.mark('err')} Not a number: {tok!r}")
                    sys.exit(1)
                if not (1 <= idx <= len(rows)):
                    err_console.print(f"  {ui.mark('err')} Out of range: {idx}")
                    sys.exit(1)
                targets.append(rows[idx - 1]["id"])

    from . import ui

    # ---- write configs (or just register) ----
    if no_sync:
        ui.blank()
        for cid in targets:
            conns.mark_connected(cid, [])
            ui.step(f"[cyan]{cid}[/cyan]", state="ok", detail="registered (no sync)")
        return

    storage = Storage(cfg.db_path)
    try:
        from .scope_resolver import auto_detect_scope
        repo_path = Path.cwd()
        scope_handle = auto_detect_scope(repo_path)
        agents_md_content = render_agents_md(
            scope_handle, storage, daemon_url=cfg.base_url,
            existing_content=(repo_path / "AGENTS.md").read_text()
                if (repo_path / "AGENTS.md").exists() else None,
        )
    finally:
        storage.close()

    result = sync_all(
        daemon_url=cfg.base_url,
        bearer_token=cfg.bearer_token,
        scope_handle=scope_handle,
        repo_path=repo_path,
        agents_md_content=agents_md_content,
        client_ids=targets,
    )

    ui.blank()
    if result.written:
        ui.section("Connected")
        ui.blank()
        for item in result.written:
            label, _, path = item.partition(": ")
            ui.step(f"[cyan]{label}[/cyan]", detail=ui.home_relative(path), state="ok")
        ui.blank()
    for item in result.errors:
        err_console.print(f"  {ui.mark('err')} {item}")
    for item in result.skipped:
        console.print(f"  {ui.mark('skip')} [dim]{item}[/dim]")







@main.command(hidden=True)
@click.argument("content")
@click.option("--type", "frag_type", required=True,
              type=click.Choice([
                  "preference", "fact", "decision", "state",
                  "observation", "requirement", "procedure", "conversation",
              ]),
              help="Fragment type.")
@click.option("--scope", default=None, help="Scope handle (default from config).")
@click.option("--territory", "-t", default=None, help="File/domain area, e.g. 'backend/auth'.")
@click.option("--tag", "-T", multiple=True, help="Add a tag (repeatable).")
@click.option("--ttl", default=None, type=int,
              help="TTL override in seconds. 0 = permanent.")
@click.option("--json", "output_json", is_flag=True, default=False)
def remember(
    content: str,
    frag_type: str,
    scope: Optional[str],
    territory: Optional[str],
    tag: tuple,
    ttl: Optional[int],
    output_json: bool,
) -> None:
    """Store a context fragment.

    \b
    Examples:
        skein remember "use Redis for session caching" --type decision
        skein remember "rate limit is 1000 req/min" --type fact --territory backend/api
        skein remember "prefer async/await over callbacks" --type preference
    """
    cfg = _get_config()
    scope_handle = _resolve_scope(scope)

    payload: dict = {
        "content": content,
        "type": frag_type,
        "scope_id": scope_handle,
        "owner_id": "",   # server fills this
        "tags": list(tag),
    }
    if territory:
        payload["territory"] = territory
    if ttl is not None:
        payload["ttl_seconds"] = ttl

    with _client() as client:
        _require_running(client)
        resp = client.post("/v1/fragments", json=payload)

    from . import ui
    if resp.status_code == 201:
        frag = resp.json()
        if output_json:
            print(json.dumps(frag, indent=2))
        else:
            ui.blank()
            preview = content if len(content) <= 60 else content[:57] + "…"
            ui.step(
                f"Stored [yellow]{frag['type']}[/yellow]",
                detail=f"{frag['id'][:8]} · {preview}",
                state="ok",
            )
            ui.blank()
    elif resp.status_code == 404 and "Scope" in resp.text:
        err_console.print(
            f"  {ui.mark('err')} Scope '[cyan]{scope_handle}[/cyan]' not found."
        )
        ui.hint(f"Create it: [bold]skein scope create {scope_handle}[/bold]")
        sys.exit(1)
    else:
        err_console.print(f"  {ui.mark('err')} Error {resp.status_code}: {resp.text}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

@main.command(hidden=True)
@click.argument("query")
@click.option("--scope", default=None, help="Scope handle (default from config).")
@click.option("--type", "types", multiple=True,
              help="Filter by fragment type (repeatable).")
@click.option("--territory", "-t", default=None)
@click.option("--limit", "-n", default=10, show_default=True)
@click.option("--json", "output_json", is_flag=True, default=False)
def recall(
    query: str,
    scope: Optional[str],
    types: tuple,
    territory: Optional[str],
    limit: int,
    output_json: bool,
) -> None:
    """Search for relevant context fragments.

    \b
    Examples:
        skein recall "caching strategy"
        skein recall "auth middleware" --type decision --type observation
        skein recall "rate limits" --territory backend/api --limit 5
    """
    cfg = _get_config()
    scope_handle = _resolve_scope(scope)

    payload: dict = {
        "query": query,
        "scope": scope_handle,
        "limit": limit,
        "include_stale": False,
    }
    if types:
        payload["types"] = list(types)
    if territory:
        payload["territory"] = territory

    with _client() as client:
        _require_running(client)
        resp = client.post("/v1/fragments/recall", json=payload)

    if resp.status_code != 200:
        err_console.print(f"[red]✗[/red] Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    if output_json:
        print(json.dumps(data, indent=2))
        return

    from . import ui
    results = data.get("results", [])
    ui.section(f"Recall: {query!r}")
    ui.blank()
    if not results:
        ui.bullet(f"No results in [cyan]{scope_handle}[/cyan].")
        ui.blank()
        return

    for r in results:
        f = r["fragment"]
        meta_parts = [f"[yellow]{f['type']}[/yellow]"]
        if f.get("territory"):
            meta_parts.append(f"[dim]{f['territory']}[/dim]")
        if f.get("tags"):
            meta_parts.append(f"[dim]#{' #'.join(f['tags'])}[/dim]")
        meta_parts.append(f"[dim]score {r['score']:.2f}[/dim]")
        console.print(
            f"  [bold cyan]{r['rank']:>2}[/bold cyan]  "
            + "  ".join(meta_parts)
        )
        console.print(f"      {f['content']}")
        console.print(
            f"      [dim]{f['id'][:8]} · {f['created_at'][:10]}[/dim]"
        )
        ui.blank()
    console.print(
        f"  [dim]{data['total']} result{'s' if data['total'] != 1 else ''} "
        f"in [cyan]{scope_handle}[/cyan][/dim]"
    )
    ui.blank()


# ---------------------------------------------------------------------------
# since (cross-tool "what changed?" feed)
# ---------------------------------------------------------------------------


def _parse_since(raw: str) -> str:
    """Turn `1h` / `2d` / `1w` or an ISO 8601 string into an ISO 8601 timestamp.

    Raises click.BadParameter on invalid input.
    """
    import re as _re
    from datetime import datetime, timedelta, timezone
    m = _re.fullmatch(r"\s*(\d+)\s*([smhdw])\s*", raw, _re.IGNORECASE)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        delta = {
            "s": timedelta(seconds=n),
            "m": timedelta(minutes=n),
            "h": timedelta(hours=n),
            "d": timedelta(days=n),
            "w": timedelta(weeks=n),
        }[unit]
        return (datetime.now(timezone.utc) - delta).isoformat()
    # Accept anything resembling ISO 8601 — let the daemon do final validation.
    if _re.match(r"^\d{4}-\d{2}-\d{2}", raw):
        return raw
    raise click.BadParameter(
        f"could not parse {raw!r}; expected ISO 8601 (2026-05-12T10:00:00) "
        "or relative form (5m, 2h, 1d, 1w)."
    )









@main.command()
@click.option("--json", "output_json", is_flag=True, default=False)
def status(output_json: bool) -> None:
    """One-screen health: daemon, watcher, clients, fragment + chunk counts.

    Per ADR-002, this is the single \"is Skein on and wired up?\" surface —
    it absorbs the old `daemon status` and `clients` commands.
    """
    from . import ui
    with _client() as client:
        try:
            resp = client.get("/health")
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            if output_json:
                print(json.dumps({"status": "offline"}))
            else:
                ui.header("Skein is offline", state="off")
                ui.hint("Run [bold]skein up[/bold] to start the daemon.")
            sys.exit(1)

    cfg = _get_config()

    # ADR-002: fold the `clients` table into status so the user sees in
    # one screen which LLM tools are actually wired through Skein.
    client_summary: list[dict] = []
    try:
        from . import clients as clients_mod
        from . import connections as conns
        connected_ids = set(conns.get_connected_ids())
        for c in clients_mod.detect_all():
            client_summary.append({
                "id": c["id"],
                "label": c.get("display_name") or c["id"],
                "detected": bool(c["detected"]),
                "connected": c["id"] in connected_ids,
            })
    except Exception:
        # status must work even if the clients module errors — best-effort.
        client_summary = []

    # ADR-002: surface the inbox depth + recent auto-gc/auto-approve counts
    # so a healthy daemon doing background work is visible at a glance.
    inbox_count = 0
    try:
        from .storage import Storage
        st = Storage(cfg.db_path)
        try:
            inbox_count = st.count_extraction_candidates(status="pending")
        finally:
            st.close()
    except Exception:
        pass

    if output_json:
        data["clients"] = client_summary
        data["inbox_pending"] = inbox_count
        print(json.dumps(data, indent=2))
        return

    db_path = str(data.get("db_path", "?")).replace(str(Path.home()), "~")
    ui.header("Skein is running", state="ok")
    ui.fields([
        ("Daemon",     f"[dim]{cfg.base_url}[/dim]"),
        ("MCP",        f"[dim]{cfg.base_url}/mcp[/dim]"),
        ("Database",   f"[dim]{db_path}[/dim]"),
    ], label_width=10)
    ui.blank()
    ui.fields([
        ("Fragments",  f"[bold]{data.get('fragment_count', 0)}[/bold]"),
        ("Scopes",     str(data.get('scope_count', 0))),
        ("Identities", str(data.get('identity_count', 0))),
        ("Inbox",      f"{inbox_count} pending"),
    ], label_width=10)
    ui.blank()

    if client_summary:
        ui.bullet("[bold]Clients[/bold]")
        for c in client_summary:
            if c["connected"]:
                marker = "[green]✓[/green]"
                tag = "connected"
            elif c["detected"]:
                marker = "[yellow]·[/yellow]"
                tag = "detected, not connected"
            else:
                marker = "[dim]·[/dim]"
                tag = "[dim]not installed[/dim]"
            console.print(f"    {marker}  {c['label']:18}  [dim]{tag}[/dim]")
        ui.blank()

    try:
        from .version_check import update_banner
        banner = update_banner()
        if banner:
            console.print(f"  {banner}")
            ui.blank()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# briefing — single-call project snapshot (LLM / human-friendly)
# ---------------------------------------------------------------------------

@main.command()
@click.option("--scope", default=None, help="Override the auto-detected scope.")
@click.option("--since", "since_arg", default=None,
              help="Show fragments created after a timestamp instead of the "
                   "default dashboard. Accepts ISO datetime or relative "
                   "(`1h`, `2d`, `2026-05-12`). Replaces `skein since`.")
@click.option("--json", "output_json", is_flag=True, default=False,
              help="Emit the raw JSON payload (LLM-friendly).")
def briefing(scope: Optional[str], since_arg: Optional[str], output_json: bool) -> None:
    """Show the project's current state in one round trip.

    Mirrors the `project_briefing` MCP tool: fragment counts by type, recent
    decisions, daemon health, and a recommended next action.

    With `--since <when>` (ADR-002), switches into the cross-tool "what
    changed?" feed — list all fragments created after the given timestamp
    so a new session can see what other agents wrote since the last one.
    """
    # --since dispatches into the existing since-command logic so the rich
    # display stays consistent. since() will be marked hidden in this iter
    # and deleted in a follow-up after a week of dogfooding.
    if since_arg is not None:
        _since_impl(since_arg=since_arg, scope=scope, types=(),
                    exclude_tool=None, limit=50, output_json=output_json)
        return
    from . import ui
    scope_handle = _resolve_scope(scope)
    with _client() as client:
        _require_running(client)
        try:
            resp = client.get("/v1/briefing", params={"scope": scope_handle})
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            err_console.print(
                f"  {ui.mark('err')} Failed to fetch briefing: {e}"
            )
            sys.exit(1)

    if output_json:
        print(json.dumps(data, indent=2))
        return

    counts = data.get("fragment_counts", {}) or {}
    daemon = data.get("daemon", {}) or {}
    recent = data.get("recent_decisions", []) or []

    uptime_s = int(daemon.get("uptime_seconds", 0) or 0)
    if uptime_s >= 3600:
        uptime_str = f"{uptime_s // 3600}h {(uptime_s % 3600) // 60}m"
    elif uptime_s >= 60:
        uptime_str = f"{uptime_s // 60}m {uptime_s % 60}s"
    else:
        uptime_str = f"{uptime_s}s"

    db_path = str(daemon.get("db_path", "?")).replace(str(Path.home()), "~")

    ui.header(f"Briefing — {data.get('scope', scope_handle)}", state="ok")
    ui.fields([
        ("Daemon",   f"v{daemon.get('version', '?')}  ·  up {uptime_str}"),
        ("Database", f"[dim]{db_path}[/dim]"),
        ("Embed",    f"{data.get('embedding_provider', '?')}"),
    ], label_width=10)
    ui.blank()

    ui.fields([
        ("Fragments", f"[bold]{data.get('fragment_total', 0)}[/bold]"),
        ("Chunks",    str(data.get("chunks_total", 0))),
        ("Inbox",     f"{data.get('active_inbox_count', 0)} pending"),
    ], label_width=10)
    ui.blank()

    # Type breakdown — only show non-zero rows so the output stays scannable.
    type_rows = [
        (k.capitalize(), str(v))
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        if v
    ]
    if type_rows:
        ui.fields(type_rows, label_width=12)
        ui.blank()

    if recent:
        ui.bullet("[bold]Recent decisions[/bold]")
        for d in recent:
            tool = d.get("created_by_tool") or "—"
            line = d.get("content_first_line") or ""
            ui.bullet(f"  [{d.get('id_short', '?')}] ({tool}) {line}")
        ui.blank()

    ui.hint(data.get("next_recommended_action", ""))


# ---------------------------------------------------------------------------
# preview — show exactly what gets injected into agent prompts
# ---------------------------------------------------------------------------












@main.command()
@click.option("--perf", "show_perf", is_flag=True, default=False,
              help="Also measure recall/search latency and chunk-index stats.")
@click.option("--clean", "do_clean", is_flag=True, default=False,
              help="Interactive cleanup: list stale chunks + expired fragments, "
                   "confirm before deleting. Replaces `skein gc` + `chunks delete-*`.")
@click.option("--reingest", "do_reingest", is_flag=True, default=False,
              help="Re-ingest the codebase at the current working directory. "
                   "Replaces `skein ingest`.")
@click.option("--reindex-embeddings", "do_reindex_embeddings", is_flag=True,
              default=False,
              help="Re-embed every fragment under the active provider — "
                   "fixes legacy fragments that return cos=0 after an "
                   "embedding-provider change.")
def doctor(show_perf: bool, do_clean: bool, do_reingest: bool,
           do_reindex_embeddings: bool) -> None:
    """Deep diagnostic: daemon, scopes, fragments, chunks, value distribution.

    Subsumes (per ADR-002) the old `daemon status`, `daemon logs`, `events`
    (snapshot), `preview`, `projects list`, `chunks stats / list / delete-*`,
    and `gc` commands. The same surface answers "is Skein healthy?" and
    "what's it actually holding?"

    Optional flags:
      --clean      run the interactive cleanup (replaces `skein gc`)
      --reingest   re-ingest a codebase path (replaces `skein ingest`)

    Checks:
      - Config file exists and token is set
      - Daemon responds
      - Claude Code has skein registered
      - AGENTS.md in cwd (if a scope is configured)
    """
    # The --clean and --reingest flags are dispatch-style: each takes a
    # specialised path that doesn't run the full diagnostic. The standard
    # doctor flow (no flags) covers the read-only health checks.
    if do_clean:
        _doctor_clean()
        return
    if do_reingest:
        _doctor_reingest()
        return
    if do_reindex_embeddings:
        _doctor_reindex_embeddings()
        return
    import shutil
    from . import ui
    from .config import _default_config_path, get_config

    cfg_path = _default_config_path()
    issues = 0

    def check(label: str, ok: bool, msg: str, fix: str = "") -> None:
        nonlocal issues
        if ok:
            ui.step(label, state="ok")
        else:
            issues += 1
            ui.step(label, state="err", detail=msg)
            if fix:
                console.print(f"        [dim]→ {fix}[/dim]")

    ui.section("Skein doctor")
    ui.blank()

    # Config
    check("Config file", cfg_path.exists(),
          f"Not found at {cfg_path}", "Run `skein init`")

    if cfg_path.exists():
        cfg = get_config()
        check("Bearer token", bool(cfg.bearer_token),
              "Token is empty", "Run `skein init --force`")
        check("DB path", bool(cfg.db_path),
              "db_path not set", "Run `skein init`")

        # Daemon
        try:
            import httpx
            resp = httpx.get(f"{cfg.base_url}/health", timeout=3)
            check("Daemon running", resp.status_code == 200,
                  f"HTTP {resp.status_code}", "Run `skein serve`")
        except Exception as e:
            check("Daemon running", False, str(e), "Run `skein serve`")

        # Claude Code
        claude_ok = False
        if shutil.which("claude"):
            try:
                result = subprocess.run(
                    ["claude", "mcp", "list"],
                    capture_output=True, text=True, timeout=10,
                )
                claude_ok = "skein" in result.stdout
            except Exception:
                pass
        check("Claude Code — skein registered", claude_ok,
              "skein not in `claude mcp list`", "Run `skein sync`")

        # AGENTS.md
        agents_md = Path.cwd() / "AGENTS.md"
        check("AGENTS.md in cwd", agents_md.exists(),
              f"Not found at {agents_md}", "Run `skein sync`")

        # Embedding provider sanity (iter 23: fastembed is the new default)
        ep = cfg.embedding_provider
        if ep == "hash":
            check("Embedding provider", False,
                  "hash provider is non-semantic — not for production use",
                  "Run `skein config set embedding_provider fastembed` for "
                  "local semantic embeddings (no API key needed)")
        elif ep == "bm25":
            ui.step("Embedding provider", state="ok",
                    detail="bm25 (keyword-only — `skein config set embedding_provider fastembed` "
                           "for local semantic search, no API key needed)")
        elif ep == "fastembed":
            ui.step("Embedding provider (fastembed, dim=384)", state="ok",
                    detail="local · BAAI/bge-small-en-v1.5 · no API key needed")
        elif ep == "openai":
            has_key = bool(os.environ.get("OPENAI_API_KEY"))
            check("Embedding provider (openai, cloud)", has_key,
                  "OPENAI_API_KEY not set in environment",
                  "export OPENAI_API_KEY=… in your shell rc, or "
                  "`skein config set embedding_provider fastembed` for the local default")
        else:
            check("Embedding provider", False,
                  f"unknown provider '{ep}'",
                  "Valid: fastembed, openai, bm25, hash")

        # Iter 23: warn if stored embeddings have a different dimension than
        # the active provider — recall results would be unreliable until
        # `skein ingest . --reset` re-embeds with the new provider.
        try:
            from .storage import Storage as _St
            _st_peek = _St(cfg.db_path)
            stored_dim = _st_peek.peek_embedding_dimension()
            from .embeddings import get_provider as _gp
            try:
                _prov = _gp(ep)
                prov_dim = getattr(_prov, "dimension", None)
            except Exception:
                prov_dim = None
            if stored_dim and prov_dim and stored_dim != prov_dim:
                check("Embedding dimension match", False,
                      f"stored dim={stored_dim} but active provider dim={prov_dim}",
                      "Re-embed with `skein ingest . --reset`")
        except Exception:
            pass

        # Chunks integrity — flag suspicious source_roots
        try:
            from .storage import Storage
            st = Storage(cfg.db_path)
            try:
                rows = st._conn.execute(
                    "SELECT source_root, COUNT(*) AS c FROM chunks "
                    "GROUP BY source_root ORDER BY c DESC"
                ).fetchall()
                suspect = []
                # Names that strongly indicate a home-dir mass-ingest leak
                # (the user's home folder name, common parent dirs, etc.).
                # Only flag these when the count is also non-trivial — a few
                # residual rows from a recent cleanup shouldn't trip doctor.
                leak_names = {Path.home().name, "Users", "Library", "Applications"}
                for r in rows:
                    root = r["source_root"]
                    count = r["c"]
                    if count > 5000:
                        suspect.append((root, count))
                    elif root in leak_names and count > 50:
                        suspect.append((root, count))
                ok = not suspect
                detail = (
                    f"{len(suspect)} suspiciously large source_root(s): "
                    + ", ".join(f"{r}={c}" for r, c in suspect[:3])
                ) if suspect else f"{len(rows)} source root(s), all reasonable size"
                check("Chunks index sanity", ok, detail,
                      "Run `skein chunks delete-root <name>` to clean up "
                      "and re-ingest with `skein up` from inside the repo")
            finally:
                st.close()
        except Exception:
            pass

        # --- Q-05 / ADR-002: value-distribution histogram + inbox depth ---
        try:
            from .storage import Storage
            st = Storage(cfg.db_path)
            try:
                rows = st._conn.execute(
                    """SELECT
                         SUM(CASE WHEN value >= 0.7 THEN 1 ELSE 0 END) AS high,
                         SUM(CASE WHEN value >= 0.4 AND value < 0.7 THEN 1 ELSE 0 END) AS mid,
                         SUM(CASE WHEN value < 0.4 THEN 1 ELSE 0 END) AS low,
                         COUNT(*) AS total
                       FROM fragments WHERE is_stale = 0"""
                ).fetchone()
                total = rows["total"] or 0
                inbox = st.count_extraction_candidates(status="pending")
                rejected = st.count_extraction_candidates(status="rejected")
                approved = st.count_extraction_candidates(status="approved")
                ui.step(
                    "Fragment-value distribution",
                    detail=(
                        f"{rows['high'] or 0} high (≥0.7) · "
                        f"{rows['mid'] or 0} mid (0.4-0.7) · "
                        f"{rows['low'] or 0} low (<0.4) · "
                        f"{total} live"
                    ),
                    state="ok",
                )
                # Oldest pending tells you whether the queue is stuck.
                oldest_pending = ""
                try:
                    row = st._conn.execute(
                        """SELECT
                             CAST((julianday('now') - julianday(MIN(created_at))) AS INTEGER) AS d
                           FROM extraction_candidates
                           WHERE status = 'pending'"""
                    ).fetchone()
                    if row and row["d"] is not None:
                        oldest_pending = f" (oldest {row['d']}d)"
                except Exception:
                    pass
                ui.step(
                    "Inbox",
                    detail=(
                        f"{inbox} pending{oldest_pending} · "
                        f"{approved} approved · {rejected} rejected (auto-sweep handles these)"
                    ),
                    state="ok" if inbox < 200 else "warn",
                )

                # --- Iter 34: "gets-better" indicators ---
                # Derived purely from existing columns — no new tables.
                # Three signals that tell the user whether usage is
                # compounding into better recall over time.
                try:
                    # Recall coverage: of the live fragments, how many
                    # have been touched by at least one recall? Median
                    # recall_hits across the touched set.
                    cov = st._conn.execute(
                        """SELECT
                             SUM(CASE WHEN recall_hits > 0 THEN 1 ELSE 0 END) AS touched,
                             COUNT(*) AS live
                           FROM fragments
                           WHERE is_stale = 0"""
                    ).fetchone()
                    touched = cov["touched"] or 0
                    live = cov["live"] or 0
                    coverage_pct = (
                        (100.0 * touched / live) if live else 0.0
                    )
                    # Cheap median: order by recall_hits, pick middle row.
                    median_hits = 0
                    if touched:
                        mh_row = st._conn.execute(
                            """SELECT recall_hits FROM fragments
                               WHERE is_stale = 0 AND recall_hits > 0
                               ORDER BY recall_hits
                               LIMIT 1 OFFSET ?""",
                            (max(0, touched // 2),),
                        ).fetchone()
                        if mh_row:
                            median_hits = int(mh_row["recall_hits"] or 0)
                    # Recalled in the last 7d — how active is the bus?
                    recent = st._conn.execute(
                        """SELECT COUNT(*) AS c FROM fragments
                           WHERE is_stale = 0
                             AND last_recalled_at IS NOT NULL
                             AND last_recalled_at >= datetime('now', '-7 days')"""
                    ).fetchone()
                    recent_recalled = int(recent["c"] or 0)
                    ui.step(
                        "Recall coverage (7d)",
                        detail=(
                            f"{touched}/{live} fragments ever recalled "
                            f"({coverage_pct:.0f}%) · median {median_hits} "
                            f"hits/fragment · {recent_recalled} active in 7d"
                        ),
                        state="ok",
                    )

                    # Inbox drain rate (7d window). Drain rate matters more
                    # than depth — a fast queue with 200 items is healthier
                    # than a stalled queue with 20.
                    drain = st._conn.execute(
                        """SELECT
                             SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) AS appr,
                             SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) AS rej,
                             SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pen
                           FROM extraction_candidates
                           WHERE created_at >= datetime('now', '-7 days')"""
                    ).fetchone()
                    d_appr = int(drain["appr"] or 0)
                    d_rej = int(drain["rej"] or 0)
                    d_pen = int(drain["pen"] or 0)
                    total7 = d_appr + d_rej + d_pen
                    if total7 == 0:
                        ui.step(
                            "Inbox drain (7d)",
                            detail="no captures in window (idle queue)",
                            state="ok",
                        )
                    else:
                        drained_pct = 100.0 * (d_appr + d_rej) / total7
                        ui.step(
                            "Inbox drain (7d)",
                            detail=(
                                f"{d_appr} approved · {d_rej} rejected · "
                                f"{d_pen} still pending of {total7} captured "
                                f"({drained_pct:.0f}% drained)"
                            ),
                            state="ok" if drained_pct >= 70 else "warn",
                        )

                    # Iter 35: recall→write rate. Of the recalls in the last
                    # 24h, how many led to a fragment back-linked via
                    # from_recall? Idle daemons read 0/0 = "no signal yet."
                    linked, total24 = st.recall_write_stats(hours=24)
                    if total24 == 0:
                        ui.step(
                            "Recall→write (24h)",
                            detail="no recalls in window (idle daemon)",
                            state="ok",
                        )
                    else:
                        pct = 100.0 * linked / total24
                        ui.step(
                            "Recall→write (24h)",
                            detail=(
                                f"{linked}/{total24} recalls produced a "
                                f"linked write ({pct:.0f}%)"
                            ),
                            state="ok",
                        )
                except Exception:
                    # Read-only diagnostics — never wedge doctor.
                    pass
            finally:
                st.close()
        except Exception:
            pass

    # --- Optional: perf measurements (`skein doctor --perf`) ---
    if show_perf and cfg_path.exists():
        import time as _time
        cfg = get_config()
        ui.blank()
        ui.section("Performance")
        ui.blank()
        try:
            client = _client(cfg.base_url, cfg.bearer_token)
        except Exception as e:
            err_console.print(f"  [red]✗[/red] Cannot reach daemon for perf checks: {e}")
        else:
            try:
                # Recall latency — three queries, take median
                from .scope_resolver import resolve_scope
                scope_handle, _src = resolve_scope(None, config_default=cfg.default_scope)
                recall_times = []
                for q in ("hello", "auth flow", "deployment"):
                    t = _time.perf_counter()
                    r = client.get("/v1/fragments/search",
                                    params={"q": q, "scope": scope_handle, "limit": 5})
                    recall_times.append((_time.perf_counter() - t) * 1000)
                recall_times.sort()
                p50 = recall_times[len(recall_times) // 2]
                p_worst = recall_times[-1]
                ui.step(
                    f"Recall (5 results, BM25+vector)",
                    detail=f"p50={p50:.0f}ms · worst={p_worst:.0f}ms",
                    state="ok" if p50 < 500 else "warn",
                )
                # Search latency
                search_times = []
                for q in ("import", "function", "config"):
                    t = _time.perf_counter()
                    r = client.get("/v1/chunks/search",
                                    params={"q": q, "scope": scope_handle, "limit": 5})
                    search_times.append((_time.perf_counter() - t) * 1000)
                search_times.sort()
                s_p50 = search_times[len(search_times) // 2]
                s_worst = search_times[-1]
                ui.step(
                    f"Code search (5 results)",
                    detail=f"p50={s_p50:.0f}ms · worst={s_worst:.0f}ms",
                    state="ok" if s_p50 < 500 else "warn",
                )
            except Exception as e:
                err_console.print(f"  [red]✗[/red] Perf probe failed: {e}")
            finally:
                try:
                    client.close()
                except Exception:
                    pass

        # Index sizing
        try:
            from .storage import Storage
            st = Storage(cfg.db_path)
            try:
                total_chunks = st._conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
                total_frags  = st._conn.execute("SELECT COUNT(*) c FROM fragments").fetchone()["c"]
                total_cands  = st.count_extraction_candidates()
                db_size = Path(cfg.db_path).stat().st_size
                ui.step(
                    "Index sizing",
                    detail=(
                        f"{total_chunks} chunks · {total_frags} fragments · "
                        f"{total_cands} inbox · DB {db_size / 1024 / 1024:.1f} MB"
                    ),
                    state="ok",
                )
            finally:
                st.close()
        except Exception as e:
            err_console.print(f"  [red]✗[/red] Index probe failed: {e}")

    ui.blank()
    if issues:
        plural = "s" if issues > 1 else ""
        console.print(f"  [red]{issues} issue{plural} found.[/red]")
    else:
        console.print("  [green]All checks passed.[/green]")

    try:
        from .version_check import update_banner
        banner = update_banner()
        if banner:
            ui.blank()
            console.print(f"  {banner}")
    except Exception:
        pass

    ui.blank()


# ---------------------------------------------------------------------------
# scope (sub-group)
# ---------------------------------------------------------------------------





@main.group(hidden=True)
def hook() -> None:
    """Hook handlers for Claude Code (and similar). Read stdin, write to stdout.

    \b
    These are not meant to be called by humans directly. They're invoked by
    Claude Code via .claude/settings.json. Run `skein hooks install` to wire
    them up automatically in a project.
    """


@hook.command("session-start")
def hook_session_start() -> None:
    """SessionStart hook handler — injects project context."""
    from .hooks import session_start
    sys.exit(session_start(sys.stdin.read() if not sys.stdin.isatty() else ""))


@hook.command("user-prompt-submit")
def hook_user_prompt_submit() -> None:
    """UserPromptSubmit hook handler — injects recall against the user's prompt."""
    from .hooks import user_prompt_submit
    sys.exit(user_prompt_submit(sys.stdin.read() if not sys.stdin.isatty() else ""))


@hook.command("stop")
def hook_stop() -> None:
    """Stop hook handler — extracts decisions from the assistant turn."""
    from .hooks import stop as hook_stop_handler
    sys.exit(hook_stop_handler(sys.stdin.read() if not sys.stdin.isatty() else ""))


@hook.command("post-tool-use")
def hook_post_tool_use() -> None:
    """PostToolUse hook handler — captures file edits as observations."""
    from .hooks import post_tool_use
    sys.exit(post_tool_use(sys.stdin.read() if not sys.stdin.isatty() else ""))


# ---------------------------------------------------------------------------
# hooks (plural) — install/uninstall/list autonomous wiring
# ---------------------------------------------------------------------------







@main.group("config")
def config_cmd() -> None:
    """View or set runtime configuration."""


@config_cmd.command("show")
def config_show() -> None:
    """Print current config (with the bearer token redacted)."""
    cfg = _get_config()
    data = cfg.to_dict()
    if data.get("bearer_token"):
        data["bearer_token"] = data["bearer_token"][:8] + "…[redacted]"
    console.print(json.dumps(data, indent=2))


@config_cmd.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a config key. Example: skein config set embedding_provider fastembed"""
    from .config import SkeinConfig, _default_config_path, load_config

    cfg = load_config()
    data = cfg.to_dict()
    if key not in data:
        err_console.print(f"[red]✗[/red] Unknown key '{key}'. Known keys:")
        for k in sorted(data):
            err_console.print(f"  {k}")
        sys.exit(1)
    # Light-touch type coercion based on the existing default
    current = data[key]
    if isinstance(current, int):
        data[key] = int(value)
    elif isinstance(current, bool):
        data[key] = value.lower() in ("1", "true", "yes")
    else:
        data[key] = value

    new_cfg = SkeinConfig(data)
    new_cfg.save(_default_config_path())
    console.print(f"[green]✓[/green] {key} = {data[key]!r}")


# ---------------------------------------------------------------------------
# archaeology — provenance-aware "where did this decision come from?"
# ---------------------------------------------------------------------------



@main.command()
@click.option("-n", "n_lines", default=20, show_default=True, type=int,
              help="Number of trailing lines to print before following.")
@click.option("--follow/--no-follow", "-f/-F", default=True, show_default=True,
              help="Keep reading new lines as they appear (Ctrl-C to stop).")
@click.option("--filter", "filter_events", default=None,
              help="Comma-separated event names to keep (e.g. 'recall,remember').")
@click.option("--scope", default=None,
              help="Only show events for this scope handle.")
@click.option("--json", "output_json", is_flag=True, default=False,
              help="Emit raw JSONL — useful for piping into jq.")
def tail(
    n_lines: int,
    follow: bool,
    filter_events: Optional[str],
    scope: Optional[str],
    output_json: bool,
) -> None:
    """Tail the Skein event log (recall, remember, supersede, leases, …).

    By default prints the last N events and then follows new ones live.
    Stop with Ctrl-C. The event log lives at ``~/.config/skein/events.jsonl``
    (override via ``SKEIN_EVENTS_PATH``).
    """
    from .events import default_path

    path = default_path()
    if not path.exists():
        err_console.print(
            f"[yellow]No event log yet at {path}.[/yellow]\n"
            "Trigger an MCP call (recall/remember/…) to seed it, "
            "or run [bold]skein up[/bold] in a project to start the watcher."
        )
        if not follow:
            return

    wanted = set()
    if filter_events:
        wanted = {x.strip() for x in filter_events.split(",") if x.strip()}

    def _format(line: str) -> Optional[str]:
        line = line.strip()
        if not line:
            return None
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return None
        if wanted and rec.get("event") not in wanted:
            return None
        if scope and rec.get("scope") != scope:
            return None
        if output_json:
            return line
        ts = rec.get("ts", "?")
        ev = rec.get("event", "?")
        sc = rec.get("scope") or "—"
        details = rec.get("details", {})
        # Compact detail rendering — keep tail readable for humans
        detail_bits = []
        for key in ("query", "preview", "glob"):
            if key in details and details[key]:
                detail_bits.append(f"{key}={details[key]!r}")
        for key in ("hits", "type", "fragment_id", "old_fragment_id", "new_fragment_id", "lease_id"):
            if key in details:
                v = details[key]
                if isinstance(v, str) and len(v) > 12:
                    v = v[:8] + "…"
                detail_bits.append(f"{key}={v}")
        detail_str = " ".join(detail_bits)
        return f"[dim]{ts}[/dim] [bold cyan]{ev:<14}[/bold cyan] [magenta]{sc:<24}[/magenta] {detail_str}"

    # --- replay last N lines ---
    try:
        with open(path, "r", encoding="utf-8") as f:
            tail_lines = f.readlines()[-n_lines:]
    except FileNotFoundError:
        tail_lines = []
    for line in tail_lines:
        formatted = _format(line)
        if formatted:
            console.print(formatted)

    if not follow:
        return

    # --- follow new appends ---
    import time as _time
    try:
        with open(path, "r", encoding="utf-8") as f:
            f.seek(0, 2)  # EOF
            while True:
                line = f.readline()
                if not line:
                    _time.sleep(0.2)
                    continue
                formatted = _format(line)
                if formatted:
                    console.print(formatted)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        # File rotated mid-follow; bail rather than spin
        err_console.print("[yellow]Event log rotated; exiting follow.[/yellow]")


# ---------------------------------------------------------------------------
# tui — Textual control-panel
# ---------------------------------------------------------------------------

@main.command()
@click.option("--scope", default=None,
              help="Scope handle (default: auto-resolve like every other command).")
def tui(scope: Optional[str]) -> None:
    """Launch the Skein control-panel TUI.

    A single-window Textual app with five tabs:

    \b
      1. Briefing  — project dashboard (default tab)
      2. Fragments — recall / hybrid search
      3. Inbox     — pending extraction candidates
      4. Events    — live event log tail
      5. Clients   — wired LLM client status

    Press [?] at any time for the chord-shortcut reference.
    """
    try:
        from .tui.app import SkeinApp
    except ImportError as e:
        err_console.print(
            f"[red]✗[/red] Could not load the TUI: {e}\n"
            "Install the `textual` extra with [bold]pip install textual>=0.80[/bold]."
        )
        sys.exit(1)
    SkeinApp(scope=scope).run()
