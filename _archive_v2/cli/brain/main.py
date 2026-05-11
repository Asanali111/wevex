import json
import os
import re
from pathlib import Path

import click
import yaml

from .api import (
    get_api_url, get_api_key, save_config, load_config,
    list_skills as api_list_skills,
    get_skill as api_get_skill,
    search_skills as api_search_skills,
)

SKILLS_DIR = Path(os.getcwd()) / ".brain" / "skills"


def get_skill_path(name: str) -> Path:
    return SKILLS_DIR / name / "SKILL.md"


def parse_skill(content: str) -> dict:
    """Parse a SKILL.md file into frontmatter + body."""
    if not content.startswith("---"):
        return {"frontmatter": {}, "body": content}
    
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {"frontmatter": {}, "body": content}
    
    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        frontmatter = {}
    
    body = parts[2].strip()
    return {"frontmatter": frontmatter, "body": body}


def render_skill(frontmatter: dict, body: str) -> str:
    """Render a SKILL.md file from frontmatter + body."""
    fm = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return f"---\n{fm}---\n\n{body}\n"


@click.group()
@click.version_option(version=click.__version__, prog_name="brain")
def cli():
    """Company Brain CLI - manage agent skills locally and sync with cloud."""
    pass


@cli.command()
@click.option("--api-url", help="API URL for the Company Brain instance")
@click.option("--api-key", help="API key for agent access")
def init(api_url, api_key):
    """Initialize a local brain directory and configure API access."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    
    config = load_config()
    if api_url:
        config["api_url"] = api_url
    if api_key:
        config["api_key"] = api_key
    
    if api_url or api_key:
        save_config(config)
    
    click.echo(f"✓ Initialized at {SKILLS_DIR}")
    click.echo(f"  API URL: {get_api_url()}")
    click.echo(f"  API Key: {'configured' if get_api_key() else 'not set'}")


@cli.command()
@click.argument("name")
@click.option("--description", default="", help="Skill description")
@click.option("--tag", multiple=True, help="Tags for the skill")
def create(name, description, tag):
    """Create a new skill locally."""
    if not re.match(r"^[a-z0-9-]+$", name):
        click.echo("Error: name must be kebab-case (lowercase, hyphens, numbers only)", err=True)
        return
    
    skill_dir = SKILLS_DIR / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    
    frontmatter = {
        "name": name,
        "description": description or f"Procedure for {name}",
        "author": "@local",
        "version": "1.0.0",
        "status": "draft",
        "tags": list(tag),
        "requires_approval": False,
    }
    
    body = """## Trigger
- [When does this procedure apply?]

## Prerequisites
- [Required tools or permissions]

## Logical Steps
1. **[Step name]**: [Instruction]
   - Validation: [How to verify]

## Constraints
- [Rules the agent must follow]
"""
    
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(render_skill(frontmatter, body))
    
    click.echo(f"✓ Created skill: {skill_path}")
    
    # Open in editor
    editor = os.environ.get("EDITOR", "nano")
    click.echo(f"  Opening in {editor}...")
    os.system(f"{editor} '{skill_path}'")


@cli.command()
def list():
    """List all local skills."""
    if not SKILLS_DIR.exists():
        click.echo("No skills directory found. Run 'brain init' first.")
        return
    
    skills = []
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        
        parsed = parse_skill(skill_file.read_text())
        fm = parsed["frontmatter"]
        skills.append({
            "name": fm.get("name", skill_dir.name),
            "status": fm.get("status", "unknown"),
            "version": fm.get("version", "?"),
            "description": fm.get("description", "")[:60],
        })
    
    if not skills:
        click.echo("No local skills found.")
        return
    
    click.echo(f"\n{'NAME':<30} {'STATUS':<12} {'VERSION':<8} DESCRIPTION")
    click.echo("-" * 80)
    for s in skills:
        click.echo(f"{s['name']:<30} {s['status']:<12} {s['version']:<8} {s['description']}")


@cli.command()
@click.argument("query")
@click.option("--limit", default=5, help="Maximum results")
def search(query, limit):
    """Search local skills by name or description."""
    if not SKILLS_DIR.exists():
        click.echo("No skills directory found.")
        return
    
    query_lower = query.lower()
    matches = []
    
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        
        parsed = parse_skill(skill_file.read_text())
        fm = parsed["frontmatter"]
        name = fm.get("name", "")
        desc = fm.get("description", "")
        content = parsed["body"]
        
        if query_lower in name.lower() or query_lower in desc.lower() or query_lower in content.lower():
            matches.append({
                "name": name,
                "status": fm.get("status", "unknown"),
                "description": desc,
            })
    
    if not matches:
        click.echo(f"No local skills match '{query}'")
        return
    
    click.echo(f"\nFound {len(matches)} local skill(s) matching '{query}':\n")
    for m in matches:
        click.echo(f"  {m['name']} [{m['status']}]")
        click.echo(f"    {m['description'][:80]}\n")


@cli.command()
@click.argument("name")
def show(name):
    """Display a local skill."""
    skill_path = get_skill_path(name)
    if not skill_path.exists():
        click.echo(f"Skill '{name}' not found locally.")
        return
    
    click.echo(skill_path.read_text())


@cli.command()
def push():
    """Push local skills to cloud (requires API key)."""
    if not get_api_key():
        click.echo("Error: API key not configured. Run 'brain init --api-key <key>'", err=True)
        return
    
    click.echo("Push not implemented in MVP. Use the web UI to create skills.")


@cli.command()
def pull():
    """Pull skills from cloud to local directory."""
    if not get_api_key():
        click.echo("Error: API key not configured. Run 'brain init --api-key <key>'", err=True)
        return
    
    try:
        skills = api_list_skills()
        click.echo(f"Found {len(skills)} skill(s) in cloud.\n")
        
        for skill in skills:
            name = skill.get("name", "unknown")
            skill_dir = SKILLS_DIR / name
            skill_dir.mkdir(parents=True, exist_ok=True)
            
            frontmatter = {
                k: v for k, v in skill.items()
                if k not in ("id", "team_id", "embedding", "created_at", "updated_at", "content")
            }
            body = skill.get("content", "")
            
            skill_path = skill_dir / "SKILL.md"
            skill_path.write_text(render_skill(frontmatter, body))
            click.echo(f"  ✓ {name}")
        
        click.echo(f"\nSynced to {SKILLS_DIR}")
    except Exception as e:
        click.echo(f"Error pulling skills: {e}", err=True)


@cli.command()
@click.argument("intent")
@click.option("--limit", default=3, help="Maximum skills to retrieve")
def context(intent, limit):
    """Get agent context for an intent (requires API key)."""
    if not get_api_key():
        click.echo("Error: API key not configured. Run 'brain init --api-key <key>'", err=True)
        return
    
    try:
        result = api_search_skills(intent, limit)
        skills = result.get("skills", [])
        
        click.echo(f"\nIntent: '{intent}'")
        click.echo(f"Found {len(skills)} relevant skill(s):\n")
        
        for i, skill in enumerate(skills, 1):
            sim = round(skill.get("similarity", 0) * 100)
            click.echo(f"[{i}] {skill['name']} ({sim}% match)")
            click.echo(f"    {skill['description'][:80]}\n")
    except Exception as e:
        click.echo(f"Error retrieving context: {e}", err=True)


if __name__ == "__main__":
    cli()
