import json
import os
from pathlib import Path
import requests

CONFIG_DIR = Path.home() / ".config" / "company-brain"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def save_config(config):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_api_url():
    return os.environ.get("BRAIN_API_URL") or load_config().get("api_url", "http://localhost:8000")


def get_api_key():
    return os.environ.get("BRAIN_API_KEY") or load_config().get("api_key", "")


def get_headers():
    key = get_api_key()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["x-api-key"] = key
    return headers


def list_skills():
    url = f"{get_api_url()}/agent/skills"
    r = requests.get(url, headers=get_headers())
    r.raise_for_status()
    return r.json()


def get_skill(name):
    url = f"{get_api_url()}/agent/skills/{name}"
    r = requests.get(url, headers=get_headers())
    r.raise_for_status()
    return r.json()


def search_skills(query, limit=10):
    url = f"{get_api_url()}/agent/context"
    r = requests.post(url, json={"intent": query, "limit": limit}, headers=get_headers())
    r.raise_for_status()
    return r.json()
