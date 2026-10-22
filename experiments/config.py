"""Central configuration loading for experiments (CP7.1)."""
import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "default.yaml"


def deep_merge(base, override):
    """Return a copy of base with override's keys merged in recursively."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_override(text):
    """Turn 'model.lr=0.01' into {'model': {'lr': 0.01}}; values are parsed as YAML."""
    if "=" not in text:
        raise ValueError(f"Override must look like section.key=value, got: {text!r}")
    dotted, raw = text.split("=", 1)
    node = yaml.safe_load(raw)
    for part in reversed(dotted.strip().split(".")):
        node = {part: node}
    return node


def load_config(path=None, overrides=()):
    """Load the default config, then an optional experiment config, then CLI overrides."""
    with open(DEFAULT_CONFIG) as f:
        config = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path) as f:
            config = deep_merge(config, yaml.safe_load(f) or {})
    for text in overrides:
        config = deep_merge(config, parse_override(text))
    return config
