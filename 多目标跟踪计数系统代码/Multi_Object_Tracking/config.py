import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

try:
    import yaml
except Exception:
    yaml = None

from .tracker.deepsort_mod import DeepSortArgs


def load_deepsort_args(path: str) -> Optional[DeepSortArgs]:
    p = Path(path)
    if not p.exists():
        return None
    data = None
    try:
        if p.suffix.lower() in {'.yml', '.yaml'} and yaml is not None:
            with p.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        elif p.suffix.lower() == '.json':
            with p.open('r', encoding='utf-8') as f:
                data = json.load(f)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return None
    base = DeepSortArgs()
    base_dict = asdict(base)
    for k, v in data.items():
        if k in base_dict:
            setattr(base, k, v)
    return base
