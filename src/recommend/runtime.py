"""One composition point for P4 explanations, customer memory and the P5 bundle."""
import os
import hashlib
import json
from pathlib import Path

from src.common import config
from src.llm.explain import explain, explain_with_provenance
from src.memory.store import MemoryStore
from .service import RecommendationService


def load_runtime_service(path=None):
    supplied = path or os.getenv('GREEN_BUNDLE_PATH')
    if supplied:
        backend = RecommendationService.from_file(supplied)
    else:
        generated = config.PROCESSED_DIR / 'team_bundle.json'
        preview = config.DATA_DIR / 'preview' / 'team_bundle.json'
        if generated.exists():
            backend = RecommendationService.from_file(generated)
        elif preview.exists():
            backend = RecommendationService.from_file(preview)
        elif os.getenv('GREEN_ALLOW_DEMO') == '1':
            backend = RecommendationService.demo()
        else:
            raise FileNotFoundError('No team bundle: run the pipeline or explicitly set GREEN_ALLOW_DEMO=1')
    if backend.insight_provider is None:
        backend.insight_provider = explain_for_service
    if backend.memory_store is None:
        fingerprint = hashlib.sha256(json.dumps(backend.bundle, sort_keys=True, allow_nan=False).encode()).hexdigest()[:12]
        default_path = config.DATA_DIR / 'memory' / f'{backend.mode.lower()}-{fingerprint}.db'
        backend.memory_store = MemoryStore(os.getenv('GREEN_MEMORY_PATH') or default_path)
    return backend


def explain_for_service(row):
    try:
        return explain_with_provenance(row)
    except Exception:
        from src.llm.templates import template_explanation
        from src.llm.validator import validate
        fallback = template_explanation(row)
        if validate(fallback, row)['ok']:
            return {**fallback, 'provenance': {'text_source': 'TEMPLATE'}}
        return None
