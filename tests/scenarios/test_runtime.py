import pytest
from src.common import config
from src.recommend.runtime import load_runtime_service


def test_fresh_checkout_loads_committed_trained_preview(tmp_path, monkeypatch):
    monkeypatch.delenv('GREEN_BUNDLE_PATH', raising=False)
    monkeypatch.setenv('GREEN_MEMORY_PATH', ':memory:')
    monkeypatch.setattr(config, 'PROCESSED_DIR', tmp_path)
    service = load_runtime_service()
    assert service.mode == 'TEAM_OUTPUTS'
    assert service.bundle['bundle_scope'] == 'PREVIEW'
    assert len(service.customers) == 213
    assert all(not p['model_version'].startswith('mock_')
               for rows in service.bundle['predictions'].values() for p in rows.values())


def test_explicit_invalid_bundle_does_not_use_preview(tmp_path, monkeypatch):
    monkeypatch.setenv('GREEN_BUNDLE_PATH', str(tmp_path / 'missing.json'))
    with pytest.raises(FileNotFoundError):
        load_runtime_service()


def test_no_bundle_requires_explicit_demo_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv('GREEN_BUNDLE_PATH', raising=False)
    monkeypatch.delenv('GREEN_ALLOW_DEMO', raising=False)
    monkeypatch.setenv('GREEN_MEMORY_PATH', ':memory:')
    monkeypatch.setattr(config, 'PROCESSED_DIR', tmp_path)
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
    with pytest.raises(FileNotFoundError, match='No team bundle'):
        load_runtime_service()
    monkeypatch.setenv('GREEN_ALLOW_DEMO', '1')
    assert load_runtime_service().mode == 'DEMO_MOCK'
