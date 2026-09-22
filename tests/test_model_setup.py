"""Installer checks use tiny synthetic bytes and never contact a network/model."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "model_setup", Path(__file__).resolve().parents[1] / "scripts/download_encoder.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def artifact(data=b"tiny synthetic weights", name="model.bin"):
    return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "url": "https://example.invalid/pinned/" + name}


class Response:
    def __init__(self, data, status=200, headers=None):
        self.data, self.status, self.headers = data, status, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        pass

    def read(self, size):
        value, self.data = self.data[:size], self.data[size:]
        return value


class Opener:
    def __init__(self, response):
        self.response, self.requests = response, []

    def open(self, request, timeout):
        self.requests.append(request)
        return self.response


def test_offline_copy_and_idempotent_verify(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    content = b"tiny synthetic weights"
    (source / "model.bin").write_bytes(content)
    item = artifact(content)
    setup.install_files([item], destination, source=source)
    setup.install_files([item], destination, verify_only=True)
    assert (destination / "model.bin").read_bytes() == content
    assert not (destination / "model.bin.part").exists()


def test_corrupt_source_does_not_publish(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Offline source"):
        setup.install_files([artifact()], tmp_path / "destination", source=source)
    assert not (tmp_path / "destination/model.bin").exists()


@pytest.mark.parametrize("path", ["../secret", "/absolute", "C:/secret", "file:stream", "a\\secret"])
def test_artifact_path_escape_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        setup.install_files([artifact(name=path)], tmp_path, verify_only=True)


def test_resume_requires_correct_content_range(tmp_path):
    data = b"abcdefgh"
    target = tmp_path / "model.bin"
    target.with_suffix(".bin.part").write_bytes(data[:3])
    opener = Opener(Response(data[3:], 206, {"Content-Range": "bytes 3-7/8"}))
    setup.fetch(artifact(data), target, opener=opener)
    assert opener.requests[0].get_header("Range") == "bytes=3-"
    assert target.read_bytes() == data


def test_bad_resume_cannot_publish(tmp_path):
    target = tmp_path / "model.bin"
    target.with_suffix(".bin.part").write_bytes(b"abc")
    opener = Opener(Response(b"defgh", 206, {"Content-Range": "bytes 4-7/8"}))
    with pytest.raises(ValueError, match="range"):
        setup.fetch(artifact(b"abcdefgh"), target, opener=opener)
    assert not target.exists()


def test_server_ignoring_range_restarts_safely(tmp_path):
    target = tmp_path / "model.bin"
    target.with_suffix(".bin.part").write_bytes(b"abc")
    setup.fetch(artifact(b"abcdefgh"), target, opener=Opener(Response(b"abcdefgh")))
    assert target.read_bytes() == b"abcdefgh"


def test_corrupt_download_is_not_published(tmp_path):
    target = tmp_path / "model.bin"
    with pytest.raises(ValueError, match="SHA-256"):
        setup.fetch(artifact(b"abcdefgh"), target, opener=Opener(Response(b"abcdxxxx")))
    assert not target.exists()


def test_oversize_download_stops_before_publish(tmp_path):
    target = tmp_path / "model.bin"
    with pytest.raises(ValueError, match="exceeds"):
        setup.fetch(artifact(b"abc"), target, opener=Opener(Response(b"abcdef")))
    assert not target.exists()


def test_existing_wrong_file_is_preserved(tmp_path):
    target = tmp_path / "model.bin"
    target.write_bytes(b"existing user content")
    with pytest.raises(ValueError, match="move it aside"):
        setup.install_files([artifact()], tmp_path, verify_only=True)
    assert target.read_bytes() == b"existing user content"


def test_no_https_downgrade(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        setup.fetch({**artifact(), "url": "http://example.invalid/model"}, tmp_path / "model.bin")
    with pytest.raises(ValueError, match="HTTPS"):
        setup.HttpsRedirect().redirect_request(None, None, 302, "Found", {}, "http://example.invalid/model")


def test_optional_audit_reports_missing_installation(tmp_path, monkeypatch):
    from assistant_runtime.service import AuditError, AuditService
    monkeypatch.setenv("SISU_READER_AUDIT_PYTHON", str(tmp_path / "missing-python"))
    monkeypatch.setenv("SISU_READER_AUDIT_MANIFEST", str(tmp_path / "missing-manifest.json"))
    service = AuditService.__new__(AuditService)
    with pytest.raises(AuditError) as error:
        service._default_client()
    assert error.value.code == "audit_unavailable"
    assert "OPTIONAL_CITATION_AUDIT.md" in error.value.message


def test_encoder_explicit_path_precedes_environment(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sisu_reader.hybrid_retrieval import HybridIndex
    configured, override = tmp_path / "env-model", tmp_path / "explicit-model"
    monkeypatch.setenv("SISU_READER_ENCODER_PATH", str(configured))
    config = SimpleNamespace(workspace_dir=tmp_path)
    assert HybridIndex(config, None).model_path == configured
    assert HybridIndex(config, None, model_path=override).model_path == override
