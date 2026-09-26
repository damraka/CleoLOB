import json

import pytest

from lob import evidence
from lob.artifacts import seal_artifacts
from lob.experiments.registry import sha256_file


@pytest.fixture
def registered(tmp_path, monkeypatch):
    source = tmp_path / "checkout"
    source.mkdir()
    (source / "implementation.py").write_text("# retained implementation\n")
    files = {"implementation.py": sha256_file(source / "implementation.py")}
    monkeypatch.setattr(evidence, "PROJECT_ROOT", source)
    monkeypatch.setattr(evidence, "source_manifest", lambda: files)
    monkeypatch.setattr(evidence, "portable_provenance", lambda: {
        "source_files": files, "source_sha256": evidence.document_hash(files),
        "git_commit": None, "runtime": {"python": "fixture"}})
    root = tmp_path / "run"
    kwargs = {"kind": "synthetic_contract", "config": {"size": 8},
              "inputs": [{"identity": "generator-v1", "adapter_version": "1", "role": "test",
                          "kind": "synthetic", "sha256": evidence.document_hash({"generator": 1})}],
              "seeds": {"test": [42]}, "hypotheses": ["No historical evidence is inferred."]}
    evidence.register_evidence(root, **kwargs)
    return root, kwargs


def test_registered_failure_is_retained_and_verifiable(registered):
    root, kwargs = registered
    result = evidence.finalize_evidence(root, status="FAILED", metrics={"completed": 0})
    assert result["status"] == "FAILED"
    assert evidence.verify_evidence(root)["valid"]
    with pytest.raises(FileExistsError):
        evidence.register_evidence(root, **kwargs)
    with pytest.raises(FileExistsError):
        evidence.finalize_evidence(root, status="ESTABLISHED", metrics={})


def test_registration_rejects_source_and_config_drift(registered, monkeypatch):
    root, _ = registered
    monkeypatch.setattr(evidence, "source_manifest", lambda: {"implementation.py": "f" * 64})
    with pytest.raises(ValueError, match="implementation changed"):
        evidence.finalize_evidence(root, status="INCONCLUSIVE", metrics={})
    (root / "config.json").write_text('{"size":9}')
    with pytest.raises(ValueError, match="registration seal"):
        evidence.finalize_evidence(root, status="INCONCLUSIVE", metrics={})


def test_semantic_binding_catches_result_mismatch_even_with_new_byte_seal(registered):
    root, _ = registered
    evidence.finalize_evidence(root, status="NOT_ESTABLISHED", metrics={})
    path = root / "result.json"
    result = json.loads(path.read_text())
    result["config_sha256"] = "0" * 64
    path.write_text(json.dumps(result))
    (root / "checksums.json").unlink()
    seal_artifacts(root)
    verified = evidence.verify_evidence(root)
    assert not verified["valid"]
    assert "config_sha256" in verified["issues"][0]


def test_checkpoint_identity_and_stale_hash(registered):
    root, _ = registered
    checkpoint = root / "checkpoint.bin"
    checkpoint.write_bytes(b"fixed final model")
    model = {"identity": "policy/seed42/final", "path": checkpoint.name, "sha256": sha256_file(checkpoint)}
    checkpoint.write_bytes(b"later replacement")
    with pytest.raises(ValueError, match="stale"):
        evidence.finalize_evidence(root, status="NOT_ESTABLISHED", metrics={}, models=[model])
    model["sha256"] = sha256_file(checkpoint)
    evidence.finalize_evidence(root, status="NOT_ESTABLISHED", metrics={}, models=[model])
    checkpoint.unlink()
    assert not evidence.verify_evidence(root)["valid"]


@pytest.mark.parametrize("name", ["../model", "C:/model", "a\\b", "/tmp/model", "a/./b"])
def test_checkpoint_paths_cannot_escape(registered, name):
    root, _ = registered
    with pytest.raises(ValueError, match="relative"):
        evidence.finalize_evidence(root, status="INVALID", metrics={}, models=[{
            "identity": "policy", "path": name, "sha256": "0" * 64}])


def test_missing_registration_is_invalid(tmp_path):
    assert not evidence.verify_evidence(tmp_path)["valid"]


def test_one_model_identity_cannot_resolve_to_two_checkpoints(registered):
    root, _ = registered
    models = []
    for name in ("first.bin", "second.bin"):
        path = root / name
        path.write_bytes(name.encode())
        models.append({"identity": "same-final-model", "path": name, "sha256": sha256_file(path)})
    with pytest.raises(ValueError, match="identities must be unique"):
        evidence.finalize_evidence(root, status="INCONCLUSIVE", metrics={}, models=models)


def test_nan_metrics_and_ambiguous_status_refused(registered):
    root, _ = registered
    with pytest.raises(ValueError, match="status"):
        evidence.finalize_evidence(root, status="good", metrics={})
    with pytest.raises(ValueError):
        evidence.finalize_evidence(root, status="INVALID", metrics={"mean": float("nan")})
