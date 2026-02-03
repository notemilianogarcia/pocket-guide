"""Smoke tests for dataset v2 builder (Lesson 7.2). No network calls."""

import json
import tempfile
from pathlib import Path

import pytest

from pocketguide.dataqa.build_dataset_v2 import build_dataset_v2, load_jsonl
from pocketguide.dataqa.failure_tagging import tag_failure_modes
from pocketguide.dataqa.hard_prompt_generator import generate_hard_prompts


def _make_v1_record(
    rec_id: str,
    verification_steps: list[str] | None = None,
    uncertainty_notes: str = "Rates may change; verify with official sources.",
) -> dict:
    """Build a minimal v1-style record with envelope."""
    return {
        "id": rec_id,
        "prompt": f"Create a checklist for {rec_id}.",
        "response": {
            "summary": "A short summary.",
            "assumptions": ["Assumption one."],
            "uncertainty_notes": uncertainty_notes,
            "next_steps": ["Step one."],
            "verification_steps": verification_steps if verification_steps is not None else ["Check official site."],
            "payload_type": "checklist",
            "payload": {"title": "Checklist", "groups": [{"name": "Prep", "items": [{"text": "Item"}]}]},
        },
        "category": "budgeting",
        "difficulty": "medium",
        "region_tags": ["NA"],
        "payload_type": "checklist",
    }


@pytest.fixture
def fake_v1_clean(tmp_path: Path) -> Path:
    """Create a small fake v1_clean JSONL (15 records). Some have weak verification for oversampling."""
    records = []
    for i in range(10):
        records.append(_make_v1_record(f"v1_checklist_{i:03d}"))
    # A few with missing or generic verification (match target_failure_modes)
    records.append(_make_v1_record("v1_weak_001", verification_steps=[], uncertainty_notes=""))
    records.append(_make_v1_record("v1_weak_002", verification_steps=["check online"], uncertainty_notes=""))
    records.append(_make_v1_record("v1_weak_003", verification_steps=[], uncertainty_notes="Verify later."))
    for i in range(2):
        records.append(_make_v1_record(f"v1_extra_{i:03d}"))
    path = tmp_path / "dataset_v1_clean.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def build_config_no_synthetic(tmp_path: Path, fake_v1_clean: Path) -> dict:
    """Config that does not require teacher: hard prompts + oversample, no synthetic."""
    return {
        "base_dataset_path": str(fake_v1_clean),
        "output_dir": str(tmp_path / "out"),
        "target_failure_modes": ["missing_verification_steps", "invalid_json_truncation"],
        "interventions": {
            "add_hard_prompts": True,
            "hard_prompt_count": 5,
            "add_targeted_synthetic": False,
            "synthetic_count": 0,
            "oversample_v1_matches": True,
            "oversample_factor": 1.5,
            "stricter_rejection": True,
        },
        "seeds": {"shuffle_seed": 42, "sampling_seed": 42},
    }


def test_tag_failure_modes_missing_verification() -> None:
    r = _make_v1_record("x", verification_steps=[], uncertainty_notes="")
    tags = tag_failure_modes(r)
    assert "missing_verification_steps" in tags
    assert "missing_next_steps" not in tags  # we have next_steps


def test_tag_failure_modes_generic_verification() -> None:
    r = _make_v1_record("x", verification_steps=["check online"], uncertainty_notes="Ok.")
    tags = tag_failure_modes(r)
    assert "generic_verification_only" in tags


def test_generate_hard_prompts_deterministic() -> None:
    a = generate_hard_prompts(["missing_verification_steps", "invalid_json_truncation"], 5, seed=42)
    b = generate_hard_prompts(["missing_verification_steps", "invalid_json_truncation"], 5, seed=42)
    assert len(a) == 5
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert all("target_failure_mode" in x for x in a)
    assert all("prompt" in x and "payload_type" in x for x in a)


def test_build_dataset_v2_produces_files(
    tmp_path: Path,
    fake_v1_clean: Path,
    build_config_no_synthetic: dict,
) -> None:
    """Run builder without synthetic; assert dataset_v2.jsonl, manifest, stats exist."""
    out_dir = Path(build_config_no_synthetic["output_dir"])
    project_root = Path(__file__).resolve().parents[2]
    build_config_no_synthetic["base_dataset_path"] = str(fake_v1_clean)
    build_dataset_v2(Path(fake_v1_clean), out_dir, build_config_no_synthetic, project_root)

    assert (out_dir / "dataset_v2.jsonl").exists()
    assert (out_dir / "dataset_v2_manifest.json").exists()
    assert (out_dir / "dataset_v2_stats.json").exists()


def test_build_dataset_v2_manifest_has_required_keys(
    tmp_path: Path,
    fake_v1_clean: Path,
    build_config_no_synthetic: dict,
) -> None:
    """Manifest must include v1_source_path, build_config, summary counts, intervention breakdown."""
    out_dir = Path(build_config_no_synthetic["output_dir"])
    project_root = Path(__file__).resolve().parents[2]
    manifest = build_dataset_v2(Path(fake_v1_clean), out_dir, build_config_no_synthetic, project_root)

    assert "v1_source_path" in manifest
    assert "v1_source_sha256" in manifest
    assert "build_config" in manifest
    assert "summary" in manifest
    summary = manifest["summary"]
    assert "num_v1_kept" in summary
    assert "num_v1_removed" in summary
    assert "num_oversampled" in summary
    assert "num_hard_prompts_generated" in summary
    assert "num_synthetic_generated" in summary
    assert "intervention_breakdown_by_target_failure_mode" in manifest


def test_build_dataset_v2_traceability(
    tmp_path: Path,
    fake_v1_clean: Path,
    build_config_no_synthetic: dict,
) -> None:
    """Every v2 record must have source; v1/oversampled have source_id; synthetic have recipe_id, teacher_model, created_at, target_failure_mode."""
    out_dir = Path(build_config_no_synthetic["output_dir"])
    project_root = Path(__file__).resolve().parents[2]
    build_dataset_v2(Path(fake_v1_clean), out_dir, build_config_no_synthetic, project_root)

    records = load_jsonl(out_dir / "dataset_v2.jsonl")
    assert len(records) > 0
    for r in records:
        assert "source" in r, f"Missing source in {r.get('id')}"
        if r["source"] == "v1":
            assert "source_id" in r
        elif r["source"] == "synthetic_v2":
            assert "recipe_id" in r and "teacher_model" in r and "created_at" in r and "target_failure_mode" in r


def test_build_dataset_v2_deterministic(
    tmp_path: Path,
    fake_v1_clean: Path,
    build_config_no_synthetic: dict,
) -> None:
    """Two runs with same config and seed produce same record count and same first/last ids."""
    out_dir = Path(build_config_no_synthetic["output_dir"])
    project_root = Path(__file__).resolve().parents[2]

    build_dataset_v2(Path(fake_v1_clean), out_dir, build_config_no_synthetic, project_root)
    records1 = load_jsonl(out_dir / "dataset_v2.jsonl")
    ids1 = [r["id"] for r in records1]

    # Second run (same seed)
    out_dir2 = tmp_path / "out2"
    build_config_no_synthetic["output_dir"] = str(out_dir2)
    build_dataset_v2(Path(fake_v1_clean), out_dir2, build_config_no_synthetic, project_root)
    records2 = load_jsonl(out_dir2 / "dataset_v2.jsonl")
    ids2 = [r["id"] for r in records2]

    assert len(records1) == len(records2)
    assert set(ids1) == set(ids2), "Same set of ids in both runs"
