"""Unit tests for the model-library I/O helpers (pure, no Qt)."""
import os

from syncmoss.Library_io import (
    parse_library_model_file,
    compute_versioned_title_if_needed,
    library_model_sort_key,
)


def _write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_parse_library_model_file_metadata_and_models(tmp_path):
    mdl = tmp_path / "Hematite_sample.mdl"
    _write(
        mdl,
        "#@Chemical composition Fe2O3\n"
        "#@Temperature (K) 300\n"
        "#@Comment a test sample\n"
        "baseline\tSextet\n",
    )
    result = parse_library_model_file(str(mdl))

    assert result["title"] == "Hematite_sample"
    assert result["file_name"] == "Hematite_sample.mdl"
    assert result["metadata"]["Chemical composition"] == "Fe2O3"
    assert result["metadata"]["Temperature (K)"] == "300"
    assert "a test sample" in result["comments"]
    assert "Sextet" in result["models"]


def test_compute_versioned_title_no_collision(tmp_path):
    title, idx = compute_versioned_title_if_needed(str(tmp_path), "Sample")
    assert title == "Sample"
    assert idx is None


def test_compute_versioned_title_with_collisions(tmp_path):
    _write(tmp_path / "Sample.mdl", "x")
    title, idx = compute_versioned_title_if_needed(str(tmp_path), "Sample")
    assert title == "Sample (2)"
    assert idx == 2

    _write(tmp_path / "Sample (2).mdl", "x")
    title, idx = compute_versioned_title_if_needed(str(tmp_path), "Sample")
    assert title == "Sample (3)"
    assert idx == 3


def test_compute_versioned_title_empty_title(tmp_path):
    title, idx = compute_versioned_title_if_needed(str(tmp_path), "")
    assert title == ""
    assert idx is None


def test_library_model_sort_key_orders_versions():
    names = ["test (3)", "beta", "test", "test (2)", "alpha"]
    ordered = sorted(names, key=library_model_sort_key)
    assert ordered == ["alpha", "beta", "test", "test (2)", "test (3)"]


def test_library_model_sort_key_ignores_extension():
    # Extension should not affect ordering of the base title.
    assert library_model_sort_key("alpha.mdl")[0] == "alpha"
