"""Tests for release version normalization and bumping."""

from headroom.release_version import compute_release_version, normalize_release_tag


def test_normalize_release_tag_preserves_three_part_tag() -> None:
    assert str(normalize_release_tag("v0.5.20")) == "0.5.20"


def test_normalize_release_tag_collapses_four_part_tag() -> None:
    assert str(normalize_release_tag("v0.5.25.2")) == "0.5.27"


def test_compute_patch_release_from_four_part_history() -> None:
    info = compute_release_version(
        canonical_version="0.5.25",
        level="patch",
        tags=["v0.5.20", "v0.5.25.1", "v0.5.25.2"],
    )

    assert info.version == "0.5.28"
    assert info.npm_version == "0.5.28"
    assert info.previous_tag == "v0.5.25.2"
    assert info.bump == "patch"


def test_compute_minor_release_from_four_part_history() -> None:
    info = compute_release_version(
        canonical_version="0.5.25",
        level="minor",
        tags=["v0.5.20", "v0.5.25.1", "v0.5.25.2"],
    )

    assert info.version == "0.6.0"
    assert info.npm_version == "0.6.0"
    assert info.previous_tag == "v0.5.25.2"
    assert info.bump == "minor"


def test_compute_patch_release_from_canonical_without_tags() -> None:
    info = compute_release_version(
        canonical_version="0.5.25",
        level="patch",
        tags=[],
    )

    assert info.version == "0.5.26"
    assert info.npm_version == "0.5.26"
    assert info.previous_tag == ""


def test_manual_version_override_uses_single_semver() -> None:
    info = compute_release_version(
        canonical_version="0.5.25",
        level="patch",
        tags=["v0.5.25.2"],
        manual_version="0.6.0",
    )

    assert info.version == "0.6.0"
    assert info.npm_version == "0.6.0"
    assert info.previous_tag == ""
    assert info.bump == "manual"
