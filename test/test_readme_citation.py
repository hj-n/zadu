"""Contracts for the manually maintained, versioned README citations."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
LATEST_MARKER = re.compile(r"<!-- zadu-citation-latest: ([0-9]+\.[0-9]+\.[0-9]+) -->")
RELEASE_SUMMARY = re.compile(
    r"<summary><strong>ZADU ([0-9]+\.[0-9]+\.[0-9]+)</strong></summary>"
)
HISTORICAL_RELEASES = {
    "0.1.0",
    "0.1.1",
    "0.2.0",
    "0.2.1",
    "0.3.0",
    "0.3.1",
    "0.4.1",
    "0.4.2",
    "0.5.0",
    "0.5.1",
    "0.5.2",
    "0.5.3",
}


def _project_version() -> str:
    project = re.search(
        r"(?ms)^\[project\]\s*$.*?^version\s*=\s*\"([^\"]+)\"", PYPROJECT
    )
    assert project is not None, "pyproject.toml needs [project].version"
    return project.group(1)


def _release_block(version: str) -> str:
    pattern = re.compile(
        rf"<summary><strong>ZADU {re.escape(version)}</strong></summary>"
        r"(?P<body>.*?)</details>",
        re.DOTALL,
    )
    match = pattern.search(README)
    assert match is not None, f"README needs a collapsed citation for ZADU {version}"
    return match.group("body")


def _authors(version: str) -> set[str]:
    block = _release_block(version)
    match = re.search(r"author = \{(?P<authors>.*?)\},", block, re.DOTALL)
    assert match is not None, f"ZADU {version} citation needs an author field"
    return {name.strip() for name in match.group("authors").split(" and ")}


def _details_block(summary: str) -> str:
    opening = f"<details>\n<summary><strong>{summary}</strong></summary>"
    start = README.index(opening)
    depth = 0
    for tag in re.finditer(r"<details>|</details>", README[start:]):
        depth += 1 if tag.group() == "<details>" else -1
        if depth == 0:
            return README[start : start + tag.end()]
    raise AssertionError(f"The {summary} disclosure must be closed")


def test_current_package_version_has_the_first_release_citation():
    version = _project_version()
    marker = LATEST_MARKER.search(README)
    assert marker is not None, "README needs a zadu-citation-latest marker"
    assert marker.group(1) == version, (
        "A release version change must update the README citation marker and add "
        "the new version's collapsed BibTeX entry"
    )

    versions = RELEASE_SUMMARY.findall(README)
    assert versions
    assert (
        versions[0] == version
    ), "The current package version must be the first software citation in README"
    block = _release_block(version)
    assert f"version = {{{version}}}" in block
    assert f"/tag/v{version}" in block


def test_release_citations_are_unique_and_newest_first():
    versions = RELEASE_SUMMARY.findall(README)
    assert len(versions) == len(set(versions))
    assert (
        set(versions) >= HISTORICAL_RELEASES
    ), "Published citations must not disappear"
    numeric_versions = [tuple(map(int, version.split("."))) for version in versions]
    assert numeric_versions == sorted(numeric_versions, reverse=True)


def test_software_authorship_is_cumulative():
    versions = RELEASE_SUMMARY.findall(README)
    chronological = list(reversed(versions))
    previous_authors: set[str] = set()
    for version in chronological:
        current_authors = _authors(version)
        assert previous_authors <= current_authors, (
            f"ZADU {version} drops previously credited software authors: "
            f"{sorted(previous_authors - current_authors)}"
        )
        previous_authors = current_authors


def test_citation_section_and_individual_entries_are_collapsed():
    section = _details_block("Citation")
    assert len(RELEASE_SUMMARY.findall(section)) == len(RELEASE_SUMMARY.findall(README))
    assert "### Original paper" in section
    assert "<summary><strong>Original ZADU paper</strong></summary>" not in section
    for version in RELEASE_SUMMARY.findall(section):
        assert (
            f"<details>\n<summary><strong>ZADU {version}</strong></summary>" in section
        )


def test_pre_0_5_releases_are_grouped():
    earlier = _details_block("Earlier releases (0.1.0-0.4.2)")
    versions = set(RELEASE_SUMMARY.findall(earlier))
    assert versions == {version for version in HISTORICAL_RELEASES if version < "0.5.0"}
