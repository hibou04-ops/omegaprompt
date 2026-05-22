#!/usr/bin/env python
"""Offline Markdown link checker for repository and PyPI-facing docs.

Default mode performs no network access. External URLs are recorded as OK
without fetching unless --network or OMEGAPROMPT_LINK_CHECK_NETWORK=1 is used.
Repository-local GitHub blob/tree links are mapped back to local files so
README.md can stay PyPI-safe without losing offline validation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO


ROOT = Path(__file__).resolve().parents[1]
GITHUB_BLOB_PREFIX = "https://github.com/hibou04-ops/omegaprompt/blob/main/"
GITHUB_TREE_PREFIX = "https://github.com/hibou04-ops/omegaprompt/tree/main/"
README_VARIANTS = {
    "README.md",
    "README_KR.md",
    "EASY_README.md",
    "EASY_README_KR.md",
}
SCAN_PATTERNS = [
    "README.md",
    "README_KR.md",
    "EASY_README.md",
    "EASY_README_KR.md",
    "docs/**/*.md",
    "examples/**/*.md",
    ".github/pull_request_template.md",
    ".github/ISSUE_TEMPLATE/*.yml",
]

LINK_RE = re.compile(r"(!?)\[[^\]\n]*\]\(([^)\n]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
HTML_ID_RE = re.compile(r"""<[^>]+?\sid=["']([^"']+)["'][^>]*>""", re.IGNORECASE)


@dataclass(frozen=True)
class LinkCheck:
    status: str
    code: str
    source: str
    line: int
    target: str
    message: str
    details: dict[str, Any]

    @property
    def blocking(self) -> bool:
        return self.status in {"BROKEN", "ENVIRONMENT_BLOCKED", "TOOLING_MISSING"}

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "code": self.code,
            "source": self.source,
            "line": self.line,
            "target": self.target,
            "message": self.message,
            "details": self.details,
            "blocking": self.blocking,
        }


def ok(source: Path, line: int, target: str, message: str, **details: Any) -> LinkCheck:
    return LinkCheck("OK", "OK", _repo_rel(source), line, target, message, details)


def broken(source: Path, line: int, target: str, code: str, message: str, **details: Any) -> LinkCheck:
    return LinkCheck("BROKEN", code, _repo_rel(source), line, target, message, details)


def environment_blocked(source: Path, line: int, target: str, message: str, **details: Any) -> LinkCheck:
    return LinkCheck("ENVIRONMENT_BLOCKED", "NETWORK_BLOCKED", _repo_rel(source), line, target, message, details)


def default_scan_files(root: Path = ROOT) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in SCAN_PATTERNS:
        matches = sorted(root.glob(pattern)) if any(token in pattern for token in "*?[") else [root / pattern]
        for path in matches:
            if path.is_file():
                resolved = path.resolve()
                if resolved not in seen:
                    files.append(path)
                    seen.add(resolved)
    return files


def run_checks(
    root: Path | str = ROOT,
    *,
    scan_files: Iterable[Path] | None = None,
    network: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve()
    files = [Path(path) for path in scan_files] if scan_files is not None else default_scan_files(repo_root)
    checks: list[LinkCheck] = []

    for source in files:
        path = source if source.is_absolute() else repo_root / source
        if not path.exists():
            checks.append(
                LinkCheck(
                    "BROKEN",
                    "SCAN_FILE_MISSING",
                    _repo_rel(path),
                    0,
                    _repo_rel(path),
                    "Configured markdown scan file does not exist.",
                    {},
                )
            )
            continue
        checks.extend(_check_file(repo_root, path, network=network))

    status_counts: dict[str, int] = {}
    for check in checks:
        status_counts[check.status] = status_counts.get(check.status, 0) + 1

    return {
        "schema_version": "1.0",
        "tool": "tools/check_markdown_links.py",
        "root": str(repo_root),
        "network_enabled": network,
        "summary": {
            "files_scanned": len(files),
            "total_checks": len(checks),
            "blocking_checks": sum(1 for check in checks if check.blocking),
            "status_counts": status_counts,
        },
        "checks": [check.to_json() for check in checks],
    }


def _check_file(root: Path, source: Path, *, network: bool) -> list[LinkCheck]:
    checks: list[LinkCheck] = []
    text = source.read_text(encoding="utf-8")
    for line_number, line in _iter_non_fenced_lines(text):
        for match in LINK_RE.finditer(line):
            raw_target = _clean_destination(match.group(2))
            if not raw_target:
                continue
            checks.append(_check_target(root, source, line_number, raw_target, network=network))
    return checks


def _check_target(root: Path, source: Path, line: int, raw_target: str, *, network: bool) -> LinkCheck:
    if "\\" in raw_target:
        return broken(source, line, raw_target, "WINDOWS_BACKSLASH", "Markdown links must use forward slashes.")

    split = urllib.parse.urlsplit(raw_target)
    if split.scheme in {"mailto", "tel"}:
        return ok(source, line, raw_target, "Non-file link scheme is allowed.", scheme=split.scheme)

    if split.scheme in {"http", "https"}:
        mapped = _map_repo_github_url(raw_target)
        if mapped is not None:
            return _check_local_path(root, source, line, raw_target, mapped.path, mapped.fragment)
        if network:
            return _check_external_url(source, line, raw_target)
        return ok(source, line, raw_target, "External URL recorded without network fetch.", network_checked=False)

    if split.scheme:
        return ok(source, line, raw_target, "Non-file URI scheme recorded without local validation.", scheme=split.scheme)

    path_part = urllib.parse.unquote(split.path)
    fragment = urllib.parse.unquote(split.fragment)
    if _is_pypi_unsafe_readme_variant_link(root, source, path_part):
        return broken(
            source,
            line,
            raw_target,
            "PYPI_UNSAFE_RELATIVE_LINK",
            "README.md must use an absolute GitHub URL for cross-file README variant links.",
        )
    return _check_local_path(root, source, line, raw_target, path_part, fragment)


def _check_external_url(source: Path, line: int, raw_target: str) -> LinkCheck:
    request = urllib.request.Request(raw_target, method="HEAD", headers={"User-Agent": "omegaprompt-link-check/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - explicit opt-in network mode.
            status = getattr(response, "status", 200)
    except Exception as exc:  # pragma: no cover - default tests do not use network.
        return environment_blocked(source, line, raw_target, f"Network link check failed: {exc}")
    if 200 <= int(status) < 400:
        return ok(source, line, raw_target, "External URL fetched successfully.", http_status=int(status))
    return broken(source, line, raw_target, "EXTERNAL_HTTP_ERROR", "External URL returned an error status.", http_status=int(status))


def _check_local_path(root: Path, source: Path, line: int, raw_target: str, path_part: str, fragment: str) -> LinkCheck:
    target_path = source if path_part == "" else source.parent / path_part
    resolved = _resolve_inside_root(root, target_path)
    if resolved is None:
        return broken(source, line, raw_target, "OUTSIDE_REPOSITORY", "Link target escapes the repository root.")

    case_result = _case_sensitive_existing_path(root, resolved)
    if case_result["status"] == "missing":
        return broken(source, line, raw_target, "MISSING_FILE", "Linked file does not exist.", expected=case_result["expected"])
    if case_result["status"] == "case_mismatch":
        return broken(
            source,
            line,
            raw_target,
            "CASE_MISMATCH",
            "Linked path case does not match the repository entry.",
            expected=case_result["expected"],
            actual=case_result["actual"],
        )

    existing = Path(case_result["path"])
    if fragment and _should_validate_anchor(existing):
        anchors = _anchors_for(existing)
        if fragment not in anchors:
            return broken(
                source,
                line,
                raw_target,
                "MISSING_ANCHOR",
                "Linked markdown anchor does not exist.",
                anchor=fragment,
                resolved_target=_repo_rel(existing),
            )
    return ok(source, line, raw_target, "Local link target exists.", resolved_target=_repo_rel(existing), anchor=fragment or None)


def _resolve_inside_root(root: Path, target: Path) -> Path | None:
    root_resolved = Path(os.path.abspath(root))
    try:
        resolved = Path(os.path.abspath(target))
        resolved.relative_to(root_resolved)
    except ValueError:
        return None
    return resolved


def _case_sensitive_existing_path(root: Path, target: Path) -> dict[str, str]:
    root_resolved = Path(os.path.abspath(root))
    try:
        rel = Path(os.path.abspath(target)).relative_to(root_resolved)
    except ValueError:
        return {"status": "missing", "expected": str(target)}

    current = root_resolved
    for part in rel.parts:
        entries = {child.name: child for child in current.iterdir()} if current.is_dir() else {}
        if part in entries:
            current = entries[part]
            continue
        lowered = {child.name.lower(): child for child in entries.values()}
        actual = lowered.get(part.lower())
        if actual is not None:
            return {"status": "case_mismatch", "expected": _repo_rel(current / part), "actual": _repo_rel(actual)}
        return {"status": "missing", "expected": _repo_rel(current / part)}
    return {"status": "ok", "path": str(current)}


def _anchors_for(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    anchors: set[str] = set()
    seen_slugs: dict[str, int] = {}

    for match in HTML_ID_RE.finditer(text):
        anchors.add(match.group(1))

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        heading = match.group(2).strip().strip("#").strip()
        base = _github_slug(heading)
        if not base:
            continue
        count = seen_slugs.get(base, 0)
        seen_slugs[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def _github_slug(heading: str) -> str:
    heading = re.sub(r"`([^`]*)`", r"\1", heading)
    heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = heading.lower()
    chars: list[str] = []
    for char in heading:
        if char.isalnum() or char in {" ", "-", "_"}:
            chars.append(char)
    slug = "".join(chars).strip()
    slug = re.sub(r"\s", "-", slug)
    return slug.strip("-")


def _should_validate_anchor(path: Path) -> bool:
    return path.suffix.lower() in {".md", ".markdown"} or path.name.lower().endswith(".md")


def _map_repo_github_url(raw_target: str) -> urllib.parse.SplitResult | None:
    split = urllib.parse.urlsplit(raw_target)
    for prefix in (GITHUB_BLOB_PREFIX, GITHUB_TREE_PREFIX):
        if raw_target.startswith(prefix):
            rest = raw_target[len(prefix) :]
            rest_split = urllib.parse.urlsplit(rest)
            path = urllib.parse.unquote(rest_split.path)
            fragment = urllib.parse.unquote(split.fragment)
            return urllib.parse.SplitResult("", "", path, "", fragment)
    return None


def _is_pypi_unsafe_readme_variant_link(root: Path, source: Path, path_part: str) -> bool:
    try:
        rel_source = source.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return False
    normalized_target = path_part.lstrip("./")
    return rel_source == "README.md" and normalized_target in (README_VARIANTS - {"README.md"})


def _iter_non_fenced_lines(text: str) -> Iterable[tuple[int, str]]:
    in_fence = False
    fence_marker = ""
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue
        if not in_fence:
            yield line_number, line


def _clean_destination(destination: str) -> str:
    destination = destination.strip()
    if destination.startswith("<") and ">" in destination:
        return destination[1 : destination.index(">")].strip()
    if " " in destination:
        before_title, _, title = destination.partition(" ")
        if title.lstrip().startswith(("\"", "'")):
            return before_title.strip()
    return destination


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def render_human(report: dict[str, Any], stream: TextIO = sys.stdout) -> None:
    summary = report["summary"]
    print("Markdown link check", file=stream)
    print(f"Root: {report['root']}", file=stream)
    print(f"Network enabled: {report['network_enabled']}", file=stream)
    print(f"Files scanned: {summary['files_scanned']}", file=stream)
    print(f"Status counts: {summary['status_counts']}", file=stream)
    print(f"Blocking checks: {summary['blocking_checks']}", file=stream)
    for check in report["checks"]:
        if check["blocking"]:
            print(
                f"- {check['status']} {check['code']} {check['source']}:{check['line']} -> {check['target']}: {check['message']}",
                file=stream,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Markdown links without network access by default.")
    parser.add_argument("--strict", action="store_true", help="Return nonzero when broken links are found.")
    parser.add_argument("--json-output", type=Path, help="Write a machine-readable JSON report.")
    parser.add_argument(
        "--network",
        action="store_true",
        help="Opt into fetching external links. Default mode never uses network.",
    )
    args = parser.parse_args(argv)

    network = args.network or os.environ.get("OMEGAPROMPT_LINK_CHECK_NETWORK") == "1"
    report = run_checks(ROOT, network=network)
    render_human(report)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.strict and report["summary"]["blocking_checks"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
