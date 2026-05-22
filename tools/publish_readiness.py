#!/usr/bin/env python
"""Publish-readiness wrapper for the local omegaprompt release audit.

This tool produces a final readiness status only. It never uploads to PyPI,
creates tags, pushes tags, or creates/edits GitHub Releases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import release_audit  # noqa: E402


def build_readiness_report(root: Path | str = ROOT) -> dict[str, object]:
    audit = release_audit.run_release_audit(Path(root).resolve())
    return {
        "schema_version": "1.0",
        "tool": "tools/publish_readiness.py",
        "root": audit["root"],
        "final_status": audit["final_status"],
        "summary": audit["summary"],
        "release_audit": audit,
        "mutations": {
            "pypi_publish": False,
            "git_tags_created": False,
            "git_tags_pushed": False,
            "github_releases_created_or_edited": False,
        },
    }


def render_human(report: dict[str, object]) -> str:
    audit = report["release_audit"]
    assert isinstance(audit, dict)
    lines = [
        "omegaprompt publish readiness report",
        f"Root: {report['root']}",
        f"Final status: {report['final_status']}",
        "",
        "Blocking checks:",
    ]
    blocking = [check for check in audit["checks"] if check["blocking"]]
    if not blocking:
        lines.append("  none")
    else:
        for check in blocking:
            lines.append(f"- [{check['status']}] {check['id']}: {check['message']}")
            if check.get("remediation"):
                lines.append(f"  remediation: {check['remediation']}")
    lines.extend(
        [
            "",
            "Release mutations:",
            "  PyPI publish: false",
            "  git tags created/pushed: false",
            "  GitHub Releases created/edited: false",
        ]
    )
    return "\n".join(lines)


def _write_text(stream: TextIO, text: str) -> None:
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        stream.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")


def main(argv: list[str] | None = None, *, root: Path | str | None = None, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assess local publish readiness without publishing.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless final status is READY.")
    parser.add_argument("--json-output", type=Path, help="Write machine-readable readiness report to this path.")
    args = parser.parse_args(argv)

    repo_root = Path(root).resolve() if root is not None else Path.cwd().resolve()
    report = build_readiness_report(repo_root)
    if args.json_output:
        output = args.json_output if args.json_output.is_absolute() else repo_root / args.json_output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_text(stdout or sys.stdout, render_human(report))
    if args.strict:
        return release_audit.strict_exit_code(str(report["final_status"]))  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
