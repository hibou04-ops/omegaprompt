"""Single-file HTML scorecard for calibration artifacts (stdlib only).

Produces a self-contained HTML document (inline CSS, no external assets,
no JS) suitable for attaching to a CI artifact bucket or opening locally.
The artifact JSON remains the source of truth; this is a human-friendly
view of the same numbers the markdown report renders.
"""

from __future__ import annotations

import html
import json
from typing import Any

from omegaprompt.core.overfit import extract_overfit_metrics


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _row(label: str, value: Any) -> str:
    return f"<tr><th>{_esc(label)}</th><td>{_esc(value)}</td></tr>"


def _verdict_class(verdict: str) -> str:
    return {
        "GENERALIZES": "ok",
        "OVERFIT": "bad",
        "UNVERIFIABLE": "warn",
        "UNKNOWN": "warn",
    }.get(verdict, "warn")


def render_html(a) -> str:
    """Render a CalibrationArtifact as a self-contained HTML scorecard."""

    status = _esc(getattr(a.status, "value", a.status))
    ship = _esc(a.ship_recommendation.value)
    profile = _esc(a.selected_profile.value)

    overfit = extract_overfit_metrics(a)
    verdict = overfit.overfit_verdict
    verdict_cls = _verdict_class(verdict)

    summary_rows = [
        _row("Status", status),
        _row("Ship recommendation", ship),
        _row("Profile", profile),
        _row("Method", f"{a.method} (unlock_k={a.unlock_k})"),
        _row("Neutral fitness", f"{a.neutral_fitness:.4f}"),
        _row("Calibrated fitness", f"{a.calibrated_fitness:.4f}"),
        _row("Uplift", f"{a.uplift_absolute:+.4f} ({a.uplift_percent:+.2f}%)"),
        _row(
            "Quality per cost",
            f"{a.quality_per_cost_neutral:.6f} -> {a.quality_per_cost_best:.6f}",
        ),
        _row(
            "Quality per latency",
            f"{a.quality_per_latency_neutral:.6f} -> {a.quality_per_latency_best:.6f}",
        ),
        _row("Hard gate pass rate", f"{a.hard_gate_pass_rate:.1%}"),
        _row(
            "Target / Judge",
            f"{a.target_provider or '-'}/{a.target_model or '-'}  |  "
            f"{a.judge_provider or '-'}/{a.judge_model or '-'}",
        ),
    ]

    overfit_rows = [
        _row("Overfit verdict", verdict),
    ]
    if overfit.available:
        overfit_rows.extend(
            [
                _row(
                    "Transfer correlation (KC-4)",
                    overfit.transfer_correlation
                    if overfit.transfer_correlation is not None
                    else f"not computed ({overfit.transfer_correlation_status})",
                ),
                _row(
                    "min transfer correlation threshold",
                    overfit.min_transfer_correlation_threshold,
                ),
                _row("Generalization gap", overfit.generalization_gap),
                _row(
                    "max generalization gap threshold",
                    overfit.max_generalization_gap_threshold,
                ),
                _row("Validation mode", overfit.validation_mode),
                _row("Shared item count", overfit.shared_item_count),
                _row("Walk-forward passed", overfit.walk_forward_passed),
            ]
        )
    else:
        overfit_rows.append(
            _row("Walk-forward", "absent — generalization not measured")
        )

    sensitivity_html = ""
    if a.sensitivity_ranking:
        body = "".join(
            "<tr><td>{rank}</td><td>{axis}</td><td>{gd}</td></tr>".format(
                rank=_esc(row.get("rank")),
                axis=_esc(row.get("axis")),
                gd=_esc(
                    f"{row.get('gini_delta'):.3f}"
                    if isinstance(row.get("gini_delta"), (int, float))
                    else row.get("gini_delta")
                ),
            )
            for row in a.sensitivity_ranking
        )
        sensitivity_html = (
            "<h2>Sensitivity ranking</h2>"
            "<table class='grid'><thead><tr><th>rank</th><th>axis</th>"
            f"<th>gini_delta</th></tr></thead><tbody>{body}</tbody></table>"
        )

    params_json = _esc(json.dumps(a.calibrated_params, indent=2, default=str))
    neutral_json = _esc(json.dumps(a.neutral_baseline_params, indent=2, default=str))

    return _TEMPLATE.format(
        status=status,
        status_cls=_esc(_status_class(getattr(a.status, "value", a.status))),
        verdict=_esc(verdict),
        verdict_cls=_esc(verdict_cls),
        summary_rows="".join(summary_rows),
        overfit_rows="".join(overfit_rows),
        sensitivity_html=sensitivity_html,
        params_json=params_json,
        neutral_json=neutral_json,
    )


def _status_class(status: str) -> str:
    return "ok" if status == "OK" else "bad"


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>omegaprompt calibration scorecard</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 2rem; line-height: 1.5; background: #f6f7f9; color: #1b1f23; }}
.card {{ max-width: 860px; margin: 0 auto; background: #fff; border: 1px solid #e1e4e8;
        border-radius: 10px; padding: 1.5rem 2rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
h1 {{ font-size: 1.4rem; margin: 0 0 1rem; }}
h2 {{ font-size: 1.05rem; margin: 1.5rem 0 .5rem; border-bottom: 1px solid #eee; padding-bottom: .25rem; }}
.badges {{ margin-bottom: 1rem; }}
.badge {{ display: inline-block; padding: .2rem .6rem; border-radius: 999px; font-size: .8rem;
         font-weight: 600; margin-right: .5rem; }}
.badge.ok {{ background: #e6ffed; color: #1a7f37; border: 1px solid #a6f0bf; }}
.badge.bad {{ background: #ffeef0; color: #b91c1c; border: 1px solid #f5b9c0; }}
.badge.warn {{ background: #fff8e1; color: #8a6d00; border: 1px solid #ffe08a; }}
table {{ border-collapse: collapse; width: 100%; margin: .25rem 0; }}
th, td {{ text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #eef0f2; font-size: .92rem; }}
table.kv th {{ width: 42%; color: #57606a; font-weight: 600; }}
table.grid th {{ background: #f0f2f4; }}
pre {{ background: #0d1117; color: #e6edf3; padding: .9rem; border-radius: 8px; overflow-x: auto;
      font-size: .82rem; }}
footer {{ margin-top: 1.5rem; color: #8b949e; font-size: .8rem; }}
</style>
</head>
<body>
<div class="card">
  <h1>omegaprompt calibration scorecard</h1>
  <div class="badges">
    <span class="badge {status_cls}">status: {status}</span>
    <span class="badge {verdict_cls}">overfit: {verdict}</span>
  </div>

  <h2>Summary</h2>
  <table class="kv"><tbody>{summary_rows}</tbody></table>

  <h2>Overfit / generalization</h2>
  <table class="kv"><tbody>{overfit_rows}</tbody></table>

  {sensitivity_html}

  <h2>Calibrated parameters</h2>
  <pre>{params_json}</pre>

  <h2>Neutral baseline</h2>
  <pre>{neutral_json}</pre>

  <footer>The CalibrationArtifact JSON remains the source of truth. This scorecard is a rendered view.</footer>
</div>
</body>
</html>
"""
