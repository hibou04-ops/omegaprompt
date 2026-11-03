"""Compact offline preflight replay; a validated preflight never grants deployment."""
from omegaprompt.runtime import verify_calibration_preflight


def main() -> None:
    cases = [
        ("unmeasured_noise", None, "PREFLIGHT_UNMEASURED_NOISE"),
        ("below_noise", 0.3, "PREFLIGHT_REJECTED"),
        ("preflight_only", 0.1, "PREFLIGHT_VALIDATED"),
    ]
    for name, noise, expected in cases:
        result = verify_calibration_preflight(0.2, noise)
        assert result["preflight_status"] == expected, name
        assert result["deployment_authorized"] is False
        prefix = "+ [PASS]" if expected == "PREFLIGHT_VALIDATED" else "- [BLOCKED]"
        print(f"{prefix} {name}: {result['preflight_status']}; deployment_authorized=False")


if __name__ == "__main__":
    main()
