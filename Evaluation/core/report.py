from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(path: Path, spec, metrics: dict[str, Any], command: list[str]) -> None:
    lines = [
        f"# {spec.dataset} / {spec.model} / {spec.condition}", "",
        "## Result", "",
    ]
    for key, value in sorted(metrics.items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Reproduction command", "", "```text", " ".join(command), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
