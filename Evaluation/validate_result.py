from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.io import sha256


REQUIRED = (
    "manifest.json", "status.json", "metrics.json", "sequence_metrics.csv",
    "predictions.jsonl.gz", "resources.json", "report.md", "checkpoint",
    "logs", "plots", "tree_events.jsonl",
)


def validate(path: Path) -> list[str]:
    errors = [f"missing {name}" for name in REQUIRED if not (path / name).exists()]
    if errors:
        return errors
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        errors.append(f"status is {status.get('state')!r}, not 'complete'")
    for item in manifest.get("inputs", []):
        source = Path(item["path"])
        if source.is_file() and sha256(source) != item["sha256"]:
            errors.append(f"input hash changed: {source}")
    if manifest.get("dataset") == "dws" and not manifest.get("arguments", {}).get("checkpoint"):
        prepared = path / "prepared" / "split_manifest.json"
        if manifest.get("model") in {"HM", "S2P2", "AttNHP"} and prepared.exists():
            split = json.loads(prepared.read_text(encoding="utf-8"))
            if not split.get("oracle_fields_hidden_from_model"):
                errors.append("DWS split lacks oracle-isolation declaration")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one portable result bundle")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    errors = validate(args.result_dir.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"Valid result: {args.result_dir.resolve()}")


if __name__ == "__main__":
    main()
