from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import sha256, write_json
from .paths import PROJECT_ROOT


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch
        info.update({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as error:
        info["torch_error"] = str(error)
    return info


def build_manifest(spec, args, inputs: list[Path], command: list[str] | None) -> dict[str, Any]:
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required evaluation input is missing: "
            + ", ".join(str(path) for path in missing)
        )
    return {
        "format_version": 1,
        "job_key": spec.job_key,
        "script": spec.script,
        "kind": spec.kind,
        "dataset": spec.dataset,
        "model": spec.model,
        "condition": spec.condition,
        "seed": args.seed,
        "variant": getattr(args, "variant", None),
        "rank": getattr(args, "rank", None),
        "task_start": getattr(args, "task_start", None),
        "task_end": getattr(args, "task_end", None),
        "arguments": vars(args),
        "command": command,
        "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in inputs],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "runtime": runtime_info(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def compatible(existing: dict[str, Any], current: dict[str, Any]) -> bool:
    keys = ("job_key", "dataset", "model", "condition", "seed", "variant", "rank", "task_start", "task_end", "inputs")
    if not all(existing.get(key) == current.get(key) for key in keys):
        return False
    ignored = {"resume", "dry_run", "output_root", "run_id", "eval_batch_size"}

    def comparable_arguments(payload: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: value for key, value in payload.items() if key not in ignored
        }
        # Manifests written before the scope API have neither field. Treat
        # them as the default no-event-output mode so --resume remains usable
        # after the evaluator upgrade. Explicit scope changes still remain
        # incompatible, because they change the requested artifact contract.
        if "event_prediction_scope" not in result:
            result["event_prediction_scope"] = (
                "all" if result.get("save_event_predictions", False) else "none"
            )
        result.pop("save_event_predictions", None)
        return result

    old_args = comparable_arguments(existing.get("arguments", {}))
    new_args = comparable_arguments(current.get("arguments", {}))
    return old_args == new_args


def begin(result_dir: Path, manifest: dict[str, Any], resume: bool) -> bool:
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = result_dir / "manifest.json"
    status_path = result_dir / "status.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not compatible(existing, manifest):
            raise RuntimeError(f"refusing to reuse incompatible result directory: {result_dir}")
        if resume and status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("state") == "complete":
            return False
        if not resume:
            raise FileExistsError(f"result exists; use --resume or another --run-id: {result_dir}")
    write_json(manifest_path, manifest)
    write_json(status_path, {"state": "running", "updated_at": datetime.now(timezone.utc).isoformat()})
    return True


def finish(result_dir: Path, state: str, error: str | None = None) -> None:
    write_json(result_dir / "status.json", {"state": state, "error": error, "updated_at": datetime.now(timezone.utc).isoformat()})
