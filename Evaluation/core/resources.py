from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def resource_record(result_dir: Path, started: float, command: list[str]) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    record: dict[str, Any] = {
        "wall_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0 if any("cuda" in token for token in command) else 0.0,
        "result_bytes": directory_bytes(result_dir),
        "checkpoint_bytes": directory_bytes(result_dir / "checkpoint"),
        "host": platform.node(),
        "pid": os.getpid(),
    }
    checkpoints = sorted((result_dir / "checkpoint").rglob("*.pt"))
    if checkpoints:
        try:
            import torch
            try:
                payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(checkpoints[-1], map_location="cpu")
            def tensor_count(value: Any) -> int:
                if isinstance(value, torch.Tensor):
                    return value.numel()
                if isinstance(value, dict):
                    return sum(tensor_count(item) for item in value.values())
                if isinstance(value, (list, tuple)):
                    return sum(tensor_count(item) for item in value)
                return 0
            model_payload = payload
            if isinstance(payload, dict):
                if "tree_state_dict" in payload:
                    model_payload = {
                        key: payload[key] for key in (
                            "tree_state_dict", "hawkes_state_dict",
                            "encoder_state_dict", "split_module_state_dicts",
                            "deep_sleep_gate_state_dict",
                            "topology_selector_state_dict",
                        ) if key in payload
                    }
                    controller = payload.get("controller_state", {})
                    if "module_state_dict" in controller:
                        model_payload["controller_state_dict"] = controller[
                            "module_state_dict"
                        ]
                else:
                    model_payload = payload.get(
                        "model_state_dict", payload.get("state_dict", payload)
                    )
            record["serialized_parameter_values"] = int(tensor_count(model_payload))
        except Exception as error:
            record["checkpoint_inspection_warning"] = str(error)
    try:
        import torch
        if torch.cuda.is_available():
            record["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
            record["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if "peak_gpu_memory_bytes" not in record:
        record["peak_gpu_memory_bytes"] = None
        record["gpu_measurement_note"] = "Child-process peak allocation unavailable; use the per-model native log or an external GPU monitor."
    return record
