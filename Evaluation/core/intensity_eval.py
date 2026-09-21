"""Model-native conditional-intensity evaluation for continual baselines.

The adapters in this module deliberately expose each neural TPP's own
conditional intensity.  They never decode, fit, or approximate a Hawkes law
from a baseline checkpoint.  A shared strict-causal grid and artifact writer
then make the resulting curves comparable with one another and with the
ground-truth Hawkes process used to generate the CL benchmark.
"""

from __future__ import annotations

import csv
import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class IntensityAdapter(ABC):
    """Uniform interface for a model's native marked conditional intensity."""

    def __init__(self, model: torch.nn.Module, event_type_count: int):
        if int(event_type_count) <= 0:
            raise ValueError("event_type_count must be positive")
        self.model = model
        self.event_type_count = int(event_type_count)

    @abstractmethod
    def intensity_curve(
        self,
        event_times: np.ndarray,
        event_types: np.ndarray,
        grid: np.ndarray,
    ) -> np.ndarray:
        """Return ``lambda_d(g | H_g)`` with shape ``[grid, event type]``."""

    @staticmethod
    def _model_device_dtype(model: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
        parameter = next(model.parameters())
        # S2P2 stores complex-valued state parameters, while its event-time
        # inputs remain real.  Use the matching real dtype in that case.
        dtype = parameter.real.dtype if torch.is_complex(parameter) else parameter.dtype
        return parameter.device, dtype


class EasyTPPIntensityAdapter(IntensityAdapter):
    """Native intensity adapter for RMTPP, FullyNN, S2P2, and AttNHP."""

    def __init__(
        self,
        model: torch.nn.Module,
        event_type_count: int,
        *,
        model_name: str,
    ):
        super().__init__(model, event_type_count)
        normalized = str(model_name).strip().lower()
        if normalized not in {"rmtpp", "fullynn", "s2p2", "attnhp"}:
            raise ValueError(f"unsupported EasyTPP intensity model: {model_name}")
        self.model_name = normalized

    @torch.no_grad()
    def intensity_curve(
        self,
        event_times: np.ndarray,
        event_types: np.ndarray,
        grid: np.ndarray,
    ) -> np.ndarray:
        times, types, grid = _validated_curve_inputs(
            event_times,
            event_types,
            grid,
            self.event_type_count,
        )
        device, dtype = self._model_device_dtype(self.model)
        time_tensor = torch.as_tensor(times, device=device, dtype=dtype)[None, :]
        type_tensor = torch.as_tensor(types, device=device, dtype=torch.long)[None, :]
        delta_tensor = torch.empty_like(time_tensor)
        delta_tensor[:, 0] = time_tensor[:, 0]
        delta_tensor[:, 1:] = time_tensor[:, 1:] - time_tensor[:, :-1]
        grid_tensor = torch.as_tensor(grid, device=device, dtype=dtype)

        if self.model_name == "attnhp":
            # AttNHP's public interface consumes absolute sample times.
            sample_times = grid_tensor.reshape(1, 1, -1).expand(
                1, time_tensor.size(1), -1
            )
        else:
            # RMTPP and S2P2 consume waiting times relative to each prefix.
            sample_times = (
                grid_tensor.reshape(1, 1, -1) - time_tensor[:, :, None]
            ).clamp_min(0.0)

        self.model.eval()
        # FullyNN differentiates its cumulative-hazard network with respect
        # to elapsed time, so its native intensity needs autograd even during
        # evaluation.  Other EasyTPP models stay under the method's no-grad
        # context.
        grad_context = (
            torch.enable_grad()
            if self.model_name == "fullynn"
            else torch.no_grad()
        )
        with grad_context:
            values = self.model.compute_intensities_at_sample_times(
                time_tensor,
                delta_tensor,
                type_tensor,
                sample_times,
                compute_last_step_only=False,
            )
            if self.model_name == "fullynn" and values.size(-1) == 1:
                if getattr(self.model, "mark_linear", None) is None:
                    raise ValueError(
                        "factorized FullyNN intensity requires mark_linear"
                    )
                hidden = self.model.forward(
                    time_tensor, delta_tensor, type_tensor
                )
                mark_probabilities = torch.softmax(
                    self.model.mark_linear(hidden), dim=-1
                )
                values = values * mark_probabilities[:, :, None, :]
        if values.ndim != 4 or values.size(0) != 1:
            raise ValueError(
                f"{self.model_name} intensity output must be [1, N, G, D]; "
                f"got {tuple(values.shape)}"
            )
        prefix_indices = np.searchsorted(times, grid, side="left") - 1
        if np.any(prefix_indices < 0):
            raise ValueError("intensity grid contains a point before the first event")
        grid_indices = torch.arange(grid.size, device=device)
        prefix_tensor = torch.as_tensor(
            prefix_indices, device=device, dtype=torch.long
        )
        selected = values[0, prefix_tensor, grid_indices, :]
        return _validated_intensity_output(
            selected.detach().cpu().numpy(),
            grid.size,
            self.event_type_count,
            self.model_name,
        )


class THPIntensityAdapter(IntensityAdapter):
    """Native interval-intensity adapter for ``Models/THP``."""

    @torch.no_grad()
    def intensity_curve(
        self,
        event_times: np.ndarray,
        event_types: np.ndarray,
        grid: np.ndarray,
    ) -> np.ndarray:
        times, types, grid = _validated_curve_inputs(
            event_times,
            event_types,
            grid,
            self.event_type_count,
        )
        device, dtype = self._model_device_dtype(self.model)
        time_tensor = torch.as_tensor(times, device=device, dtype=dtype)[None, :]
        # The standalone THP uses 1..D event IDs and reserves zero for padding.
        type_tensor = (
            torch.as_tensor(types, device=device, dtype=torch.long)[None, :] + 1
        )
        grid_tensor = torch.as_tensor(grid, device=device, dtype=dtype)

        self.model.eval()
        encoded, _prediction = self.model(type_tensor, time_tensor)
        history_logits = self.model.linear(encoded)[0]
        prefix_indices = np.searchsorted(times, grid, side="left") - 1
        if np.any(prefix_indices < 0):
            raise ValueError("intensity grid contains a point before the first event")
        prefix_tensor = torch.as_tensor(
            prefix_indices, device=device, dtype=torch.long
        )
        prefix_times = time_tensor[0, prefix_tensor]
        scaled_duration = (grid_tensor - prefix_times) / (prefix_times + 1.0)
        logits = (
            history_logits[prefix_tensor]
            + self.model.alpha.reshape(1, -1) * scaled_duration[:, None]
        )
        positive_beta = F.softplus(self.model.beta) + 1e-6
        values = F.softplus(positive_beta * logits) / positive_beta
        return _validated_intensity_output(
            values.detach().cpu().numpy(),
            grid.size,
            self.event_type_count,
            "thp",
        )


def strict_causal_grid(event_times: Sequence[float], samples: int = 256) -> np.ndarray:
    """Return the shared grid on which every model sees exactly ``t_j < g``.

    Neural baselines define their first native conditional intensity only
    after observing one event.  The grid therefore starts immediately to the
    right of the first event and ends at the final event.  ``searchsorted``
    with ``side='left'`` keeps later event-time grid points strict-causal.
    """

    times = np.asarray(event_times, dtype=np.float64).reshape(-1)
    if int(samples) < 2:
        raise ValueError("intensity samples must be at least two")
    if times.size < 2 or np.any(~np.isfinite(times)):
        raise ValueError("intensity evaluation needs at least two finite events")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("event times must be non-decreasing")
    start = np.nextafter(float(times[0]), math.inf)
    stop = float(times[-1])
    if not stop > start:
        raise ValueError("intensity evaluation needs a positive post-first-event span")
    return np.linspace(start, stop, int(samples), dtype=np.float64)


def evaluate_intensity_curves(
    adapter: IntensityAdapter,
    records: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    ground_truth_dir: Path,
    regime_id: str,
    model_name: str,
    checkpoint_task: int,
    samples: int = 256,
    plot_anchors: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate model-native curves, write NISE rows, raw curves, and plots."""

    if not records:
        raise ValueError("intensity evaluation records cannot be empty")
    if int(plot_anchors) < 0:
        raise ValueError("plot_anchors must be non-negative")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mu, W, betas = _load_ground_truth_law(
        Path(ground_truth_dir),
        str(regime_id),
        adapter.event_type_count,
    )
    rows: list[dict[str, Any]] = []
    for anchor_index, record in enumerate(records):
        event_times, event_types = _record_arrays(record)
        grid = strict_causal_grid(event_times, samples=samples)
        predicted = adapter.intensity_curve(event_times, event_types, grid)
        target = _hawkes_intensity_curve(
            event_times,
            event_types,
            grid,
            mu,
            W,
            betas,
        )
        nise, per_type = _nise(predicted, target, grid)
        plot_path: str | None = None
        curve_path: str | None = None
        if anchor_index < int(plot_anchors):
            curve_file = output_dir / "curves" / f"anchor_{anchor_index:03d}.csv"
            _write_curve_csv(curve_file, grid, target, predicted)
            curve_path = str(curve_file.resolve())
            plot_file = output_dir / "plots" / f"anchor_{anchor_index:03d}.png"
            _plot_curve(
                plot_file,
                grid=grid,
                target=target,
                predicted=predicted,
                event_times=event_times,
                model_name=model_name,
                regime_id=regime_id,
                checkpoint_task=checkpoint_task,
                anchor_index=anchor_index,
                nise=nise,
            )
            plot_path = str(plot_file.resolve())
        row: dict[str, Any] = {
            "model": str(model_name),
            "checkpoint_task": int(checkpoint_task),
            "regime_id": str(regime_id),
            "anchor_index": int(anchor_index),
            "events": int(event_times.size),
            "grid_samples": int(grid.size),
            "grid_start": float(grid[0]),
            "grid_end": float(grid[-1]),
            "history_rule": "event_time < grid_time",
            "nise": float(nise),
            "curve_path": curve_path,
            "plot_path": plot_path,
        }
        for event_type, value in enumerate(per_type.tolist()):
            row[f"nise_type_{event_type}"] = float(value)
        rows.append(row)

    nise_values = np.asarray([row["nise"] for row in rows], dtype=np.float64)
    summary = {
        "model": str(model_name),
        "checkpoint_task": int(checkpoint_task),
        "regime_id": str(regime_id),
        "sequence_count": len(rows),
        "grid_samples": int(samples),
        "grid_support": "post_first_event_to_last_event",
        "history_rule": "event_time < grid_time",
        "nise_mean": float(np.mean(nise_values)),
        "nise_median": float(np.median(nise_values)),
    }
    _write_rows(output_dir / "intensity_metrics.csv", rows)
    _write_rows(output_dir / "intensity_summary.csv", [summary])
    (output_dir / "intensity_manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "ground_truth_dir": str(Path(ground_truth_dir).resolve()),
                "native_conditional_intensity": True,
                "surrogate_hawkes_fit": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows, summary


def _validated_curve_inputs(
    event_times: np.ndarray,
    event_types: np.ndarray,
    grid: np.ndarray,
    event_type_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(event_times, dtype=np.float64).reshape(-1)
    types = np.asarray(event_types, dtype=np.int64).reshape(-1)
    grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    if times.size != types.size or times.size < 1:
        raise ValueError("event times/types must be non-empty and aligned")
    if grid.size < 2 or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) < 0.0):
        raise ValueError("intensity grid must be finite and non-decreasing")
    if np.any(types < 0) or np.any(types >= int(event_type_count)):
        raise ValueError("event type is outside the model vocabulary")
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) < 0.0):
        raise ValueError("event times must be finite and non-decreasing")
    return times, types, grid


def _validated_intensity_output(
    values: np.ndarray,
    grid_size: int,
    event_type_count: int,
    model_name: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    expected = (int(grid_size), int(event_type_count))
    if values.shape != expected:
        raise ValueError(
            f"{model_name} intensity curve has shape {values.shape}; expected {expected}"
        )
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError(
            f"{model_name} intensity curve contains non-finite/negative values"
        )
    return np.maximum(values, 1e-12)


def _record_arrays(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_times = record.get("time_since_start", record.get("event_times"))
    raw_types = record.get("type_event", record.get("event_types"))
    if raw_times is None or raw_types is None:
        raise KeyError("intensity record needs event times and event types")
    return (
        np.asarray(raw_times, dtype=np.float64).reshape(-1),
        np.asarray(raw_types, dtype=np.int64).reshape(-1),
    )


def _load_ground_truth_law(
    ground_truth_dir: Path,
    regime_id: str,
    event_type_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata_path = ground_truth_dir / "regimes.json"
    arrays_path = ground_truth_dir / "regimes.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(
            f"ground-truth intensity files are missing below {ground_truth_dir}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    regimes = metadata.get("regimes", {})
    if regime_id not in regimes:
        raise KeyError(f"ground-truth regime is missing: {regime_id}")
    info = regimes[regime_id]
    array_key = str(info.get("array_key", regime_id))
    betas = np.asarray(metadata.get("betas", ()), dtype=np.float64).reshape(-1)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        mu = np.asarray(arrays[f"{array_key}__mu"], dtype=np.float64).copy()
        W = np.asarray(arrays[f"{array_key}__W"], dtype=np.float64).copy()
    expected_W = (int(event_type_count), int(event_type_count), betas.size)
    if mu.shape != (int(event_type_count),) or W.shape != expected_W:
        raise ValueError(
            f"ground-truth law shape mismatch: mu={mu.shape}, W={W.shape}, "
            f"expected {(int(event_type_count),)} and {expected_W}"
        )
    if (
        betas.size == 0
        or np.any(~np.isfinite(mu))
        or np.any(~np.isfinite(W))
        or np.any(~np.isfinite(betas))
        or np.any(mu < 0.0)
        or np.any(W < 0.0)
        or np.any(betas <= 0.0)
    ):
        raise ValueError("ground-truth law must be finite and non-negative")
    return mu, W, betas


def _hawkes_intensity_curve(
    event_times: np.ndarray,
    event_types: np.ndarray,
    grid: np.ndarray,
    mu: np.ndarray,
    W: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    values = []
    for time in grid:
        intensity = mu.copy()
        causal = event_times < float(time)
        if np.any(causal):
            past_times = event_times[causal]
            past_types = event_types[causal]
            kernels = np.exp(
                -(float(time) - past_times)[:, None] * betas[None, :]
            )
            intensity += (W[:, past_types, :] * kernels[None, :, :]).sum(
                axis=(1, 2)
            )
        values.append(np.maximum(intensity, 1e-12))
    return np.stack(values, axis=0)


def _nise(
    predicted: np.ndarray,
    target: np.ndarray,
    grid: np.ndarray,
) -> tuple[float, np.ndarray]:
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    squared_error = (predicted - target) ** 2
    numerator_by_type = np.asarray(
        [integrate(squared_error[:, index], grid) for index in range(target.shape[1])],
        dtype=np.float64,
    )
    denominator_by_type = np.asarray(
        [integrate(target[:, index] ** 2, grid) for index in range(target.shape[1])],
        dtype=np.float64,
    ) + 1e-12
    return (
        float(numerator_by_type.sum() / denominator_by_type.sum()),
        numerator_by_type / denominator_by_type,
    )


def _plot_curve(
    path: Path,
    *,
    grid: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    event_times: np.ndarray,
    model_name: str,
    regime_id: str,
    checkpoint_task: int,
    anchor_index: int,
    nise: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    event_type_count = int(target.shape[1])
    panel_count = 1 + event_type_count
    columns = min(3, panel_count)
    rows = int(math.ceil(panel_count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 3.0 * rows),
        sharex=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    axes[0].plot(grid, target.sum(axis=1), ":", linewidth=1.8, label="ground truth")
    axes[0].plot(grid, predicted.sum(axis=1), "-", linewidth=1.4, label="predicted")
    axes[0].set_ylabel("total intensity")
    figure.suptitle(
        f"{model_name} | checkpoint task_{checkpoint_task:02d} | {regime_id} | "
        f"anchor {anchor_index:03d} | NISE={nise:.5f}"
    )
    axes[0].legend(loc="upper right")
    for event_type, axis in enumerate(axes[1:panel_count]):
        axis.plot(grid, target[:, event_type], ":", linewidth=1.8, label="ground truth")
        axis.plot(grid, predicted[:, event_type], "-", linewidth=1.4, label="predicted")
        axis.set_ylabel(f"type {event_type}")
        axis.legend(loc="upper right")
    for axis in axes[:panel_count]:
        axis.vlines(
            event_times,
            0.0,
            1.0,
            transform=axis.get_xaxis_transform(),
            color="0.65",
            linewidth=0.35,
            alpha=0.45,
        )
        axis.grid(alpha=0.18)
        axis.set_xlabel("time")
    for axis in axes[panel_count:]:
        axis.set_visible(False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_curve_csv(
    path: Path,
    grid: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
) -> None:
    rows = []
    for index, time in enumerate(grid.tolist()):
        row: dict[str, Any] = {
            "grid_index": index,
            "time": float(time),
            "target_total": float(target[index].sum()),
            "predicted_total": float(predicted[index].sum()),
        }
        for event_type in range(target.shape[1]):
            row[f"target_type_{event_type}"] = float(target[index, event_type])
            row[f"predicted_type_{event_type}"] = float(
                predicted[index, event_type]
            )
        rows.append(row)
    _write_rows(path, rows)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "EasyTPPIntensityAdapter",
    "IntensityAdapter",
    "THPIntensityAdapter",
    "evaluate_intensity_curves",
    "strict_causal_grid",
]
