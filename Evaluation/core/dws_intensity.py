"""DWS-only conditional-intensity curve evaluation and aggregation.

This module is intentionally separate from :mod:`intensity_eval`, whose
checkpoint/task protocol belongs to the continual-learning benchmark.  DWS
has a different oracle: every synthetic cluster is one known Hawkes law and
all models must be evaluated on the same held-out source sequences.

The raw artifact is a long-form ``intensity_points.csv``.  It contains both
the total intensity (``event_type == -1``) and every marked intensity, so a
single post-processing pass can produce the paper overlay as well as later
type-wise appendix figures without rerunning a checkpoint.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .intensity_eval import IntensityAdapter


RAW_FIELDS = (
    "model",
    "variant",
    "regime_id",
    "cluster_id",
    "anchor_id",
    "source_index",
    "tau",
    "event_type",
    "gt_intensity",
    "pred_intensity",
)


def load_dws_laws(parameters_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    """Load and validate ``parameters_<variant>.json``.

    DWS stores the integrated branching matrix ``A``.  Its simulation kernel
    is ``A * decay * exp(-decay * dt)``, so the returned amplitude is
    deliberately *not* divided by ``decay``.
    """

    path = Path(parameters_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"DWS parameter file must be a non-empty object: {path}")
    laws: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    for raw_cluster, raw_law in payload.items():
        if str(raw_cluster).startswith("_"):
            # Dataset-build metadata (for example ``_cluster_format``) is not
            # an evaluation law.
            continue
        if not isinstance(raw_law, Mapping):
            raise ValueError(f"DWS law {raw_cluster!r} is not an object")
        cluster_id = int(raw_cluster)
        mu = np.asarray(raw_law.get("mu"), dtype=np.float64).reshape(-1)
        A = np.asarray(raw_law.get("A"), dtype=np.float64)
        decay = float(raw_law.get("decay"))
        if (
            mu.size == 0
            or A.shape != (mu.size, mu.size)
            or not math.isfinite(decay)
            or decay <= 0.0
            or np.any(~np.isfinite(mu))
            or np.any(~np.isfinite(A))
            or np.any(mu < 0.0)
            or np.any(A < 0.0)
        ):
            raise ValueError(
                f"invalid DWS law {cluster_id}: mu={mu.shape}, A={A.shape}, "
                f"decay={decay}"
            )
        laws[cluster_id] = (mu, A, decay)
    return dict(sorted(laws.items()))


def load_dws_clusters(dataset_path: Path) -> dict[int, int]:
    """Return the oracle ``source_index -> cluster`` mapping.

    Only this post-training diagnostic reads the cluster column.  Model input
    records remain cluster-free, preserving the benchmark's oracle-isolation
    contract.
    """

    path = Path(dataset_path).expanduser().resolve()
    mapping: dict[int, int] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"cluster"}.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"DWS dataset is missing columns {sorted(missing)}: {path}")
        for row_index, row in enumerate(reader):
            raw_source = row.get("source_index")
            source_index = row_index if raw_source in (None, "") else int(raw_source)
            if source_index in mapping:
                raise ValueError(f"duplicate DWS source_index {source_index}: {path}")
            mapping[source_index] = int(row["cluster"])
    if not mapping:
        raise ValueError(f"DWS dataset contains no rows: {path}")
    return mapping


def fixed_dws_anchor_bank(
    records: Sequence[Mapping[str, Any]],
    *,
    source_clusters: Mapping[int, int],
    anchors_per_law: int | None,
) -> dict[int, list[Mapping[str, Any]]]:
    """Select the same deterministic held-out anchors for every model."""

    if anchors_per_law is not None and int(anchors_per_law) <= 0:
        raise ValueError("anchors_per_law must be positive or None")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    seen_sources: set[int] = set()
    for record in records:
        if "source_index" not in record:
            raise KeyError("DWS intensity records require source_index")
        source_index = int(record["source_index"])
        if source_index in seen_sources:
            raise ValueError(f"duplicate DWS anchor source_index {source_index}")
        seen_sources.add(source_index)
        if source_index not in source_clusters:
            raise KeyError(f"DWS source_index {source_index} has no cluster label")
        grouped[int(source_clusters[source_index])].append(record)
    selected: dict[int, list[Mapping[str, Any]]] = {}
    for cluster_id, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: int(row["source_index"]))
        selected[cluster_id] = (
            ordered if anchors_per_law is None else ordered[: int(anchors_per_law)]
        )
    return selected


def dws_future_grid(
    event_times: Sequence[float],
    *,
    horizon: float,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``tau`` and absolute model-query times after the last event.

    ``tau[0]`` is reported as exactly zero, while its absolute query is the
    next representable float after ``t_last``.  This evaluates the predictable
    right-limit and therefore includes the final observed event in the common
    history without assigning it a visible non-zero relative time.
    """

    times = np.asarray(event_times, dtype=np.float64).reshape(-1)
    if times.size == 0 or np.any(~np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("DWS anchor event times must be finite and non-decreasing")
    if not math.isfinite(float(horizon)) or float(horizon) <= 0.0:
        raise ValueError("DWS intensity horizon must be positive")
    if int(samples) < 2:
        raise ValueError("DWS intensity samples must be at least two")
    tau = np.linspace(0.0, float(horizon), int(samples), dtype=np.float64)
    absolute = float(times[-1]) + tau
    absolute[0] = np.nextafter(float(times[-1]), math.inf)
    return tau, absolute


def dws_ground_truth_curve(
    event_times: Sequence[float],
    event_types: Sequence[int],
    grid: Sequence[float],
    mu: np.ndarray,
    A: np.ndarray,
    decay: float,
) -> np.ndarray:
    """Evaluate the strict-causal DWS Hawkes law on one absolute grid."""

    times = np.asarray(event_times, dtype=np.float64).reshape(-1)
    types = np.asarray(event_types, dtype=np.int64).reshape(-1)
    query = np.asarray(grid, dtype=np.float64).reshape(-1)
    if times.size != types.size or times.size == 0:
        raise ValueError("DWS anchor event times/types must be non-empty and aligned")
    if np.any(types < 0) or np.any(types >= mu.size):
        raise ValueError("DWS anchor contains an event type outside the law")
    delta = query[:, None] - times[None, :]
    causal = delta > 0.0
    kernels = (
        float(decay)
        * np.exp(-float(decay) * np.maximum(delta, 0.0))
        * causal
    )
    excitation = (A[:, types][None, :, :] * kernels[:, None, :]).sum(axis=2)
    return np.maximum(mu[None, :] + excitation, 1e-12)


def evaluate_dws_intensity_curves(
    adapter: IntensityAdapter,
    records: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    parameters_path: Path,
    dataset_path: Path,
    model_name: str,
    variant: str | int,
    horizon: float = 10.0,
    samples: int = 200,
    anchors_per_law: int | None = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one checkpoint on the fixed DWS anchor bank and write raw CSVs."""

    laws = load_dws_laws(parameters_path)
    source_clusters = load_dws_clusters(dataset_path)
    anchor_bank = fixed_dws_anchor_bank(
        records,
        source_clusters=source_clusters,
        anchors_per_law=anchors_per_law,
    )
    missing_laws = sorted(set(laws).difference(anchor_bank))
    if missing_laws:
        raise ValueError(f"DWS anchor bank is missing laws: {missing_laws}")
    if adapter.event_type_count != next(iter(laws.values()))[0].size:
        raise ValueError(
            "DWS model/law event-type mismatch: "
            f"model={adapter.event_type_count}, law={next(iter(laws.values()))[0].size}"
        )

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    point_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    for cluster_id, law in laws.items():
        mu, A, decay = law
        regime_id = f"law_{cluster_id:02d}"
        for anchor_id, record in enumerate(anchor_bank[cluster_id]):
            event_times, event_types = _record_arrays(record)
            tau, absolute_grid = dws_future_grid(
                event_times,
                horizon=horizon,
                samples=samples,
            )
            predicted = np.asarray(
                adapter.intensity_curve(event_times, event_types, absolute_grid),
                dtype=np.float64,
            )
            expected_shape = (int(samples), adapter.event_type_count)
            if predicted.shape != expected_shape:
                raise ValueError(
                    f"{model_name} DWS intensity shape {predicted.shape}; "
                    f"expected {expected_shape}"
                )
            target = dws_ground_truth_curve(
                event_times,
                event_types,
                absolute_grid,
                mu,
                A,
                decay,
            )
            squared = (predicted - target) ** 2
            numerator_by_type = np.asarray(
                [integrate(squared[:, index], tau) for index in range(mu.size)],
                dtype=np.float64,
            )
            denominator_by_type = np.asarray(
                [integrate(target[:, index] ** 2, tau) for index in range(mu.size)],
                dtype=np.float64,
            ) + 1e-12
            metric_rows.append({
                "model": str(model_name),
                "variant": str(variant),
                "regime_id": regime_id,
                "cluster_id": cluster_id,
                "anchor_id": anchor_id,
                "source_index": int(record["source_index"]),
                "nise_total": float(numerator_by_type.sum() / denominator_by_type.sum()),
                "nise_type_macro": float(np.mean(numerator_by_type / denominator_by_type)),
                "total_ise": float(integrate((predicted.sum(1) - target.sum(1)) ** 2, tau)),
            })
            for grid_index, relative_time in enumerate(tau.tolist()):
                common = {
                    "model": str(model_name),
                    "variant": str(variant),
                    "regime_id": regime_id,
                    "cluster_id": cluster_id,
                    "anchor_id": anchor_id,
                    "source_index": int(record["source_index"]),
                    "tau": float(relative_time),
                }
                point_rows.append({
                    **common,
                    "event_type": -1,
                    "gt_intensity": float(target[grid_index].sum()),
                    "pred_intensity": float(predicted[grid_index].sum()),
                })
                for event_type in range(mu.size):
                    point_rows.append({
                        **common,
                        "event_type": event_type,
                        "gt_intensity": float(target[grid_index, event_type]),
                        "pred_intensity": float(predicted[grid_index, event_type]),
                    })

    summary_rows = _summary_rows(metric_rows)
    _write_csv(output / "intensity_points.csv", point_rows, RAW_FIELDS)
    _write_csv(output / "intensity_metrics.csv", metric_rows)
    _write_csv(output / "intensity_summary.csv", summary_rows)
    (output / "intensity_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "dws_fixed_anchor_future_intensity_v1",
                "dataset": "dws",
                "continual_learning": False,
                "model": str(model_name),
                "variant": str(variant),
                "parameters_path": str(Path(parameters_path).resolve()),
                "dataset_path": str(Path(dataset_path).resolve()),
                "anchors_per_law": anchors_per_law,
                "horizon": float(horizon),
                "grid_samples": int(samples),
                "history_rule": "all anchor events; tau=0 is the right limit after t_last",
                "raw_points": str((output / "intensity_points.csv").resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return point_rows, summary_rows


def aggregate_dws_intensity_points(
    point_files: Sequence[Path],
    *,
    output_dir: Path,
    title: str | None = None,
) -> tuple[Path, Path, Path]:
    """Combine raw files and draw one overview plus per-cluster PDFs per model."""

    if not point_files:
        raise ValueError("at least one DWS intensity_points.csv is required")
    rows: list[dict[str, Any]] = []
    for raw_path in point_files:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["event_type"]) != -1:
                    continue
                rows.append({
                    **row,
                    "anchor_id": int(row["anchor_id"]),
                    "source_index": int(row["source_index"]),
                    "tau": float(row["tau"]),
                    "gt_intensity": float(row["gt_intensity"]),
                    "pred_intensity": float(row["pred_intensity"]),
                })
    if not rows:
        raise ValueError("DWS point files contain no total-intensity rows")
    _validate_shared_anchor_protocol(rows)

    grouped_pred: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    grouped_gt: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for row in rows:
        regime = str(row["regime_id"])
        model = str(row["model"])
        tau = float(row["tau"])
        anchor = int(row["anchor_id"])
        grouped_pred[(regime, model, tau)].append(float(row["pred_intensity"]))
        grouped_gt[(regime, anchor, tau)].append(float(row["gt_intensity"]))

    curve_rows: list[dict[str, Any]] = []
    for (regime, model, tau), values in sorted(grouped_pred.items()):
        curve_rows.append({
            "model": model,
            "regime_id": regime,
            "tau": tau,
            "mean_intensity": float(np.mean(values)),
            "std_intensity": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n_anchor": len(values),
        })
    for regime in sorted({key[0] for key in grouped_gt}):
        taus = sorted({key[2] for key in grouped_gt if key[0] == regime})
        for tau in taus:
            # Average once over anchors, not once per duplicated model file.
            values = [
                float(np.mean(value))
                for (law, _anchor, grid_tau), value in grouped_gt.items()
                if law == regime and grid_tau == tau
            ]
            curve_rows.append({
                "model": "Ground Truth",
                "regime_id": regime,
                "tau": tau,
                "mean_intensity": float(np.mean(values)),
                "std_intensity": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_anchor": len(values),
            })

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    curve_path = output / "intensity_curves.csv"
    _write_csv(output / "intensity_points.csv", rows)
    _write_csv(curve_path, curve_rows)
    figure_root = output / "intensity_curves"
    figure_manifest = _plot_model_figures(
        curve_rows,
        figure_root,
        title=title,
    )
    manifest_path = output / "intensity_figure_manifest.json"
    manifest_path.write_text(
        json.dumps(figure_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return curve_path, figure_root, manifest_path


def _record_arrays(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_times = record.get("time_since_start", record.get("event_times"))
    raw_types = record.get("type_event", record.get("event_types"))
    if raw_times is None or raw_types is None:
        raise KeyError("DWS intensity record needs event times and event types")
    times = np.asarray(raw_times, dtype=np.float64).reshape(-1)
    types = np.asarray(raw_types, dtype=np.int64).reshape(-1)
    if times.size != types.size or times.size == 0:
        raise ValueError("DWS intensity record is empty or misaligned")
    return times, types


def _summary_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["model"]), str(row["variant"]), str(row["regime_id"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (model, variant, regime_id), values in sorted(grouped.items()):
        rows.append({
            "model": model,
            "variant": variant,
            "regime_id": regime_id,
            "n_anchor": len(values),
            "nise_total_mean": float(np.mean([float(row["nise_total"]) for row in values])),
            "nise_total_std": float(np.std([float(row["nise_total"]) for row in values], ddof=1)) if len(values) > 1 else 0.0,
            "nise_type_macro_mean": float(np.mean([float(row["nise_type_macro"]) for row in values])),
            "total_ise_mean": float(np.mean([float(row["total_ise"]) for row in values])),
        })
    return rows


def _validate_shared_anchor_protocol(rows: Sequence[Mapping[str, Any]]) -> None:
    model_points: dict[str, set[tuple[str, int, int, float]]] = defaultdict(set)
    gt_values: dict[tuple[str, int, int, float], float] = {}
    for row in rows:
        model = str(row["model"])
        point = (
            str(row["regime_id"]),
            int(row["anchor_id"]),
            int(row["source_index"]),
            float(row["tau"]),
        )
        if point in model_points[model]:
            raise ValueError(f"duplicate DWS intensity point for {model}: {point}")
        model_points[model].add(point)
        gt_key = point
        value = float(row["gt_intensity"])
        previous = gt_values.setdefault(gt_key, value)
        if not math.isclose(previous, value, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError(f"ground-truth intensity mismatch at {gt_key}")
    reference_model = sorted(model_points)[0]
    reference = model_points[reference_model]
    for model, points in model_points.items():
        if points != reference:
            missing = sorted(reference.difference(points))[:5]
            extra = sorted(points.difference(reference))[:5]
            raise ValueError(
                f"DWS anchor/time grid differs for {model} versus {reference_model}; "
                f"missing={missing}, extra={extra}"
            )


def _plot_model_figures(
    curve_rows: Sequence[Mapping[str, Any]],
    figure_root: Path,
    *,
    title: str | None,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = sorted({str(row["regime_id"]) for row in curve_rows})
    models = sorted(
        {str(row["model"]) for row in curve_rows if row["model"] != "Ground Truth"}
    )
    if not regimes or not models:
        raise ValueError("DWS figure data requires at least one law and one model")
    figure_root.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")
    manifest: dict[str, Any] = {
        "protocol": "dws_model_level_intensity_figures_v1",
        "intensity": "total",
        "ground_truth_style": "dotted",
        "prediction_style": "solid",
        "models": {},
    }

    def select(model: str, regime: str) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
        ground_truth = sorted(
            (
                row for row in curve_rows
                if row["regime_id"] == regime and row["model"] == "Ground Truth"
            ),
            key=lambda row: float(row["tau"]),
        )
        prediction = sorted(
            (
                row for row in curve_rows
                if row["regime_id"] == regime and row["model"] == model
            ),
            key=lambda row: float(row["tau"]),
        )
        if not ground_truth or not prediction:
            raise ValueError(f"missing DWS figure curve for {model}/{regime}")
        return ground_truth, prediction

    def draw(axis, ground_truth, prediction, color, *, legend: bool) -> None:
        axis.plot(
            [float(row["tau"]) for row in ground_truth],
            [float(row["mean_intensity"]) for row in ground_truth],
            color="black",
            linestyle=":",
            linewidth=2.4,
            label="Ground Truth",
            zorder=5,
        )
        axis.plot(
            [float(row["tau"]) for row in prediction],
            [float(row["mean_intensity"]) for row in prediction],
            color=color,
            linestyle="-",
            linewidth=2.0,
            label="Estimated" if legend else None,
        )
        axis.grid(alpha=0.2)
        axis.set_xlabel(r"Relative time $\tau$")
        axis.set_ylabel(r"Total intensity $\lambda_{\mathrm{total}}(t)$")

    for model_index, model in enumerate(models):
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"
        model_dir = figure_root / safe_model
        model_dir.mkdir(parents=True, exist_ok=True)
        color = colors(model_index % 10)

        columns = 4
        rows_count = int(math.ceil(len(regimes) / columns))
        figure, axes = plt.subplots(
            rows_count,
            columns,
            figsize=(4.2 * columns, 3.1 * rows_count),
            sharex=True,
            squeeze=False,
        )
        axes_flat = axes.reshape(-1)
        for panel, regime in enumerate(regimes):
            axis = axes_flat[panel]
            ground_truth, prediction = select(model, regime)
            draw(axis, ground_truth, prediction, color, legend=True)
            axis.set_title(_cluster_label(regime))
        for axis in axes_flat[len(regimes):]:
            axis.set_visible(False)
        handles, labels = axes_flat[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
        overview_title = f"{model} - DWS total-intensity recovery"
        if title:
            overview_title = f"{model} - {title}"
        figure.suptitle(overview_title, y=1.02)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        overview_pdf = model_dir / "overview.pdf"
        overview_png = model_dir / "overview.png"
        figure.savefig(overview_pdf, bbox_inches="tight")
        figure.savefig(overview_png, dpi=180, bbox_inches="tight")
        plt.close(figure)

        cluster_files: list[str] = []
        for cluster_number, regime in enumerate(regimes, start=1):
            ground_truth, prediction = select(model, regime)
            figure, axis = plt.subplots(figsize=(6.4, 4.2))
            draw(axis, ground_truth, prediction, color, legend=True)
            axis.set_title(f"{model} - Cluster {cluster_number:02d}")
            axis.legend(frameon=False)
            figure.tight_layout()
            cluster_pdf = model_dir / f"cluster_{cluster_number:02d}.pdf"
            figure.savefig(cluster_pdf, bbox_inches="tight")
            plt.close(figure)
            cluster_files.append(str(cluster_pdf.resolve()))

        manifest["models"][model] = {
            "overview_pdf": str(overview_pdf.resolve()),
            "overview_png": str(overview_png.resolve()),
            "cluster_pdfs": cluster_files,
            "figure_count": 1 + len(cluster_files),
        }
        # External macOS volumes may create AppleDouble sidecar files for PDF
        # metadata.  They are not figures and would make the documented
        # 14-PDF/model contract appear to contain duplicates.
        for sidecar in model_dir.glob("._*"):
            sidecar.unlink(missing_ok=True)
    for sidecar in figure_root.glob("._*"):
        sidecar.unlink(missing_ok=True)
    return manifest


def _cluster_label(regime_id: str) -> str:
    match = re.search(r"(\d+)$", regime_id)
    if match is None:
        return regime_id.replace("_", " ").title()
    return f"Cluster {int(match.group(1)) + 1:02d}"


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or sorted({str(key) for row in rows for key in row}))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "aggregate_dws_intensity_points",
    "dws_future_grid",
    "dws_ground_truth_curve",
    "evaluate_dws_intensity_curves",
    "fixed_dws_anchor_bank",
    "load_dws_clusters",
    "load_dws_laws",
]
