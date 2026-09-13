#!/usr/bin/env python3
"""Run S2P2 or AttNHP through the benchmark's native baseline contract.

The repository vendors EasyTPP's model implementations, but the upstream
runner evaluates the test loader during every validation epoch.  That is not
the benchmark protocol used here: validation selects one checkpoint and the
test split is evaluated exactly once afterwards.  This small adapter keeps
the model code untouched and owns the protocol, data loading, checkpoint, and
artifact details needed by :mod:`Evaluation.core`.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import pickle
import random
import shutil
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


_CACHE_ROOT = Path(tempfile.gettempdir()) / "easytpp_baseline_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))


EASYTPP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EASYTPP_ROOT.parents[1]
DATASETS_ROOT = PROJECT_ROOT / "Datasets"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EASYTPP_ROOT))

from _data_configuration_common import load_all_dws, load_standard_splits  # noqa: E402
from easy_tpp.model.torch_model.torch_attnhp import AttNHP  # noqa: E402
from easy_tpp.model.torch_model.torch_s2p2 import S2P2  # noqa: E402
from easy_tpp.preprocess.dataset import TPPDataset, get_data_loader  # noqa: E402
from easy_tpp.preprocess.event_tokenizer import EventTokenizer  # noqa: E402


MODEL_CLASSES = {"S2P2": S2P2, "AttNHP": AttNHP}
SUPPORTED_DATASETS = (
    "amazon",
    "retweet",
    "taxi",
    "stackoverflow",
    "taobao",
    "dws",
    "dws_8",
    "dws_10",
    "dws_13",
    "dws_15",
    "dws_17",
    "dws_20",
)
DEFAULT_EPOCHS_BY_MODEL = {"S2P2": 300, "AttNHP": 200}
DEFAULT_BATCH_SIZE = 256
DEFAULT_EARLY_STOP_PATIENCE = 25
DEFAULT_THINNING = {
    "num_sample": 1,
    "num_exp": 50,
    "over_sample_rate": 5,
    "patience_counter": 5,
    "num_samples_boundary": 5,
    "num_step_gen": 1,
}


@dataclass
class _ThinningConfig:
    num_sample: int
    num_exp: int
    over_sample_rate: float
    patience_counter: int
    num_samples_boundary: int
    dtime_max: float
    num_step_gen: int = 1


@dataclass
class _ModelConfig:
    """The subset of EasyTPP.ModelConfig consumed by the two torch models."""

    hidden_size: int
    time_emb_size: int
    num_layers: int
    num_heads: int
    use_mc_samples: bool
    loss_integral_num_sample_per_step: int
    dropout_rate: float
    use_ln: bool
    thinning: _ThinningConfig
    num_event_types_pad: int
    num_event_types: int
    pad_token_id: int
    model_id: str
    gpu: int
    model_specs: dict[str, Any]


class _TokenizerConfig:
    """Avoid importing OmegaConf just to construct EasyTPP's tokenizer."""

    def __init__(self, num_event_types: int):
        self.num_event_types = int(num_event_types)
        self.pad_token_id = int(num_event_types)
        self.padding_side = "right"
        self.truncation_side = "right"
        self.padding_strategy = None
        self.truncation_strategy = None
        self.max_len = None
        self.model_input_names = [
            "time_seqs",
            "time_delta_seqs",
            "type_seqs",
            "seq_non_pad_mask",
            "attention_mask",
        ]

    def pop(self, key: str, default: Any = None) -> Any:
        return vars(self).pop(key, default)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _evaluation_rng(seed: int):
    """Make stochastic thinning deterministic without perturbing training."""

    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if cuda_states is not None:
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _record_to_stream(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    times = [float(value) for value in record["time_since_start"]]
    deltas = [float(value) for value in record["time_since_last_event"]]
    types = [int(value) for value in record["type_event"]]
    if not (len(times) == len(deltas) == len(types)) or len(times) < 2:
        raise ValueError("EasyTPP sequences must contain at least two events")
    return [
        {
            "time_since_start": time,
            "time_since_last_event": delta,
            "type_event": event_type,
        }
        for time, delta, event_type in zip(times, deltas, types)
    ]


def _write_native_pickles(
    output: Path,
    dim_process: int,
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    """Materialize raw standard data in the same pkl schema as THP."""

    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        streams = [_record_to_stream(record) for record in splits[split]]
        with (output / f"{split}.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "dim_process": int(dim_process),
                    split: streams,
                    "source_index_by_seq": list(range(len(streams))),
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


def _prepare_raw_dataset(dataset: str, variant: str | None, seed: int, output: Path) -> Path:
    dws_variant = variant
    if dataset == "dws" or dataset.startswith("dws_"):
        if dws_variant is None and dataset.startswith("dws_"):
            dws_variant = dataset.removeprefix("dws_")
        if not dws_variant:
            raise ValueError("DWS requires --variant when no prepared data directory is supplied")
        loaded = load_all_dws(DATASETS_ROOT, variants=[dws_variant], seed=seed)
        dim_process, splits = loaded[str(dws_variant)]
    else:
        dim_process, splits = load_standard_splits(DATASETS_ROOT, dataset)
    _write_native_pickles(output, dim_process, splits)
    _write_json(
        output / "adapter_manifest.json",
        {
            "format_version": 1,
            "dataset": dataset,
            "variant": dws_variant,
            "seed": seed,
            "dim_process": dim_process,
            "splits": {name: len(list(values)) for name, values in splits.items()},
            "cluster_oracle_used_for_model_input": False,
        },
    )
    return output


def _load_pickle_split(path: Path, split: str, expected_dim: int | None = None) -> tuple[int, list[dict[str, Any]]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a mapping")
    try:
        dim_process = int(payload["dim_process"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} is missing a valid dim_process") from error
    if expected_dim is not None and dim_process != expected_dim:
        raise ValueError(
            f"{path} has dim_process={dim_process}, expected {expected_dim}"
        )
    streams = payload.get(split)
    if not isinstance(streams, (list, tuple)) or not streams:
        raise ValueError(f"{path} has no non-empty {split} split")

    records: list[dict[str, Any]] = []
    for sequence_index, stream in enumerate(streams):
        if isinstance(stream, Mapping):
            times = stream.get("time_since_start")
            deltas = stream.get("time_since_last_event")
            types = stream.get("type_event")
        else:
            times = deltas = types = None
            if isinstance(stream, (list, tuple)):
                times = [event.get("time_since_start") for event in stream if isinstance(event, Mapping)]
                deltas = [event.get("time_since_last_event") for event in stream if isinstance(event, Mapping)]
                types = [event.get("type_event") for event in stream if isinstance(event, Mapping)]
        if times is None or deltas is None or types is None:
            raise ValueError(f"{path} {split} sequence {sequence_index} has an invalid schema")
        times = [float(value) for value in times]
        deltas = [float(value) for value in deltas]
        types = [int(value) for value in types]
        if not (len(times) == len(deltas) == len(types)) or len(times) < 2:
            raise ValueError(
                f"{path} {split} sequence {sequence_index} must contain at least two aligned events"
            )
        if any(not math.isfinite(value) for value in times + deltas):
            raise ValueError(f"{path} {split} sequence {sequence_index} contains non-finite values")
        if any(right < left for left, right in zip(times, times[1:])):
            raise ValueError(f"{path} {split} sequence {sequence_index} has decreasing timestamps")
        if any(value < 0 or value >= dim_process for value in types):
            raise ValueError(
                f"{path} {split} sequence {sequence_index} contains an event type outside [0, {dim_process})"
            )
        records.append(
            {
                "seq_idx": sequence_index,
                "time_since_start": times,
                "time_since_last_event": deltas,
                "type_event": types,
            }
        )
    return dim_process, records


def _load_prepared_dataset(
    prepared: Path,
    max_sequences: int | None = None,
    max_events_per_sequence: int | None = None,
) -> tuple[int, dict[str, list[dict[str, Any]]]]:
    if not prepared.is_dir():
        raise FileNotFoundError(prepared)
    dim_process: int | None = None
    result: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        path = prepared / f"{split}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        split_dim, records = _load_pickle_split(path, split, dim_process)
        dim_process = split_dim if dim_process is None else dim_process
        if max_sequences is not None:
            records = records[:max_sequences]
        if max_events_per_sequence is not None:
            if max_events_per_sequence < 2:
                raise ValueError("--max-events-per-sequence must be at least 2")
            records = [
                {
                    **record,
                    "time_since_start": record["time_since_start"][:max_events_per_sequence],
                    "time_since_last_event": record["time_since_last_event"][:max_events_per_sequence],
                    "type_event": record["type_event"][:max_events_per_sequence],
                }
                for record in records
            ]
        if not records:
            raise ValueError(f"{split} split is empty after smoke limits")
        result[split] = records
    assert dim_process is not None
    return dim_process, result


def _make_loader(
    records: list[dict[str, Any]],
    dim_process: int,
    batch_size: int,
    shuffle: bool,
):
    data = {
        "time_seqs": [record["time_since_start"] for record in records],
        "time_delta_seqs": [record["time_since_last_event"] for record in records],
        "type_seqs": [record["type_event"] for record in records],
    }
    dataset = TPPDataset(data)
    tokenizer = EventTokenizer(_TokenizerConfig(dim_process))
    return get_data_loader(
        dataset,
        "torch",
        tokenizer,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def compute_training_dtime_max(records: Iterable[Mapping[str, Any]]) -> float:
    """Calibrate thinning only from the training split."""

    values = [
        float(delta)
        for record in records
        for delta in record["time_since_last_event"]
        if math.isfinite(float(delta)) and float(delta) > 0
    ]
    if not values:
        raise ValueError("training split has no positive inter-event duration")
    return max(1e-6, 1.2 * max(values))


def _make_model_config(model: str, dim_process: int, dtime_max: float, gpu: int) -> _ModelConfig:
    if model == "S2P2":
        return _ModelConfig(
            hidden_size=128,
            time_emb_size=16,
            num_layers=4,
            num_heads=2,
            use_mc_samples=True,
            loss_integral_num_sample_per_step=10,
            dropout_rate=0.1,
            use_ln=False,
            thinning=_ThinningConfig(dtime_max=dtime_max, **DEFAULT_THINNING),
            num_event_types_pad=dim_process + 1,
            num_event_types=dim_process,
            pad_token_id=dim_process,
            model_id=model,
            gpu=gpu,
            model_specs={
                "P": 16,
                "dropout_rate": 0.1,
                "act_func": "gelu",
                "for_loop": True,
                "pre_norm": False,
                "post_norm": True,
                "int_forward_variant": False,
                "int_backward_variant": True,
                "relative_time": True,
            },
        )
    return _ModelConfig(
        hidden_size=16,
        time_emb_size=4,
        num_layers=2,
        num_heads=2,
        use_mc_samples=True,
        loss_integral_num_sample_per_step=10,
        dropout_rate=0.0,
        use_ln=False,
        thinning=_ThinningConfig(dtime_max=dtime_max, **DEFAULT_THINNING),
        num_event_types_pad=dim_process + 1,
        num_event_types=dim_process,
        pad_token_id=dim_process,
        model_id=model,
        gpu=gpu,
        model_specs={},
    )


def _resolve_device(requested: str) -> tuple[torch.device, int]:
    value = str(requested).strip().lower()
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        value = "cuda:0"
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
        index = 0 if device.index is None else int(device.index)
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is unavailable")
        return device, index
    if device.type not in {"cpu", "mps"}:
        raise ValueError(f"unsupported device: {requested}")
    if device.type == "mps":
        raise RuntimeError("EasyTPP's bundled device adapter supports CPU/CUDA, not MPS")
    return device, -1


def _move_batch(batch, device: torch.device) -> list[torch.Tensor]:
    return list(batch.to(device).values())


def _train_epoch(model: torch.nn.Module, loader, device: torch.device) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_events = 0
    for batch in loader:
        values = _move_batch(batch, device)
        loss, num_events = model.loglike_loss(values)
        count = int(num_events.detach().cpu().item()) if torch.is_tensor(num_events) else int(num_events)
        if count <= 0:
            continue
        mean_loss = loss / count
        if not torch.isfinite(mean_loss):
            raise FloatingPointError("EasyTPP produced a non-finite training loss")
        model.optimizer.zero_grad(set_to_none=True)  # type: ignore[attr-defined]
        mean_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        model.optimizer.step()  # type: ignore[attr-defined]
        total_loss += float(loss.detach().cpu())
        total_events += count
    if total_events <= 0:
        raise ValueError("training split contains no target events")
    return {
        "loglike": -total_loss / total_events,
        "num_events": total_events,
        "accuracy": None,
        "rmse": None,
    }


def _predict_attnhp_one_step(
    model: AttNHP,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict every next event with AttNHP's absolute-time convention.

    EasyTPP's thinning sampler returns relative waiting times, while AttNHP's
    ``compute_intensities_at_sample_times`` expects absolute sample times when
    it receives one sample for every prefix event.  Keep the upstream model
    unchanged and adapt the sampler callback at this wrapper boundary.
    """

    time_seq, time_delta_seq, event_seq, _, _ = batch
    time_seq = time_seq[:, :-1]
    time_delta_seq = time_delta_seq[:, :-1]
    event_seq = event_seq[:, :-1]
    dtime_boundary = torch.max(
        time_delta_seq * model.event_sampler.dtime_max,
        time_delta_seq + model.event_sampler.dtime_max,
    )

    def intensity_at_relative_dtimes(
        prefix_times,
        prefix_dtimes,
        prefix_types,
        relative_dtimes,
        **kwargs,
    ):
        sample_times = relative_dtimes + prefix_times.unsqueeze(-1)
        return model.compute_intensities_at_sample_times(
            prefix_times,
            prefix_dtimes,
            prefix_types,
            sample_times,
            **kwargs,
        )

    accepted_dtimes, weights = model.event_sampler.draw_next_time_one_step(
        time_seq,
        time_delta_seq,
        event_seq,
        dtime_boundary,
        intensity_at_relative_dtimes,
        compute_last_step_only=False,
    )
    intensities_at_times = intensity_at_relative_dtimes(
        time_seq,
        time_delta_seq,
        event_seq,
        accepted_dtimes,
    )
    intensities_normalized = intensities_at_times / intensities_at_times.sum(
        dim=-1, keepdim=True
    )
    intensities_weighted = torch.einsum(
        "...s,...sm->...m", weights, intensities_normalized
    )
    types_pred = torch.argmax(intensities_weighted, dim=-1)
    dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)
    return dtimes_pred, types_pred


def _predict_one_step_at_every_event(
    model: torch.nn.Module,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, AttNHP):
        return _predict_attnhp_one_step(model, batch)
    return model.predict_one_step_at_every_event(batch)


def _evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    dim_process: int,
    seed: int,
    collect_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_events = 0
    correct = 0
    squared_error = 0.0
    prediction_count = 0
    rows: list[dict[str, Any]] = []
    sequence_cursor = 0
    with _evaluation_rng(seed), torch.no_grad():
        for batch in loader:
            values = _move_batch(batch, device)
            loss, num_events = model.loglike_loss(values)
            count = int(num_events.detach().cpu().item()) if torch.is_tensor(num_events) else int(num_events)
            total_loss += float(loss.detach().cpu())
            total_events += count
            if not collect_predictions:
                continue
            predicted_dtimes, predicted_types = _predict_one_step_at_every_event(model, values)
            target_dtimes = values[1][:, 1:]
            target_types = values[2][:, 1:]
            mask = values[3][:, 1:].bool()
            valid_types = predicted_types[mask]
            valid_dtimes = predicted_dtimes[mask]
            target_types_valid = target_types[mask]
            target_dtimes_valid = target_dtimes[mask]
            correct += int((valid_types == target_types_valid).sum().cpu())
            squared_error += float(((valid_dtimes - target_dtimes_valid) ** 2).sum().cpu())
            prediction_count += int(mask.sum().cpu())
            batch_size = int(mask.size(0))
            for batch_index in range(batch_size):
                positions = torch.nonzero(mask[batch_index], as_tuple=False).flatten().tolist()
                for position in positions:
                    predicted_type = int(predicted_types[batch_index, position].cpu())
                    true_type = int(target_types[batch_index, position].cpu())
                    rows.append(
                        {
                            "sequence_id": sequence_cursor + batch_index,
                            "event_index": int(position) + 1,
                            "true_type": true_type,
                            "predicted_type": predicted_type,
                            # EasyTPP's public one-step API returns argmax
                            # marks, not the normalized mark distribution at
                            # the observed event time.  Do not serialize an
                            # argmax one-hot vector as if it were a model
                            # probability.
                            "type_probabilities": None,
                            "true_delta_time": float(target_dtimes[batch_index, position].cpu()),
                            "predicted_delta_time": float(predicted_dtimes[batch_index, position].cpu()),
                            # A scalar sequence NLL is available from
                            # ``loglike_loss``.  Per-event decomposition is
                            # not exposed by these model implementations.
                            "event_nll": None,
                        }
                    )
            sequence_cursor += batch_size
    if total_events <= 0:
        raise ValueError("evaluation split contains no target events")
    metrics = {
        "loglike": -total_loss / total_events,
        "num_events": total_events,
        "accuracy": (correct / prediction_count) if prediction_count else None,
        "rmse": math.sqrt(squared_error / prediction_count) if prediction_count else None,
    }
    return metrics, rows


def _load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    metadata: dict[str, Any] = {}
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
        metadata = dict(payload)
    elif isinstance(payload, Mapping) and "state_dict" in payload:
        state = payload["state_dict"]
        metadata = dict(payload)
    else:
        state = payload
    if not isinstance(state, Mapping):
        raise ValueError(f"{path} does not contain a model state dict")
    model.load_state_dict(state, strict=True)
    return metadata


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_name: str,
    dim_process: int,
    epoch: int,
    validation: float,
    config: _ModelConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model": model_name,
            "dim_process": dim_process,
            "epoch": epoch,
            "selection_metric": "validation_loglike",
            "selection_value": validation,
            "model_config": {
                "hidden_size": config.hidden_size,
                "time_emb_size": config.time_emb_size,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "model_specs": config.model_specs,
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def _write_metrics_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def _write_predictions(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def _plot_metrics(
    output: Path,
    model: str,
    dataset: str,
    epoch_rows: list[Mapping[str, Any]],
    test_row: Mapping[str, Any],
    predictions: list[Mapping[str, Any]],
) -> list[Path]:
    """Write diagnostic plots without changing the benchmark metrics contract.

    Validation intentionally collects no thinning predictions, so only the
    likelihood curve is drawn across epochs. Accuracy/RMSE and the prediction
    diagnostics are computed from the single final test evaluation. This
    keeps plots useful without inventing validation values or running an extra
    prediction pass.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        warning = output / "log" / "plot_warning.txt"
        warning.write_text(
            "Plotting skipped because matplotlib is unavailable: {}\n".format(exc),
            encoding="utf-8",
        )
        print("Warning: matplotlib is unavailable; skipping EasyTPP plots", file=sys.stderr)
        return []

    plot_dir = output / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []

    def finite(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def series(split: str, metric: str) -> tuple[list[int], list[float]]:
        epochs: list[int] = []
        values: list[float] = []
        for row in epoch_rows:
            if row.get("Split") != split:
                continue
            epoch = finite(row.get("Epoch"))
            value = finite(row.get(metric))
            if epoch is None or value is None:
                continue
            epochs.append(int(epoch))
            values.append(value)
        return epochs, values

    colors = {"train": "#2563EB", "valid": "#EA580C"}
    figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    has_curve = False
    for split in ("train", "valid"):
        epochs, values = series(split, "Log-likelihood")
        if not values:
            continue
        has_curve = True
        axis.plot(
            epochs,
            values,
            color=colors[split],
            linewidth=2.1,
            marker="o" if len(values) <= 20 else None,
            markersize=3.5,
            label="Train" if split == "train" else "Validation",
        )
    if has_curve:
        best_epoch = finite(test_row.get("Epoch"))
        if best_epoch is not None and best_epoch > 0:
            axis.axvline(
                int(best_epoch),
                color="#6B7280",
                linestyle="--",
                linewidth=1.1,
                label="Selected epoch",
            )
        axis.legend(frameon=False)
    else:
        axis.text(
            0.5,
            0.5,
            "No training epochs recorded\n(evaluate-only run)",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_title(f"{model} on {dataset} - log-likelihood")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Log-likelihood per event")
    axis.grid(True, alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    likelihood_path = plot_dir / "likelihood.png"
    figure.savefig(likelihood_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    plot_paths.append(likelihood_path)

    test_loglike = finite(test_row.get("Log-likelihood"))
    test_accuracy = finite(test_row.get("Accuracy"))
    test_rmse = finite(test_row.get("RMSE"))
    test_metrics = (
        ("NLL/event", None if test_loglike is None else -test_loglike, "#7C3AED"),
        ("Accuracy", test_accuracy, "#059669"),
        ("Time RMSE", test_rmse, "#EA580C"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 4.2), constrained_layout=True)
    for axis, (label, value, color) in zip(axes, test_metrics):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_title(label)
        axis.set_xticks([])
        if value is None:
            axis.text(0.5, 0.5, "N/A", ha="center", va="center", transform=axis.transAxes)
            axis.set_ylim(0, 1)
            continue
        axis.bar([0], [value], color=color, width=0.55)
        axis.text(0, value, f"{value:.4f}", ha="center", va="bottom", fontsize=10)
        if label == "Accuracy":
            axis.set_ylim(0, 1)
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        else:
            axis.set_ylim(bottom=0)
        axis.grid(True, axis="y", alpha=0.25)
    epoch_value = int(finite(test_row.get("Epoch")) or 0)
    figure.suptitle(
        f"{model} on {dataset} - final test metrics (epoch {epoch_value})",
        fontsize=13,
    )
    test_metrics_path = plot_dir / "test_metrics.png"
    figure.savefig(test_metrics_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    plot_paths.append(test_metrics_path)

    time_points: list[tuple[float, float]] = []
    type_pairs: list[tuple[int, int]] = []
    for row in predictions:
        true_time = finite(row.get("true_delta_time"))
        predicted_time = finite(row.get("predicted_delta_time"))
        if true_time is not None and predicted_time is not None:
            time_points.append((true_time, predicted_time))
        try:
            true_type = int(row["true_type"])
            predicted_type = int(row["predicted_type"])
        except (KeyError, TypeError, ValueError):
            continue
        type_pairs.append((true_type, predicted_type))

    if time_points:
        # Keep large test sets readable and memory-bounded while preserving a
        # deterministic sample for reproducible benchmark artifacts.
        step = max(1, math.ceil(len(time_points) / 10_000))
        sampled_points = time_points[::step]
        true_times = np.asarray([point[0] for point in sampled_points], dtype=float)
        predicted_times = np.asarray([point[1] for point in sampled_points], dtype=float)
        lower = float(min(true_times.min(), predicted_times.min()))
        upper = float(max(true_times.max(), predicted_times.max()))
        padding = max((upper - lower) * 0.05, 1e-6)
        lower -= padding
        upper += padding
        figure, axis = plt.subplots(figsize=(6.4, 5.8), constrained_layout=True)
        axis.scatter(
            true_times,
            predicted_times,
            s=10,
            alpha=0.35,
            color="#2563EB",
            edgecolors="none",
        )
        axis.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.2, color="#6B7280")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_xlabel("True inter-event time")
        axis.set_ylabel("Predicted inter-event time")
        axis.set_title(f"{model} on {dataset} - test time prediction")
        axis.text(
            0.03,
            0.97,
            f"{len(time_points):,} events ({len(sampled_points):,} plotted)",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
        axis.grid(True, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        time_path = plot_dir / "time_predictions.png"
        figure.savefig(time_path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        plot_paths.append(time_path)

    if type_pairs:
        labels = sorted({label for pair in type_pairs for label in pair})
        label_to_index = {label: index for index, label in enumerate(labels)}
        confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for true_type, predicted_type in type_pairs:
            confusion[label_to_index[true_type], label_to_index[predicted_type]] += 1
        figure_size = max(5.5, min(10.0, 3.8 + 0.28 * len(labels)))
        figure, axis = plt.subplots(
            figsize=(figure_size, figure_size),
            constrained_layout=True,
        )
        image = axis.imshow(confusion, interpolation="nearest", cmap="Blues")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(f"{model} on {dataset} - test type confusion")
        axis.set_xlabel("Predicted type")
        axis.set_ylabel("True type")
        axis.set_xticks(range(len(labels)), labels=labels)
        axis.set_yticks(range(len(labels)), labels=labels)
        if len(labels) <= 30:
            for row_index in range(len(labels)):
                for column_index in range(len(labels)):
                    count = confusion[row_index, column_index]
                    if count:
                        threshold = confusion.max() / 2 if confusion.size else 0
                        axis.text(
                            column_index,
                            row_index,
                            str(int(count)),
                            ha="center",
                            va="center",
                            color="white" if count > threshold else "black",
                            fontsize=8,
                        )
        confusion_path = plot_dir / "type_confusion.png"
        figure.savefig(confusion_path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        plot_paths.append(confusion_path)

    return plot_paths


def _archive(output: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        # Keep the portable archive aligned with the other baseline runners;
        # checkpoints and predictions remain available in the result folder.
        for name in ("log", "csv", "plot"):
            source = output / name
            if not source.exists():
                continue
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    handle.add(path, arcname=str(Path(output.name) / name / path.relative_to(source)))


def _prepare_output(output: Path, overwrite: bool) -> None:
    output = output.expanduser().resolve()
    forbidden = {EASYTPP_ROOT.resolve(), PROJECT_ROOT.resolve(), DATASETS_ROOT.resolve()}
    if output in forbidden:
        raise ValueError(f"refusing to use a repository root as output: {output}")
    if output.exists():
        if not overwrite and any(output.iterdir()):
            raise FileExistsError(f"output is non-empty; pass --overwrite: {output}")
        if overwrite:
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoint", "csv", "log", "plot"):
        (output / name).mkdir(parents=True, exist_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CLASSES), required=True)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--variant", default=None, help="DWS tree variant when adapting raw data")
    parser.add_argument("--dataset-label", default=None, help="Optional continual benchmark label")
    parser.add_argument("--prepared-data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help="validation epochs without improvement before stopping",
    )
    parser.add_argument("--max-sequences", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-events-per-sequence", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    resolved_epochs = (
        DEFAULT_EPOCHS_BY_MODEL[args.model]
        if args.epochs is None
        else args.epochs
    )
    resolved_batch_size = DEFAULT_BATCH_SIZE if args.batch_size is None else args.batch_size
    if resolved_epochs < 1:
        raise ValueError("--epochs must be positive")
    if resolved_batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.early_stop_patience < 1:
        raise ValueError("--early-stop-patience must be positive")
    if args.max_sequences is not None and args.max_sequences < 1:
        raise ValueError("--max-sequences must be positive")
    output = args.output_dir.expanduser().resolve()
    archive = (args.archive or output.with_suffix(".tar.gz")).expanduser().resolve()
    if archive.exists():
        if not args.overwrite:
            raise FileExistsError(f"archive already exists; pass --overwrite: {archive}")
        archive.unlink()
    _prepare_output(output, args.overwrite)
    device, gpu = _resolve_device(args.device)
    _set_seed(args.seed)

    resolved_variant = args.variant
    if resolved_variant is None and args.dataset.startswith("dws_"):
        resolved_variant = args.dataset.removeprefix("dws_")
    prepared = args.prepared_data_dir.expanduser().resolve() if args.prepared_data_dir else output / "prepared"
    if args.prepared_data_dir is None:
        _prepare_raw_dataset(args.dataset, resolved_variant, args.seed, prepared)
    dim_process, records = _load_prepared_dataset(
        prepared,
        max_sequences=args.max_sequences,
        max_events_per_sequence=args.max_events_per_sequence,
    )
    dtime_max = compute_training_dtime_max(records["train"])
    config = _make_model_config(args.model, dim_process, dtime_max, gpu)
    model = MODEL_CLASSES[args.model](config)
    # TorchBaseModel calls ``to(device)`` before the concrete model creates
    # its own layers.  Move the fully constructed model once more so CUDA
    # runs do not leave subclass parameters (for example S2P2's mark
    # embedding) on CPU while the batch is on the requested GPU.
    model.to(device)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-2 if args.model == "S2P2" else 1e-3)

    train_loader = _make_loader(records["train"], dim_process, resolved_batch_size, shuffle=True)
    valid_loader = _make_loader(records["dev"], dim_process, resolved_batch_size, shuffle=False)
    test_loader = _make_loader(records["test"], dim_process, resolved_batch_size, shuffle=False)

    initial_checkpoint = (
        args.initial_checkpoint.expanduser().resolve()
        if args.initial_checkpoint is not None
        else None
    )
    initial_metadata: dict[str, Any] = {}
    if initial_checkpoint is not None:
        # This must precede both the evaluate-only branch and the training
        # loop.  Sequential/replay continual runs pass the previous task's
        # checkpoint here and then fine-tune that loaded model.
        initial_metadata = _load_checkpoint(model, initial_checkpoint, device)

    config_payload = {
        "format_version": 1,
        "model": args.model,
        "dataset": args.dataset,
        "dataset_label": args.dataset_label,
        "variant": resolved_variant,
        "seed": args.seed,
        "device": str(device),
        "training": {
            "max_epochs": resolved_epochs,
            "batch_size": resolved_batch_size,
            "early_stop_patience": args.early_stop_patience,
        },
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "dim_process": dim_process,
        "prepared_data_dir": str(prepared),
        "dtime_max": dtime_max,
        "dtime_max_source": "training split only",
        "architecture": {
            "hidden_size": config.hidden_size,
            "time_emb_size": config.time_emb_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "loss_integral_num_sample_per_step": config.loss_integral_num_sample_per_step,
            "use_mc_samples": config.use_mc_samples,
            "dropout_rate": config.dropout_rate,
            "model_specs": config.model_specs,
        },
        "optimizer": {"name": "adam", "learning_rate": 1e-2 if args.model == "S2P2" else 1e-3},
        "thinning": {
            **DEFAULT_THINNING,
            "dtime_max": dtime_max,
        },
        "evaluation_protocol": {
            "selection_metric": "validation log-likelihood",
            "test_evaluations_during_training": 0,
            "test_evaluated_once_after_best_checkpoint": True,
        },
    }
    _write_json(output / "log" / "run_config.json", config_payload)

    epoch_rows: list[dict[str, Any]] = []
    checkpoint = output / "checkpoint" / "best.pt"
    best_epoch = 0
    best_validation = -math.inf
    completed_epochs = 0
    no_improvement_epochs = 0
    if args.evaluate_only:
        if initial_checkpoint is not None:
            if initial_checkpoint != checkpoint.resolve():
                shutil.copy2(initial_checkpoint, checkpoint)
        best_epoch = int(initial_metadata.get("epoch", 0)) if initial_metadata else 0
    else:
        for epoch in range(1, resolved_epochs + 1):
            completed_epochs = epoch
            train_metrics = _train_epoch(model, train_loader, device)
            valid_metrics, _ = _evaluate(
                model,
                valid_loader,
                device,
                dim_process,
                seed=args.seed + epoch,
                collect_predictions=False,
            )
            epoch_rows.extend(
                [
                    {
                        "Epoch": epoch,
                        "Split": "train",
                        "Log-likelihood": train_metrics["loglike"],
                        "Accuracy": train_metrics["accuracy"],
                        "RMSE": train_metrics["rmse"],
                        "num_events": train_metrics["num_events"],
                        "SelectionMetric": "validation_loglike",
                    },
                    {
                        "Epoch": epoch,
                        "Split": "valid",
                        "Log-likelihood": valid_metrics["loglike"],
                        "Accuracy": valid_metrics["accuracy"],
                        "RMSE": valid_metrics["rmse"],
                        "num_events": valid_metrics["num_events"],
                        "SelectionMetric": "validation_loglike",
                    },
                ]
            )
            validation_loglike = float(valid_metrics["loglike"])
            if not math.isfinite(validation_loglike):
                raise FloatingPointError("EasyTPP produced a non-finite validation log-likelihood")
            if validation_loglike > best_validation:
                best_validation = validation_loglike
                best_epoch = epoch
                no_improvement_epochs = 0
                _save_checkpoint(
                    checkpoint,
                    model,
                    args.model,
                    dim_process,
                    epoch,
                    best_validation,
                    config,
                )
            else:
                no_improvement_epochs += 1
            if no_improvement_epochs >= args.early_stop_patience:
                break
        if not checkpoint.is_file():
            raise RuntimeError("validation never produced a best checkpoint")
        _load_checkpoint(model, checkpoint, device)

    _write_metrics_csv(
        output / "csv" / "epoch_metrics.csv",
        epoch_rows,
        ["Epoch", "Split", "Log-likelihood", "Accuracy", "RMSE", "num_events", "SelectionMetric"],
    )
    test_metrics, predictions = _evaluate(
        model,
        test_loader,
        device,
        dim_process,
        seed=args.seed + 100_000,
        collect_predictions=True,
    )
    test_row = {
        "Epoch": best_epoch,
        "Split": "test",
        "Log-likelihood": test_metrics["loglike"],
        "Accuracy": test_metrics["accuracy"],
        "RMSE": test_metrics["rmse"],
        "num_events": test_metrics["num_events"],
        "SelectionMetric": "validation_loglike",
    }
    _write_metrics_csv(
        output / "csv" / "test_metrics.csv",
        [test_row],
        ["Epoch", "Split", "Log-likelihood", "Accuracy", "RMSE", "num_events", "SelectionMetric"],
    )
    _write_predictions(output / "predictions.jsonl.gz", predictions)
    plot_paths = _plot_metrics(
        output,
        args.model,
        args.dataset,
        epoch_rows,
        test_row,
        predictions,
    )
    _write_json(
        output / "log" / "summary.json",
        {
            "model": args.model,
            "dataset": args.dataset,
            "best_epoch": best_epoch,
            "epochs_completed": completed_epochs,
            "max_epochs": resolved_epochs,
            "early_stop_patience": args.early_stop_patience,
            "early_stopped": bool(
                not args.evaluate_only
                and completed_epochs < resolved_epochs
                and no_improvement_epochs >= args.early_stop_patience
            ),
            "selection_metric": "validation_loglike",
            "validation_loglike": None if args.evaluate_only else best_validation,
            "test": test_metrics,
            "artifacts": {
                "checkpoint": str(checkpoint) if checkpoint.is_file() else None,
                "epoch_metrics": str(output / "csv" / "epoch_metrics.csv"),
                "test_metrics": str(output / "csv" / "test_metrics.csv"),
                "predictions": str(output / "predictions.jsonl.gz"),
                "plots": [str(path) for path in plot_paths],
            },
        },
    )
    _archive(output, archive)
    print(json.dumps({"output_dir": str(output), "archive": str(archive), "test": test_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
