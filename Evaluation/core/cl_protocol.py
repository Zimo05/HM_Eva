"""Canonical protocol objects for continual-learning benchmarks.

The benchmark manifest is the protocol boundary.  Consumers should load it
through :class:`CLProtocol` instead of reconstructing task meaning from CSV
names, directory listings, or fixed task numbers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CLTaskSpec:
    """Semantic description of one continual-learning task."""

    task_id: int
    regime_weights: dict[str, float]
    shift_type: str
    recurrence_of: str | None = None
    paired_control: str | None = None
    stage_label: str | None = None


@dataclass(frozen=True)
class CLAnchorSpec:
    """One frozen anchor bank declared by the protocol."""

    regime_id: str
    path: Path
    evaluation_scope: str = "persistent"


@dataclass(frozen=True)
class CLProtocol:
    """Validated, immutable view of ``benchmark_manifest.json``.

    The public task metadata mirrors the protocol contract.  Split paths,
    anchors, controls, and the original manifest are kept as validated
    implementation details so every consumer can use the same path contract.
    """

    root: Path
    benchmark_id: str
    version: int
    num_tasks: int
    event_dim: int
    betas: tuple[float, ...]
    first_seen: dict[str, int]
    tasks: dict[int, CLTaskSpec]
    persistent_regimes: frozenset[str]
    transient_regimes: frozenset[str]
    _task_splits: dict[int, dict[str, Path]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _anchors: tuple[CLAnchorSpec, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    _controls: tuple[dict[str, Any], ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    _adaptation: dict[int, dict[str, Any]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _raw_manifest: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )
    _manifest_path: Path | None = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, root: Path) -> "CLProtocol":
        """Load and validate the canonical manifest below ``root``."""

        root = Path(root).expanduser().resolve()
        manifest_path = root / "benchmark_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"continual benchmark manifest is required: {manifest_path}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{manifest_path} must contain a JSON object")
        return cls._from_mapping(root, payload, manifest_path)

    @classmethod
    def _from_mapping(
        cls, root: Path, payload: Mapping[str, Any], manifest_path: Path | None = None
    ) -> "CLProtocol":
        """Build a protocol from a decoded manifest.

        This private constructor keeps all validation in one place while
        allowing compatibility callers to retain the decoded manifest.
        """

        root = Path(root).expanduser().resolve()
        source = manifest_path or root / "benchmark_manifest.json"

        if payload.get("protocol") not in (None, "continual_hawkes"):
            raise ValueError(
                f"{source} has unsupported protocol: {payload.get('protocol')!r}"
            )

        benchmark_id = payload.get("benchmark_id", payload.get("benchmark"))
        if not isinstance(benchmark_id, str) or not benchmark_id:
            raise ValueError(f"{source} must define benchmark_id")

        version_value = payload.get("format_version", payload.get("version"))
        try:
            version = int(version_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source} must define an integer version") from error
        if version <= 0:
            raise ValueError(f"{source} version must be positive")

        try:
            num_tasks = int(payload["num_tasks"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source} must define a positive num_tasks") from error
        if num_tasks <= 0:
            raise ValueError(f"{source} must define a positive num_tasks")

        try:
            event_dim = int(payload["event_dim"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source} must define a positive event_dim") from error
        if event_dim <= 0:
            raise ValueError(f"{source} event_dim must be positive")

        raw_betas = payload.get("betas")
        if not isinstance(raw_betas, (list, tuple)) or not raw_betas:
            raise ValueError(f"{source} betas must be a non-empty list")
        try:
            betas = tuple(float(value) for value in raw_betas)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source} betas must contain numbers") from error
        if any(not math.isfinite(value) or value <= 0 for value in betas):
            raise ValueError(f"{source} betas must contain positive finite values")

        persistent = cls._regime_list(payload, "persistent_regimes", source)
        transient = cls._regime_list(
            payload, "transient_regimes", source, allow_empty=True
        )
        persistent_set = frozenset(persistent)
        transient_set = frozenset(transient)
        if persistent_set.intersection(transient_set):
            raise ValueError(f"{source} persistent and transient regimes overlap")
        known_regimes = persistent_set.union(transient_set)

        raw_first_seen = payload.get("first_seen")
        if not isinstance(raw_first_seen, Mapping):
            raise ValueError(f"{source} first_seen must be an object")
        first_seen: dict[str, int] = {}
        for regime_id, task_id in raw_first_seen.items():
            if not isinstance(regime_id, str) or regime_id not in known_regimes:
                raise ValueError(
                    f"{source} first_seen references unknown regime {regime_id!r}"
                )
            try:
                first_seen[regime_id] = int(task_id)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{source} first_seen[{regime_id!r}] must be an integer"
                ) from error
        for regime_id in persistent:
            if regime_id not in first_seen:
                raise ValueError(f"{source} first_seen is missing {regime_id!r}")

        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or len(raw_tasks) != num_tasks:
            raise ValueError(
                f"{source} tasks must contain exactly {num_tasks} entries"
            )

        task_entries: dict[int, CLTaskSpec] = {}
        task_splits: dict[int, dict[str, Path]] = {}
        adaptation_specs: dict[int, dict[str, Any]] = {}
        derived_first_seen: dict[str, int] = {}
        for raw_task in raw_tasks:
            if not isinstance(raw_task, Mapping):
                raise ValueError(f"{source} contains a non-object task entry")
            try:
                task_id = int(raw_task["task_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{source} contains a task without an integer task_id"
                ) from error
            if task_id in task_entries:
                raise ValueError(f"{source} contains duplicate task_id {task_id}")
            if task_id < 0:
                raise ValueError(f"{source} task_id {task_id} must be non-negative")

            raw_weights = raw_task.get("regime_weights")
            if not isinstance(raw_weights, Mapping) or not raw_weights:
                raise ValueError(f"{source} task {task_id} has no regime_weights")
            weights: dict[str, float] = {}
            for regime_id, weight in raw_weights.items():
                if not isinstance(regime_id, str) or regime_id not in known_regimes:
                    raise ValueError(
                        f"{source} task {task_id} references unknown regime {regime_id!r}"
                    )
                try:
                    numeric_weight = float(weight)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{source} task {task_id} has a non-numeric regime weight"
                    ) from error
                if not math.isfinite(numeric_weight) or numeric_weight < 0:
                    raise ValueError(
                        f"{source} task {task_id} has an invalid regime weight"
                    )
                weights[regime_id] = numeric_weight
                if numeric_weight > 0:
                    derived_first_seen.setdefault(regime_id, task_id)
            if sum(weights.values()) <= 0:
                raise ValueError(f"{source} task {task_id} has zero total regime weight")

            shift_type = raw_task.get("shift_type")
            if not isinstance(shift_type, str) or not shift_type:
                raise ValueError(f"{source} task {task_id} must define shift_type")
            recurrence_of = raw_task.get("recurrence_of")
            if recurrence_of in (None, ""):
                recurrence_of = None
            elif not isinstance(recurrence_of, str):
                raise ValueError(
                    f"{source} task {task_id} recurrence_of must be a string or null"
                )
            if recurrence_of and any(
                parent not in known_regimes
                for parent in recurrence_of.split("|")
            ):
                raise ValueError(
                    f"{source} task {task_id} recurrence_of references an unknown regime"
                )
            paired_control = raw_task.get("paired_control", raw_task.get("control"))
            if paired_control in (None, ""):
                paired_control = None
            elif not isinstance(paired_control, str):
                raise ValueError(
                    f"{source} task {task_id} paired_control must be a string or null"
                )
            stage_label = raw_task.get("stage_label")
            if stage_label is not None and not isinstance(stage_label, str):
                raise ValueError(
                    f"{source} task {task_id} stage_label must be a string or null"
                )

            raw_splits = raw_task.get("splits")
            if not isinstance(raw_splits, Mapping):
                raise ValueError(f"{source} task {task_id} must define splits")
            task_splits[task_id] = {
                split: cls._checked_file(
                    root,
                    cls._split_value(raw_splits, split),
                    source,
                    f"task {task_id} {split}",
                )
                for split in ("train", "validation", "test")
            }
            raw_adaptation = raw_task.get("adaptation")
            if raw_adaptation is not None:
                if not isinstance(raw_adaptation, Mapping):
                    raise ValueError(
                        f"{source} task {task_id} adaptation must be an object"
                    )
                raw_k_values = raw_adaptation.get("K", raw_adaptation.get("k"))
                if not isinstance(raw_k_values, (list, tuple)) or not raw_k_values:
                    raise ValueError(
                        f"{source} task {task_id} adaptation.K must be a non-empty list"
                    )
                try:
                    k_values = tuple(sorted({int(value) for value in raw_k_values}))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{source} task {task_id} adaptation.K must contain integers"
                    ) from error
                if any(value < 0 for value in k_values) or 0 not in k_values:
                    raise ValueError(
                        f"{source} task {task_id} adaptation.K must include non-negative K=0"
                    )
                adaptation_specs[task_id] = {
                    "support": cls._checked_file(
                        root,
                        raw_adaptation.get("support"),
                        source,
                        f"task {task_id} adaptation support",
                    ),
                    "query": cls._checked_file(
                        root,
                        raw_adaptation.get("query"),
                        source,
                        f"task {task_id} adaptation query",
                    ),
                    "K": k_values,
                }
            task_entries[task_id] = CLTaskSpec(
                task_id=task_id,
                regime_weights=weights,
                shift_type=shift_type,
                recurrence_of=recurrence_of,
                paired_control=paired_control,
                stage_label=stage_label,
            )

        task_ids = sorted(task_entries)
        if len(task_ids) != num_tasks:
            raise ValueError(f"{source} task_ids must be unique and match num_tasks")
        task_id_set = set(task_ids)
        for regime_id, task_id in first_seen.items():
            if task_id not in task_id_set:
                raise ValueError(
                    f"{source} first_seen[{regime_id!r}] references unknown task {task_id}"
                )

        for regime_id, derived_task in derived_first_seen.items():
            if first_seen.get(regime_id) != derived_task:
                raise ValueError(
                    f"{source} first_seen[{regime_id!r}]={first_seen.get(regime_id)!r} "
                    f"is inconsistent with tasks (derived {derived_task})"
                )
        for regime_id in first_seen:
            if regime_id not in derived_first_seen and regime_id in persistent_set:
                raise ValueError(
                    f"{source} persistent first_seen[{regime_id!r}] has no task mixture"
                )

        raw_anchors = payload.get("anchors")
        if not isinstance(raw_anchors, list) or not raw_anchors:
            raise ValueError(f"{source} anchors must be a non-empty list")
        anchors: list[CLAnchorSpec] = []
        anchor_ids: set[str] = set()
        for raw_anchor in raw_anchors:
            if not isinstance(raw_anchor, Mapping):
                raise ValueError(f"{source} contains a non-object anchor entry")
            regime_id = raw_anchor.get("regime_id")
            if not isinstance(regime_id, str) or regime_id not in known_regimes:
                raise ValueError(
                    f"{source} anchor references unknown regime {regime_id!r}"
                )
            if regime_id in anchor_ids:
                raise ValueError(f"{source} contains duplicate anchor {regime_id!r}")
            evaluation_scope = raw_anchor.get("evaluation_scope", "persistent")
            if not isinstance(evaluation_scope, str) or not evaluation_scope:
                raise ValueError(
                    f"{source} anchor {regime_id} has an invalid evaluation_scope"
                )
            anchor_ids.add(regime_id)
            anchors.append(CLAnchorSpec(
                regime_id=regime_id,
                path=cls._checked_file(root, raw_anchor.get("path"), source, f"anchor {regime_id}"),
                evaluation_scope=evaluation_scope,
            ))

        raw_controls = payload.get("controls", [])
        if not isinstance(raw_controls, list):
            raise ValueError(f"{source} controls must be a list")
        controls: list[dict[str, Any]] = []
        control_paths: set[str] = set()
        for index, raw_control in enumerate(raw_controls):
            if not isinstance(raw_control, Mapping):
                raise ValueError(f"{source} control {index} must be an object")
            control = dict(raw_control)
            raw_splits = control.get("splits")
            if not isinstance(raw_splits, Mapping):
                raise ValueError(f"{source} control {index} must define splits")
            control["splits"] = {
                split: cls._checked_file(
                    root,
                    cls._split_value(raw_splits, split),
                    source,
                    f"control {index} {split}",
                ).relative_to(root).as_posix()
                for split in ("train", "validation", "test")
            }
            for key in ("control_id", "control_path", "path"):
                value = control.get(key)
                if isinstance(value, str):
                    control_paths.add(value.rstrip("/"))
            controls.append(control)

        for task in task_entries.values():
            if task.paired_control is not None:
                if task.paired_control.rstrip("/") not in control_paths:
                    raise ValueError(
                        f"{source} task {task.task_id} references missing control "
                        f"{task.paired_control!r}"
                    )

        missing_anchors = persistent_set.difference(anchor_ids)
        if missing_anchors:
            raise ValueError(
                f"{source} anchors are missing persistent regimes: "
                f"{sorted(missing_anchors)}"
            )

        return cls(
            root=root,
            benchmark_id=benchmark_id,
            version=version,
            num_tasks=num_tasks,
            event_dim=event_dim,
            betas=betas,
            first_seen=first_seen,
            tasks=task_entries,
            persistent_regimes=persistent_set,
            transient_regimes=transient_set,
            _task_splits=task_splits,
            _anchors=tuple(anchors),
            _controls=tuple(controls),
            _adaptation=adaptation_specs,
            _raw_manifest=dict(payload),
            _manifest_path=Path(source).resolve(),
        )

    @staticmethod
    def _regime_list(
        payload: Mapping[str, Any],
        key: str,
        source: Path,
        *,
        allow_empty: bool = False,
    ) -> list[str]:
        value = payload.get(key)
        if not isinstance(value, list) or (not allow_empty and not value) or not all(
            isinstance(item, str) and item for item in value
        ):
            qualifier = "list" if allow_empty else "non-empty list"
            raise ValueError(f"{source} {key} must be a {qualifier} of strings")
        if len(set(value)) != len(value):
            raise ValueError(f"{source} {key} contains duplicate regime IDs")
        return list(value)

    @staticmethod
    def _split_value(raw_splits: Mapping[str, Any], split: str) -> object:
        """Read the canonical ``validation`` key with ``val`` compatibility."""

        value = raw_splits.get(split)
        if value is None and split == "validation":
            value = raw_splits.get("val")
        return value

    @staticmethod
    def _checked_file(
        root: Path, raw_path: object, source: Path, description: str
    ) -> Path:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{source} {description} must be a relative path")
        candidate = (root / raw_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"{source} {description} escapes the benchmark root"
            ) from error
        if not candidate.is_file():
            raise FileNotFoundError(f"{source} {description} is missing: {candidate}")
        return candidate

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path or (self.root / "benchmark_manifest.json")

    @property
    def task_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.tasks))

    @property
    def anchors(self) -> tuple[CLAnchorSpec, ...]:
        return self._anchors

    @property
    def controls(self) -> tuple[dict[str, Any], ...]:
        return self._controls

    @property
    def raw_manifest(self) -> dict[str, Any]:
        return dict(self._raw_manifest)

    @property
    def adaptation_tasks(self) -> tuple[int, ...]:
        """Task IDs with an independent support/query adaptation protocol."""

        return tuple(sorted(self._adaptation))

    def adaptation(self, task_id: int) -> dict[str, Any] | None:
        """Return validated adaptation files and K values for one task."""

        spec = self._adaptation.get(int(task_id))
        if spec is None:
            return None
        return {
            "support": spec["support"],
            "query": spec["query"],
            "K": tuple(spec["K"]),
        }

    def task(self, task_id: int) -> CLTaskSpec:
        try:
            return self.tasks[int(task_id)]
        except (KeyError, ValueError, TypeError) as error:
            raise KeyError(f"unknown continual task {task_id!r}") from error

    def split_path(self, task_id: int, split: str) -> Path:
        normalized = "validation" if split == "val" else split
        if normalized not in {"train", "validation", "test"}:
            raise ValueError(f"unknown continual split {split!r}")
        try:
            return self._task_splits[int(task_id)][normalized]
        except (KeyError, ValueError, TypeError) as error:
            raise KeyError(f"unknown continual task {task_id!r}") from error

    def train_path(self, task_id: int) -> Path:
        """Return the declared gradient-update split for ``task_id``."""

        return self.split_path(task_id, "train")

    def val_path(self, task_id: int) -> Path:
        """Return the declared checkpoint-selection split for ``task_id``."""

        return self.split_path(task_id, "validation")

    def control_split_path(self, control_id: str, split: str) -> Path:
        """Return a declared matched-control split path."""

        normalized = "validation" if split == "val" else split
        if normalized not in {"train", "validation", "test"}:
            raise ValueError(f"unknown continual split {split!r}")
        wanted = str(control_id).rstrip("/")
        for control in self._controls:
            identifiers = {
                str(control.get(key, "")).rstrip("/")
                for key in ("control_path", "control_id", "path")
                if control.get(key) is not None
            }
            if wanted in identifiers:
                raw_path = control["splits"][normalized]
                return self._checked_file(
                    self.root,
                    raw_path,
                    self.manifest_path,
                    f"control {control_id} {normalized}",
                )
        raise KeyError(f"unknown continual control {control_id!r}")

    def seen_regimes(self, task_id: int) -> frozenset[str]:
        task_id = int(task_id)
        return frozenset(
            regime_id
            for regime_id, first_task in self.first_seen.items()
            if first_task <= task_id
        )

    def seen_persistent_regimes(self, task_id: int) -> frozenset[str]:
        return self.seen_regimes(task_id).intersection(self.persistent_regimes)

    def resolve_range(
        self, start: int | None = None, end: int | None = None
    ) -> tuple[int, int]:
        """Resolve an inclusive task range against manifest task IDs."""

        ids = self.task_ids
        if not ids:
            raise ValueError("benchmark protocol has no tasks")
        resolved_start = ids[0] if start is None else int(start)
        resolved_end = ids[-1] if end is None else int(end)
        if resolved_start > resolved_end:
            raise ValueError("task start must be less than or equal to task end")
        if resolved_start < ids[0] or resolved_end > ids[-1]:
            raise ValueError(
                f"task range {resolved_start}..{resolved_end} is outside "
                f"protocol range {ids[0]}..{ids[-1]}"
            )
        if not any(resolved_start <= task_id <= resolved_end for task_id in ids):
            raise ValueError(
                f"task range {resolved_start}..{resolved_end} selects no protocol tasks"
            )
        return resolved_start, resolved_end

    def task_ids_between(self, start: int | None = None, end: int | None = None) -> tuple[int, ...]:
        resolved_start, resolved_end = self.resolve_range(start, end)
        return tuple(
            task_id
            for task_id in self.task_ids
            if resolved_start <= task_id <= resolved_end
        )
