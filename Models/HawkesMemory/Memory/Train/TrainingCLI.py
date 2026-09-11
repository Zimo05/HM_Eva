"""Command-line pipeline for constructing and running training."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import json
from pathlib import Path
import sys

from Train.TrainingComponents import *  # noqa: F403


from Train.TrainingTrainer import MemoryTreeTrainer
from DataSplit import file_sha256, load_split_manifest, select_sequences


# These values describe the learner and must survive a continual-learning
# stage boundary.  They are temporarily parsed with a ``None`` default so a
# resume can distinguish an omitted option from an explicit override.
_PERSISTENT_CONFIG_DESTS = frozenset({
    "seed",
    "learning_rate",
    "weight_decay",
    "grad_clip",
    "sleep_every",
    "evaluation_ablation",
    "controller_target_version",
    "controller_heads",
    "controller_write_ranking",
    "prototype_duplicate_threshold",
    "prototype_duplicate_quantile",
    "prototype_mode_threshold",
    "prototype_mode_capacity",
    "prototype_context_alias_capacity",
    "count_similarity_low",
    "count_similarity_high",
    "count_exponent",
    "count_saturation",
    "count_topk",
    "route_mi_weight",
    "route_posterior_weight",
    "route_distill_weight",
    "route_mix_weight",
    "route_energy_temperature",
    "route_encoder_warmup_epochs",
    "route_encoder_grad_scale",
    "route_encoder_reliability_decay",
    "route_teacher_temperature",
    "route_probe_weight",
    "route_probe_leaves",
    "route_probe_leaf_smoothing",
    "route_probe_residual_temperature",
    "route_probe_gain_temperature",
    "route_probe_complexity_weight",
    "route_balance_weight",
    "route_balance_batch_size",
    "wake_wavefront_batch_size",
    "retrieval_microbatch",
    "route_balance_max_steps",
    "route_balance_target_kl",
    "router_lr_scale",
    "router_init_gain",
    "semantic_blend",
    "leaf_symmetry_scale",
    "residual_init_scale",
    "residual_init_rank",
    "residual_init_grad_clip",
    "alignment_epochs",
    "alignment_batch_size",
    "alignment_lr",
    "alignment_weight_decay",
    "alignment_temperature",
    "alignment_grad_clip",
    "allow_root_only_alignment",
    "prune_warmup_epochs",
    "merge_min_replay",
    "merge_stale_weight",
    "merge_dynamics_weight",
    "merge_loss_weight",
    "merge_gate_temperature",
    "merge_budget_ratio",
    "merge_dual_lr",
    "merge_dual_initial",
    "light_replay_budget",
    "split_min_structural_strength",
    "split_min_effective_sample_size",
    "split_route_loss_weight",
    "split_anchor_weight",
    "deep_availability_tau",
    "deep_probe_interval",
    "deep_computation_cost",
    "deep_prior_probability",
    "deep_prior_weight",
    "deep_evidence_budget",
    "topology_inertia_strength",
    "topology_inertia_tau",
    "z_dim",
    "node_dim",
    "memory_key_dim",
    "frontier_budget",
    "frontier_min_experts",
    "frontier_routing_temperature",
    "frontier_exploration",
    "frontier_confidence_weight",
    "frontier_compute_cost",
    "frontier_posterior_temperature",
    "frontier_credible_mass",
    "frontier_owner_confidence",
    "max_writes_per_sequence",
    "num_basis",
    "decays",
})


def _normalise_config_value(value):
    """Make checkpoint/config values directly comparable and JSON-friendly."""

    if torch.is_tensor(value):
        return _normalise_config_value(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_config_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_normalise_config_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _load_json_object(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration file must contain an object: {path}")
    return payload


def _load_checkpoint_payload(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return payload


_WAKE_ARG_TO_CHECKPOINT = {
    "route_mi_weight": "lambda_route_mi",
    "route_posterior_weight": "lambda_route_posterior",
    "route_distill_weight": "lambda_route_distill",
    "route_mix_weight": "lambda_route_mix",
    "route_probe_weight": "lambda_route_probe",
    "route_balance_weight": "lambda_route_balance",
}
_FRONTIER_ARG_TO_CHECKPOINT = {
    "frontier_budget": "frontier_budget",
    "frontier_min_experts": "frontier_min_experts",
    "frontier_routing_temperature": "routing_temperature",
    "frontier_exploration": "exploration_epsilon",
    "frontier_confidence_weight": "confidence_weight",
    "frontier_compute_cost": "expansion_compute_cost",
    "frontier_posterior_temperature": "posterior_temperature",
    "frontier_credible_mass": "credible_mass",
    "frontier_owner_confidence": "owner_confidence_threshold",
    "max_writes_per_sequence": "max_writes_per_sequence",
}
_MERGE_ARG_TO_CHECKPOINT = {
    "merge_min_replay": "min_replay",
    "merge_stale_weight": "stale_weight",
    "merge_dynamics_weight": "dynamics_weight",
    "merge_loss_weight": "loss_weight",
    "merge_gate_temperature": "gate_temperature",
    "merge_budget_ratio": "budget_ratio",
    "merge_dual_lr": "dual_lr",
    "merge_dual_initial": "dual_initial",
}


def _checkpoint_config_value(payload: Mapping, dest: str):
    """Read the persisted value corresponding to one CLI destination."""

    training = payload.get("training_config", {})
    wake = payload.get("wake_config", {})
    sleep = payload.get("sleep_config", {})
    structure = payload.get("structure_config", {})
    model = payload.get("model_config", {})
    frontier = (
        model.get("frontier_routing_config", {})
        if isinstance(model, Mapping)
        else {}
    )
    if not all(isinstance(section, Mapping) for section in (training, wake, sleep, structure, model, frontier)):
        return None

    if dest == "controller_heads":
        value = training.get("controller_train_heads")
        if value is None:
            return None
        return ",".join(str(item) for item in value)
    if dest in {"controller_target_version", "controller_write_ranking", "seed", "learning_rate", "weight_decay", "grad_clip", "sleep_every", "router_lr_scale", "evaluation_ablation"}:
        return training.get(
            "controller_target_version" if dest == "controller_target_version" else dest
        )
    if dest in {"z_dim", "node_dim", "memory_key_dim", "num_basis", "decays", "semantic_blend"}:
        if dest == "semantic_blend":
            return model.get(dest)
        return model.get(dest)
    if dest in _FRONTIER_ARG_TO_CHECKPOINT:
        return frontier.get(_FRONTIER_ARG_TO_CHECKPOINT[dest])
    if dest in _WAKE_ARG_TO_CHECKPOINT:
        return wake.get(_WAKE_ARG_TO_CHECKPOINT[dest])
    if dest.startswith("prototype_") or dest.startswith("count_"):
        return wake.get(dest)
    if dest.startswith("route_"):
        return wake.get(dest, wake.get(_WAKE_ARG_TO_CHECKPOINT.get(dest, "")))
    if dest == "residual_init_rank":
        return wake.get("route_probe_residual_rank")
    if dest == "residual_init_grad_clip":
        return wake.get("route_probe_residual_grad_clip")
    if dest in {"router_init_gain", "leaf_symmetry_scale", "residual_init_scale", "alignment_epochs", "alignment_batch_size", "alignment_lr", "alignment_weight_decay", "alignment_temperature", "alignment_grad_clip", "allow_root_only_alignment"}:
        # These are fresh-initialisation choices. They have no effect after a
        # full trainer restore, so older checkpoints legitimately omit them.
        return None
    if dest == "prune_warmup_epochs":
        return structure.get(dest)
    if dest in _MERGE_ARG_TO_CHECKPOINT:
        merge_kwargs = structure.get("merge_kwargs", {})
        if not isinstance(merge_kwargs, Mapping):
            return None
        return merge_kwargs.get(_MERGE_ARG_TO_CHECKPOINT[dest])
    if dest in {"light_replay_budget", "split_min_structural_strength", "split_min_effective_sample_size", "split_route_loss_weight", "split_anchor_weight", "deep_availability_tau", "deep_probe_interval", "deep_computation_cost", "deep_prior_probability", "deep_prior_weight", "deep_evidence_budget", "topology_inertia_strength", "topology_inertia_tau"}:
        return sleep.get(dest)
    return None


def _cl_config_value(payload: Mapping, dest: str):
    """Read a value from the learner-level ``cl_config.json`` schema."""

    training = payload.get("training", {})
    if not isinstance(training, Mapping):
        training = {}
    if dest == "epochs":
        return payload.get("epochs_per_task", payload.get("epochs"))
    if dest == "cold_start_epochs":
        return payload.get("cold_start_epochs")
    if dest in {"learning_rate", "weight_decay", "grad_clip", "sleep_every", "seed", "evaluation_ablation", "controller_target_version", "controller_write_ranking"}:
        section = training
        if dest in section:
            return section[dest]
        optimizer = payload.get("optimizer", {})
        if not isinstance(optimizer, Mapping):
            optimizer = {}
        if dest in optimizer:
            return optimizer[dest]
    if dest == "controller_heads":
        value = training.get("controller_train_heads")
        if value is not None:
            return ",".join(str(item) for item in value)
    if dest == "residual_init_rank":
        wake = payload.get("wake", {})
        return (
            wake.get("route_probe_residual_rank", wake.get(dest))
            if isinstance(wake, Mapping)
            else None
        )
    if dest == "residual_init_grad_clip":
        wake = payload.get("wake", {})
        return (
            wake.get("route_probe_residual_grad_clip", wake.get(dest))
            if isinstance(wake, Mapping)
            else None
        )
    if dest in {"z_dim", "node_dim", "memory_key_dim", "num_basis", "decays", "semantic_blend"}:
        model = payload.get("model", {})
        return model.get(dest) if isinstance(model, Mapping) else None
    if dest in _FRONTIER_ARG_TO_CHECKPOINT:
        frontier = payload.get("frontier", {})
        if not isinstance(frontier, Mapping):
            return None
        return frontier.get(
            _FRONTIER_ARG_TO_CHECKPOINT[dest],
            frontier.get(dest),
        )
    if dest in _WAKE_ARG_TO_CHECKPOINT:
        wake = payload.get("wake", {})
        if not isinstance(wake, Mapping):
            return None
        return wake.get(
            _WAKE_ARG_TO_CHECKPOINT[dest],
            wake.get(dest),
        )
    for section_name in ("wake", "sleep", "structure", "training"):
        section = payload.get(section_name, {})
        if not isinstance(section, Mapping):
            continue
        if dest in section:
            return section[dest]
    if dest in _MERGE_ARG_TO_CHECKPOINT:
        structure = payload.get("structure", {})
        merge_kwargs = (
            structure.get("merge_kwargs", {})
            if isinstance(structure, Mapping)
            else {}
        )
        return (
            merge_kwargs.get(_MERGE_ARG_TO_CHECKPOINT[dest])
            if isinstance(merge_kwargs, Mapping)
            else None
        )
    return None


def _resolve_cli_values(args, *, resume_payload: Mapping | None, cl_config: Mapping | None) -> None:
    """Resolve fresh defaults while inheriting omitted resume options."""

    parser_defaults = getattr(args, "_parser_defaults", {})
    for dest in _PERSISTENT_CONFIG_DESTS:
        if getattr(args, dest, None) is not None:
            continue
        value = None
        if resume_payload is not None:
            value = _checkpoint_config_value(resume_payload, dest)
        if value is None and cl_config is not None and resume_payload is None:
            value = _cl_config_value(cl_config, dest)
        if value is None:
            value = parser_defaults.get(dest)
        if dest == "decays" and value is not None:
            value = list(_normalise_config_value(value))
        setattr(args, dest, value)

    if args.epochs is None:
        args.epochs = (
            _cl_config_value(cl_config, "epochs")
            if cl_config is not None and resume_payload is None
            else None
        )
        if args.epochs is None:
            args.epochs = parser_defaults.get("epochs", 20)
    if args.cold_start_epochs is None:
        args.cold_start_epochs = (
            _cl_config_value(cl_config, "cold_start_epochs")
            if cl_config is not None and resume_payload is None
            else None
        )
        if args.cold_start_epochs is None:
            args.cold_start_epochs = parser_defaults.get(
                "cold_start_epochs", 5
            )
    if args.seed is None:
        args.seed = parser_defaults.get("seed", 0)

    # A checkpoint's RNG is restored by from_checkpoint(); the value below is
    # only used for epoch-indexed shuffling and therefore must match it.
    saved_training = (
        resume_payload.get("training_config", {})
        if resume_payload is not None
        else {}
    )
    if (
        isinstance(saved_training, Mapping)
        and "seed" in saved_training
    ):
        if "seed" not in getattr(args, "_explicit_options", set()):
            args.seed = int(saved_training["seed"])


def _check_resume_config_drift(args, payload: Mapping) -> None:
    """Reject implicit/accidental learner-config changes at a stage boundary."""

    mismatches = []
    for dest in sorted(
        set(getattr(args, "_explicit_options", set()))
        .intersection(_PERSISTENT_CONFIG_DESTS)
    ):
        expected = _checkpoint_config_value(payload, dest)
        if expected is None:
            continue
        actual = getattr(args, dest)
        if dest == "controller_heads" and isinstance(expected, (tuple, list)):
            expected = ",".join(str(item) for item in expected)
        if _normalise_config_value(actual) != _normalise_config_value(expected):
            mismatches.append(
                f"{dest}: checkpoint={expected!r}, cli={actual!r}"
            )

    saved_protocol = payload.get("cl_protocol", {})
    if not isinstance(saved_protocol, Mapping):
        saved_protocol = {}
    if args.cl_config and saved_protocol.get("config_sha256"):
        actual_sha = file_sha256(args.cl_config)
        if actual_sha != saved_protocol["config_sha256"]:
            mismatches.append(
                "cl_config.json SHA-256 differs from the checkpoint"
            )
    if args.benchmark_manifest and saved_protocol.get("benchmark_sha256"):
        actual_sha = file_sha256(args.benchmark_manifest)
        if actual_sha != saved_protocol["benchmark_sha256"]:
            mismatches.append(
                "benchmark_manifest.json SHA-256 differs from the checkpoint"
            )
    if mismatches and not args.allow_config_override:
        raise RuntimeError(
            "CL configuration drift detected while resuming: "
            + "; ".join(mismatches)
            + "; pass --allow-config-override to continue"
        )
    args._config_override_used = bool(mismatches)


def _apply_cl_metadata(trainer, args) -> None:
    """Attach immutable benchmark/config provenance to future checkpoints."""

    if args.cl_config:
        path = Path(args.cl_config).expanduser().resolve()
        trainer.training_config.cl_config_path = str(path)
        trainer.training_config.cl_config_sha256 = file_sha256(path)
    if args.benchmark_manifest:
        path = Path(args.benchmark_manifest).expanduser().resolve()
        trainer.training_config.benchmark_manifest_path = str(path)
        trainer.training_config.benchmark_manifest_sha256 = file_sha256(path)
    if args.cl_task_id is not None:
        trainer.training_config.cl_task_id = args.cl_task_id
    if args.cl_previous_checkpoint:
        trainer.training_config.cl_previous_checkpoint = str(
            Path(args.cl_previous_checkpoint).expanduser().resolve()
        )
    if getattr(args, "_config_override_used", False):
        trainer.training_config.cl_config_override = True
    elif getattr(args, "resume", None) is None:
        trainer.training_config.cl_config_override = False


def _start_cl_stage(trainer, args, resume_payload: Mapping | None) -> None:
    """Reset validation selection only when a new CL task starts.

    The learner state, global epoch counter, sleep state, optimizer and
    topology remain restored.  Validation history is task-local because a
    previous task's validation score cannot select this task's checkpoint.
    """

    if args.cl_task_id is None or resume_payload is None:
        return
    saved_protocol = resume_payload.get("cl_protocol", {})
    if not isinstance(saved_protocol, Mapping):
        return
    saved_task = saved_protocol.get("task_id")
    if saved_task is not None and int(saved_task) != int(args.cl_task_id):
        trainer.validation_history = []
        trainer.best_validation = None


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the Hawkes Memory Tree")
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--validation-data-path",
        default=None,
        help="Task-local validation CSV used only for checkpoint selection.",
    )
    parser.add_argument(
        "--cl-config", default=None,
        help="Learner-level continual-learning configuration JSON.",
    )
    parser.add_argument(
        "--benchmark-manifest", default=None,
        help="Canonical benchmark manifest recorded in each checkpoint.",
    )
    parser.add_argument("--cl-task-id", type=int, default=None)
    parser.add_argument("--cl-previous-checkpoint", default=None)
    parser.add_argument(
        "--allow-config-override",
        action="store_true",
        help="Permit explicit learner-config changes when resuming.",
    )
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default=None
    )
    parser.add_argument("--checkpoint", default="checkpoints/memory_tree.pt")
    parser.add_argument("--best-checkpoint", default=None)
    parser.add_argument(
        "--initial-checkpoint-output",
        default=None,
        help=(
            "Save the freshly initialized model before the first training "
            "epoch, for a fixed continual-learning scratch/FWT baseline."
        ),
    )
    parser.add_argument("--validation-history-path", default=None)
    parser.add_argument("--controller-diagnostics-path", default=None)
    parser.add_argument(
        "--unified-topology-log-path",
        default=None,
        help=(
            "Plain-text per-epoch Split candidate diagnostics. Defaults to "
            "<checkpoint_stem>_unified_topology.log."
        ),
    )
    parser.add_argument(
        "--controller-v4-fresh", action="store_true",
        help="Enable the controller validation workflow and reject --resume.",
    )
    parser.add_argument(
        "--controller-base-checkpoint", default=None,
        help="Warm-start strict Controller-only training from a v4-v6 checkpoint.",
    )
    parser.add_argument(
        "--controller-target-version", type=int, choices=(5, 6), default=5,
    )
    parser.add_argument(
        "--controller-heads", default="adapt,retrieve,write",
        help="Comma-separated trainable heads: adapt,retrieve,write.",
    )
    parser.add_argument(
        "--controller-write-ranking", action="store_true",
        help="Use sequence-relative physical-utility Write supervision.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--training-metrics-path",
        default=None,
        help=(
            "Compact JSON epoch log written after training. Defaults to "
            "<checkpoint_stem>_training_metrics.json."
        ),
    )
    parser.add_argument(
        "--training-plot-path",
        default=None,
        help=(
            "Multi-panel PNG written after training. Defaults to "
            "<checkpoint_stem>_training_curves.png."
        ),
    )
    parser.add_argument(
        "--no-training-plots",
        action="store_true",
        help="Disable the automatic post-training metrics log and PNG.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--sleep-every", type=int, default=1)
    parser.add_argument(
        "--evaluation-ablation",
        choices=(
            "full", "no_working", "no_episodic", "fixed_topology",
            "no_sleep", "heuristic_controller", "no_merge_prune",
        ),
        default="full",
        help="Mechanism configuration used by independently dispatched evaluations.",
    )
    parser.add_argument("--cold-start-epochs", type=int, default=5)
    parser.add_argument(
        "--prototype-duplicate-threshold", type=float, default=0.98,
        help=(
            "Cold-start duplicate cosine prior; mode-local Q80 distances take "
            "over after calibration."
        ),
    )
    parser.add_argument(
        "--prototype-duplicate-quantile", type=float, default=0.85,
        help="Quantile for the accepted-sample calibrated duplicate radius.",
    )
    parser.add_argument(
        "--prototype-mode-threshold", type=float, default=0.90,
        help=(
            "Cold-start dynamics cosine prior; mode-local Q95 distances take "
            "over after calibration."
        ),
    )
    parser.add_argument(
        "--prototype-mode-capacity", type=int, default=12,
        help="Maximum high-resolution prototypes retained per active dynamics mode.",
    )
    parser.add_argument(
        "--prototype-context-alias-capacity", type=int, default=3,
        help="Maximum retrieval/context aliases retained per law prototype.",
    )
    parser.add_argument(
        "--count-similarity-low",
        type=float,
        default=0.35,
        help="Similarity at or below which a memory is not a recurrence match.",
    )
    parser.add_argument(
        "--count-similarity-high",
        type=float,
        default=0.65,
        help="Similarity at or above which a memory is a full recurrence match.",
    )
    parser.add_argument(
        "--count-exponent",
        type=float,
        default=2.0,
        help="Exponent applied to the compact-support recurrence match.",
    )
    parser.add_argument(
        "--count-saturation",
        type=float,
        default=3.0,
        help="Support scale for normalized local recurrence count.",
    )
    parser.add_argument(
        "--count-topk",
        type=int,
        default=None,
        help=(
            "Optional recurrence Top-K. Default None sums the complete "
            "128-capacity bank."
        ),
    )
    parser.add_argument(
        "--route-mi-weight",
        type=float,
        default=0.2,
        help=(
            "Weight of mutual information across observations of the same "
            "expanded local branch."
        ),
    )
    parser.add_argument(
        "--route-posterior-weight",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility option. Frontier posterior KL is "
            "diagnostic-only and never enters backward."
        ),
    )
    parser.add_argument(
        "--route-distill-weight",
        type=float,
        default=1.0,
        help=(
            "Weight of reliability-gated fixed-prior child-energy "
            "distillation."
        ),
    )
    parser.add_argument(
        "--route-mix-weight",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility option. Likelihood mixture is "
            "diagnostic-only and never enters backward."
        ),
    )
    parser.add_argument(
        "--route-energy-temperature",
        type=float,
        default=1.0,
        help="Temperature applied to per-leaf sequence NLL energies.",
    )
    parser.add_argument(
        "--route-encoder-warmup-epochs",
        type=int,
        default=2,
        help=(
            "Deprecated compatibility option. Encoder routing gradients now "
            "use continuous child-teacher reliability instead of epoch gates."
        ),
    )
    parser.add_argument(
        "--route-encoder-grad-scale",
        type=float,
        default=0.1,
        help="alpha_max for reliability-gated routing gradients into Encoder.",
    )
    parser.add_argument(
        "--route-encoder-reliability-decay",
        type=float,
        default=0.9,
        help="EMA decay for online child-teacher Encoder reliability.",
    )
    parser.add_argument(
        "--route-teacher-temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature of the fixed-prior, detached child-energy teacher."
        ),
    )
    parser.add_argument(
        "--route-probe-weight",
        type=float,
        default=0.1,
        help=(
            "Weight of the training-only Hawkes probe for unexpanded "
            "coarse frontier regions; set 0 to disable."
        ),
    )
    parser.add_argument(
        "--route-probe-leaves",
        type=int,
        default=2,
        help=(
            "Deprecated compatibility option. Each coarse region now probes "
            "ceil(number of descendant leaves / 2)."
        ),
    )
    parser.add_argument(
        "--route-probe-leaf-smoothing",
        type=float,
        default=0.05,
        help="Uniform floor mixed into conditional probe-leaf credit.",
    )
    parser.add_argument(
        "--route-probe-residual-temperature",
        type=float,
        default=1.0,
        help="tau_p for the counterfactual stop+leaf Hawkes teacher.",
    )
    parser.add_argument(
        "--route-probe-gain-temperature",
        type=float,
        default=0.1,
        help=(
            "tau_G for the counterfactual leaf-mixture energy used as "
            "observed frontier expansion gain."
        ),
    )
    parser.add_argument(
        "--route-probe-complexity-weight",
        type=float,
        default=0.01,
        help="Deprecated compatibility option; no longer used by the probe.",
    )
    parser.add_argument(
        "--route-balance-weight",
        type=float,
        default=0.05,
        help=(
            "Weak anti-collapse weight for KL(mean sequence routing || "
            "uniform leaves); set 0 to disable it."
        ),
    )
    parser.add_argument(
        "--route-balance-batch-size",
        type=int,
        default=64,
        help=(
            "Number of sequences per persistent-parameter optimizer step. "
            "Set at least the dataset size for one global step per epoch."
        ),
    )
    parser.add_argument(
        "--wake-wavefront-batch-size",
        type=int,
        default=64,
        help=(
            "Number of sequence-local working-memory rows advanced together "
            "at each Wake time position."
        ),
    )
    parser.add_argument(
        "--retrieval-microbatch",
        type=int,
        default=1024,
        help=(
            "Number of flat event rows per episodic read_packed call at "
            "Wake batch entry."
        ),
    )
    parser.add_argument(
        "--route-balance-max-steps",
        type=int,
        default=8,
        help=(
            "Deprecated compatibility option; global training now makes one "
            "optimizer step per cross-sequence batch."
        ),
    )
    parser.add_argument(
        "--route-balance-target-kl",
        type=float,
        default=0.1,
        help="Deprecated compatibility option; retained for old commands.",
    )
    parser.add_argument(
        "--router-lr-scale",
        type=float,
        default=1.0,
        help="Router learning-rate multiplier relative to the main optimizer.",
    )
    parser.add_argument(
        "--router-init-gain",
        type=float,
        default=0.05,
        help=(
            "Small Xavier gain for the local Compat(z_t, u_child) score head."
        ),
    )
    parser.add_argument(
        "--semantic-blend",
        type=float,
        default=0.1,
        help=(
            "Node semantic contribution alpha in "
            "(1-alpha)*cold_Hawkes + alpha*Hyper(u_n). "
            "Use 0 for the legacy exact Hawkes initialization."
        ),
    )
    parser.add_argument(
        "--leaf-symmetry-scale",
        type=float,
        default=0.0,
        help=(
            "Legacy random leaf-ID perturbation. Disabled by default; cannot "
            "be combined with --sequence-summary residual initialization."
        ),
    )
    parser.add_argument(
        "--sequence-summary",
        default=None,
        help=(
            "H-tree sequence_summary.csv containing leaf_position, cluster_id, "
            "and optionally sequence indices. Enables data-driven residual "
            "signature initialization."
        ),
    )
    parser.add_argument(
        "--residual-init-scale",
        type=float,
        default=0.01,
        help=(
            "epsilon_init multiplying mass-centered leaf residual prototypes. "
            "Active when --sequence-summary is provided."
        ),
    )
    parser.add_argument(
        "--residual-init-rank",
        type=int,
        default=2,
        help=(
            "Per-basis low-rank projection used for cold-start residual "
            "signatures; matches online Memory-write projection by default."
        ),
    )
    parser.add_argument(
        "--residual-init-grad-clip",
        type=float,
        default=0.0,
        help=(
            "Optional global norm bound for each sequence negative-gradient "
            "signature; default 0 implements the construction without clipping."
        ),
    )
    parser.add_argument(
        "--alignment-epochs",
        type=int,
        default=0,
        help=(
            "Initialization-only epochs for offline H-tree membership "
            "calibration of CausalPrefixEncoder + semantic compatibility."
        ),
    )
    parser.add_argument(
        "--alignment-batch-size",
        type=int,
        default=16,
        help="Sequence batch size for membership alignment.",
    )
    parser.add_argument(
        "--alignment-lr",
        type=float,
        default=1e-3,
        help="Learning rate used only during membership alignment.",
    )
    parser.add_argument(
        "--alignment-weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay used only during membership alignment.",
    )
    parser.add_argument(
        "--alignment-temperature",
        type=float,
        default=1.0,
        help="Temperature of the prior-independent semantic branch student.",
    )
    parser.add_argument(
        "--alignment-grad-clip",
        type=float,
        default=5.0,
        help="Gradient-norm bound for Encoder + compatibility calibration.",
    )
    parser.add_argument(
        "--allow-root-only-alignment",
        action="store_true",
        help=(
            "Treat requested alignment as a recorded no-op when the initial "
            "tree has only one root/leaf and therefore no routing branch. "
            "Intended for CL task-0 initialization."
        ),
    )
    parser.add_argument(
        "--prune-warmup-epochs",
        type=int,
        default=10,
        help="Disable leaf pruning for the first N epochs.",
    )
    parser.add_argument(
        "--merge-min-replay",
        type=int,
        default=8,
        help="Minimum child replay windows for a differentiable Merge pair.",
    )
    parser.add_argument(
        "--merge-stale-weight",
        type=float,
        default=0.2,
        help=(
            "Penalty weight for stale replay evidence in the Merge "
            "retention score."
        ),
    )
    parser.add_argument(
        "--merge-dynamics-weight",
        type=float,
        default=0.1,
        help=(
            "Penalty weight for Hawkes-law/TTP divergence in the Merge "
            "retention score."
        ),
    )
    parser.add_argument(
        "--merge-loss-weight",
        type=float,
        default=0.1,
        help="Weight of the differentiable prediction-compression objective.",
    )
    parser.add_argument(
        "--merge-gate-temperature",
        type=float,
        default=1.0,
        help="Temperature of the probabilistic keep-versus-parent gate.",
    )
    parser.add_argument(
        "--merge-budget-ratio",
        type=float,
        default=0.95,
        help="Initial global complexity budget as a fraction of keep-all cost.",
    )
    parser.add_argument(
        "--merge-dual-lr",
        type=float,
        default=1e-6,
        help="Projected dual-ascent rate for the global complexity budget.",
    )
    parser.add_argument(
        "--merge-dual-initial",
        type=float,
        default=0.0,
        help="Initial non-negative lambda_T for a fresh trainer.",
    )
    parser.add_argument(
        "--light-replay-budget",
        type=int,
        default=32,
        help=(
            "Global Light Sleep evidence budget per epoch. This bounds "
            "memory inspection independently of total Memory Bank size."
        ),
    )
    parser.add_argument(
        "--split-min-structural-strength",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility value; Split decisions use predictive "
            "objective competition and do not apply this threshold."
        ),
    )
    parser.add_argument(
        "--split-min-effective-sample-size",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility value; Split decisions use predictive "
            "objective competition and do not apply this threshold."
        ),
    )
    parser.add_argument(
        "--split-route-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Weight of the standalone Bank-to-router distillation loss during "
            "Split fitting; it is excluded from structural gain."
        ),
    )
    parser.add_argument(
        "--split-anchor-weight",
        type=float,
        default=1e-2,
        help=(
            "Anchor weight for Deep refinement around the frozen Bank child "
            "initialization."
        ),
    )
    parser.add_argument(
        "--deep-availability-tau",
        "--deep-cooldown-tau",
        "--deep-min-interval",
        dest="deep_availability_tau",
        type=float,
        default=3.0,
        help=(
            "Time constant for recovery of Deep availability after a Deep "
            "evaluation. Older cooldown/min-interval flags are aliases."
        ),
    )
    parser.add_argument(
        "--deep-probe-interval",
        type=int,
        default=5,
        help=(
            "Evaluate non-committing shadow topology proposals every N Light "
            "cycles so the differentiable Deep gate receives a value target."
        ),
    )
    parser.add_argument(
        "--deep-computation-cost",
        type=float,
        default=0.05,
        help="Expected-computation penalty for opening Deep Sleep.",
    )
    parser.add_argument(
        "--deep-prior-probability",
        type=float,
        default=0.15,
        help="Sparse Bernoulli prior probability for Deep Sleep.",
    )
    parser.add_argument(
        "--deep-prior-weight",
        type=float,
        default=0.01,
        help="Weight of the Deep gate Bernoulli-prior KL.",
    )
    parser.add_argument(
        "--deep-evidence-budget",
        type=int,
        default=32,
        help=(
            "Maximum replay memories inspected for each Deep Sleep structural "
            "proposal."
        ),
    )
    parser.add_argument(
        "--topology-inertia-strength",
        type=float,
        default=0.01,
        help=(
            "Maximum local gain penalty immediately after a topology edit."
        ),
    )
    parser.add_argument(
        "--topology-inertia-tau",
        type=float,
        default=3.0,
        help="Deep-cycle decay time for the local topology edit penalty.",
    )
    parser.add_argument(
        "--hawkes-checkpoint",
        default=None,
        help=(
            "Reuse a checkpoint produced by Train.ConstructTree. When set, "
            "--cold-start-epochs is ignored."
        ),
    )
    parser.add_argument(
        "--semantic-smoke-only",
        action="store_true",
        help=(
            "Initialize H_tree + cold-start Hawkes semantics, verify routing "
            "and one event NLL, then exit without Wake/Sleep training."
        ),
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional smoke-test limit applied after loading the raw CSV.",
    )
    parser.add_argument(
        "--max-events-per-sequence",
        type=int,
        default=None,
        help="Optional smoke-test prefix length for each loaded sequence.",
    )
    parser.add_argument("--z-dim", type=int, default=50)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--memory-key-dim", type=int, default=64)
    parser.add_argument(
        "--tree-init-depth",
        type=int,
        default=None,
        help=(
            "Initial complete Memory-tree depth. Defaults to 0 with --h-tree "
            "(then topology is reconstructed from encoder node_ids), otherwise 1."
        ),
    )
    parser.add_argument(
        "--frontier-budget",
        type=int,
        default=4,
        help="Maximum active coarse-to-fine experts per prediction.",
    )
    parser.add_argument(
        "--frontier-min-experts",
        type=int,
        default=2,
        help="Minimum computed frontier width before utility-based stopping.",
    )
    parser.add_argument(
        "--frontier-routing-temperature",
        type=float,
        default=1.5,
        help="Local left/right soft-routing temperature.",
    )
    parser.add_argument(
        "--frontier-exploration",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility option; exact fixed-prior routing "
            "does not apply a post-softmax exploration mixture."
        ),
    )
    parser.add_argument(
        "--frontier-confidence-weight", type=float, default=0.25
    )
    parser.add_argument(
        "--frontier-compute-cost", type=float, default=0.05
    )
    parser.add_argument(
        "--frontier-posterior-temperature", type=float, default=1.0
    )
    parser.add_argument(
        "--frontier-credible-mass", type=float, default=0.90
    )
    parser.add_argument(
        "--frontier-owner-confidence", type=float, default=0.80
    )
    parser.add_argument(
        "--max-writes-per-sequence", type=int, default=8
    )
    parser.add_argument("--num-basis", type=int, default=2)
    parser.add_argument("--decays", type=float, nargs="+", default=[0.5, 1.5])
    parser.add_argument(
        "--h-tree",
        default=None,
        help=(
            "Optional path to AttenEncoderMain_v1 --node_only output (.pt). "
            "Copies H_tree_refined into tree.node_emb; --node-dim must equal d_model."
        ),
    )
    parser.add_argument("--device", default=None)
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser_defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest in _PERSISTENT_CONFIG_DESTS
        or action.dest in {"epochs", "cold_start_epochs"}
    }
    for action in parser._actions:
        if action.dest in _PERSISTENT_CONFIG_DESTS or action.dest in {
            "epochs", "cold_start_epochs"
        }:
            action.default = None
    args = parser.parse_args(argv_list)
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    explicit = set()
    for token in argv_list:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            explicit.add(option_to_dest[option])
    args._explicit_options = frozenset(explicit)
    args._parser_defaults = parser_defaults
    return args


@torch.no_grad()
def _run_semantic_smoke_test(
    tree: HawkesTree,
    hawkes: HawkesFamily,
    dataset: Sequence[Dict[str, Tensor]],
    *,
    semantic_blend: float,
) -> None:
    """Validate the static semantic bridge without mutating online memory."""
    if not dataset or not dataset[0]["times"].numel():
        raise ValueError("semantic smoke test requires at least one event")

    target = torch.cat(
        [hawkes.raw_mu.reshape(-1), hawkes.raw_W.reshape(-1)], dim=0
    )
    node_theta = torch.stack(
        [tree.semantic_theta(node_id) for node_id in tree.all_node_ids], dim=0
    )
    expected_theta = torch.stack(
        [
            torch.lerp(
                target,
                tree.base_semantic_theta(node_id),
                float(semantic_blend),
            )
            for node_id in tree.all_node_ids
        ],
        dim=0,
    )
    semantic_error = float(
        (node_theta - expected_theta).abs().max().cpu()
    )
    cold_deviation = float((node_theta - target).abs().max().cpu())

    batch_size = 2
    z_t = torch.zeros(batch_size, tree.z_dim, device=tree._device_anchor.device)
    output = tree(
        z_t,
        decays=hawkes.decays,
        update_memory_state=False,
    )
    responsibility = output["r"]
    route_error = float(
        (responsibility.sum(dim=-1) - 1.0).abs().max().cpu()
    )
    effective = output["effective_params"].select(0)
    effective_error = max(
        float((effective.raw_mu - hawkes.raw_mu).abs().max().cpu()),
        float((effective.raw_W - hawkes.raw_W).abs().max().cpu()),
    )

    sequence = {
        key: value.to(tree._device_anchor.device)
        for key, value in dataset[0].items()
    }
    event_index = min(1, int(sequence["times"].numel()) - 1)
    event_nll = hawkes.event_NLL(sequence, effective, event_index)

    finite = (
        torch.isfinite(node_theta).all()
        and torch.isfinite(responsibility).all()
        and torch.isfinite(effective.mu).all()
        and torch.isfinite(effective.W).all()
        and torch.isfinite(event_nll)
    )
    tolerance = 1e-5
    if not bool(finite):
        raise FloatingPointError("semantic smoke test produced NaN or Inf")
    if semantic_error > tolerance:
        raise RuntimeError(
            "semantic blend initialization mismatch: "
            f"max_abs_error={semantic_error:.3e}"
        )
    if route_error > tolerance:
        raise RuntimeError(
            f"routing probabilities do not sum to one: max_error={route_error:.3e}"
        )

    print("=" * 60)
    print("[Semantic Smoke] PASS")
    print(
        f"  topology       : {len(tree.all_node_ids)} nodes, "
        f"{len(tree.leaf_ids)} leaves"
    )
    print(f"  semantic blend : {semantic_blend:.4f}")
    print(f"  blend error    : {semantic_error:.3e}")
    print(f"  cold deviation : {cold_deviation:.3e}")
    print(f"  effective delta: {effective_error:.3e}")
    print(f"  route sum error: {route_error:.3e}")
    print(f"  event NLL      : {float(event_nll.cpu()):.6f}")
    print(
        f"  mu range       : [{float(effective.mu.min().cpu()):.6e}, "
        f"{float(effective.mu.max().cpu()):.6e}]"
    )
    print("=" * 60)


@torch.no_grad()
def _leaf_spectral_radius_summary(
    tree: HawkesTree,
    hawkes: HawkesFamily,
) -> Dict[str, float]:
    """Measure Hawkes branching stability of every current leaf."""
    radii = []
    decays = hawkes.decays.to(tree._device_anchor.device)
    D = hawkes.num_types
    for leaf_id in tree.leaf_ids:
        theta = tree.semantic_theta(leaf_id)
        W = F.softplus(
            theta[D:].reshape(D, D, hawkes.num_basis)
        )
        branching = (
            W / decays.reshape(1, 1, -1)
        ).sum(dim=-1)
        radius = torch.linalg.eigvals(branching).abs().max().real
        radii.append(float(radius.cpu()))
    return {
        "min": min(radii) if radii else 0.0,
        "mean": sum(radii) / max(len(radii), 1),
        "max": max(radii) if radii else 0.0,
    }


def _apply_evaluation_ablation(trainer, name: str) -> None:
    trainer.training_config.evaluation_ablation = str(name)
    if name == "full":
        return
    if name == "no_working":
        trainer.tree.working_memory.eta = 0.0
        trainer.tree.working_memory.reset()
        trainer.controller.eta_fast = 0.0
    elif name == "no_episodic":
        trainer.controller.set_calibration_thresholds(
            retrieve=1.01, adapt=trainer.controller.calibration_dict()["adapt_threshold"], write=1.01
        )
    elif name == "no_sleep":
        trainer.training_config.sleep_every = trainer.training_config.epochs + 1
    elif name == "fixed_topology":
        # Light Sleep remains enabled for semantic consolidation, while Deep
        # structural commits and pruning/merge are made unreachable.
        trainer.sleep_config.deep_probe_interval = trainer.training_config.epochs + 1
        trainer.controller.split_enabled = False
        trainer.structure_config.prune_warmup_epochs = trainer.training_config.epochs + 1
        trainer.structure_config.merge_kwargs["min_replay"] = 10**12
    elif name == "no_merge_prune":
        trainer.structure_config.prune_warmup_epochs = trainer.training_config.epochs + 1
        trainer.structure_config.merge_kwargs["min_replay"] = 10**12
    elif name == "heuristic_controller":
        # The monotone bootstrap controller is the observable-feature
        # heuristic; freezing it removes learned counterfactual control.
        trainer.controller.set_calibration_thresholds(0.5, 0.5, 0.75)
        trainer.controller.exploration_rate = 0.0
        for parameter in trainer.controller.parameters():
            parameter.requires_grad_(False)
    else:
        raise ValueError(f"unknown evaluation ablation: {name}")


def main() -> None:
    args = _parse_args()
    resume_source = args.resume or args.controller_base_checkpoint
    resume_payload = (
        _load_checkpoint_payload(resume_source)
        if resume_source is not None
        else None
    )
    cl_config = (
        _load_json_object(args.cl_config)
        if args.cl_config is not None
        else None
    )
    _resolve_cli_values(
        args,
        resume_payload=resume_payload,
        cl_config=cl_config,
    )
    if resume_payload is not None:
        _check_resume_config_drift(args, resume_payload)
    if args.cl_task_id is not None:
        if args.cl_task_id < 0:
            raise ValueError("--cl-task-id must be non-negative")
        if args.validation_data_path is None:
            raise RuntimeError(
                "continual stages require --validation-data-path so checkpoint "
                "selection cannot fall back to training or test data"
            )
        if args.benchmark_manifest is None:
            raise RuntimeError(
                "continual stages require --benchmark-manifest"
            )
    if args.cl_task_id is not None and args.cl_task_id > 0:
        if args.resume is None:
            raise RuntimeError(
                f"continual task {args.cl_task_id} requires --resume from "
                "the previous task checkpoint"
            )
        saved_protocol = (
            resume_payload.get("cl_protocol", {})
            if resume_payload is not None
            else {}
        )
        if isinstance(saved_protocol, Mapping):
            saved_task = saved_protocol.get("task_id")
            if saved_task is not None and int(saved_task) not in {
                int(args.cl_task_id) - 1,
                int(args.cl_task_id),
            }:
                raise RuntimeError(
                    f"continual task {args.cl_task_id} requires a checkpoint "
                    f"from task {int(args.cl_task_id) - 1}; got task {saved_task}"
                )
        if (
            args.cl_previous_checkpoint is not None
            and Path(args.cl_previous_checkpoint).expanduser().resolve()
            != Path(args.resume).expanduser().resolve()
        ):
            raise RuntimeError(
                "--resume must identify the same checkpoint as "
                "--cl-previous-checkpoint"
            )
    if not 0.0 <= args.semantic_blend <= 1.0:
        raise ValueError("--semantic-blend must lie in [0, 1]")
    if args.leaf_symmetry_scale < 0.0:
        raise ValueError("--leaf-symmetry-scale must be non-negative")
    if args.residual_init_scale < 0.0:
        raise ValueError("--residual-init-scale must be non-negative")
    if args.residual_init_rank < 0:
        raise ValueError("--residual-init-rank must be non-negative")
    if args.residual_init_grad_clip < 0.0:
        raise ValueError("--residual-init-grad-clip must be non-negative")
    if args.alignment_epochs < 0:
        raise ValueError("--alignment-epochs must be non-negative")
    if args.alignment_batch_size <= 0:
        raise ValueError("--alignment-batch-size must be positive")
    if args.alignment_lr <= 0.0:
        raise ValueError("--alignment-lr must be positive")
    if args.alignment_weight_decay < 0.0:
        raise ValueError("--alignment-weight-decay must be non-negative")
    if args.alignment_temperature <= 0.0:
        raise ValueError("--alignment-temperature must be positive")
    if args.alignment_grad_clip <= 0.0:
        raise ValueError("--alignment-grad-clip must be positive")
    if args.alignment_epochs > 0 and args.sequence_summary is None:
        raise ValueError(
            "--alignment-epochs requires --sequence-summary membership"
        )
    residual_initialization = (
        args.sequence_summary is not None
        and args.residual_init_scale > 0.0
    )
    if residual_initialization and args.leaf_symmetry_scale > 0.0:
        raise ValueError(
            "data-driven residual initialization and legacy random leaf "
            "perturbation are mutually exclusive"
        )
    if args.route_balance_weight < 0.0:
        raise ValueError("--route-balance-weight must be non-negative")
    if not 0.0 <= args.route_encoder_grad_scale <= 1.0:
        raise ValueError("--route-encoder-grad-scale must lie in [0, 1]")
    if not 0.0 <= args.route_encoder_reliability_decay < 1.0:
        raise ValueError(
            "--route-encoder-reliability-decay must lie in [0, 1)"
        )
    if args.route_teacher_temperature <= 0.0:
        raise ValueError("--route-teacher-temperature must be positive")
    if args.route_probe_weight < 0.0:
        raise ValueError("--route-probe-weight must be non-negative")
    if (
        not math.isfinite(args.split_route_loss_weight)
        or args.split_route_loss_weight < 0.0
    ):
        raise ValueError(
            "--split-route-loss-weight must be finite and non-negative"
        )
    split_anchor_weight = getattr(args, "split_anchor_weight", 1e-2)
    if (
        not math.isfinite(split_anchor_weight)
        or split_anchor_weight < 0.0
    ):
        raise ValueError(
            "--split-anchor-weight must be finite and non-negative"
        )
    if not 0.0 <= args.route_probe_leaf_smoothing < 1.0:
        raise ValueError(
            "--route-probe-leaf-smoothing must lie in [0, 1)"
        )
    if args.route_probe_residual_temperature <= 0.0:
        raise ValueError(
            "--route-probe-residual-temperature must be positive"
        )
    if args.route_probe_gain_temperature <= 0.0:
        raise ValueError("--route-probe-gain-temperature must be positive")
    if args.route_probe_complexity_weight < 0.0:
        raise ValueError(
            "--route-probe-complexity-weight must be non-negative"
        )
    if args.deep_availability_tau <= 0.0:
        raise ValueError("--deep-availability-tau must be positive")
    if args.deep_probe_interval <= 0:
        raise ValueError("--deep-probe-interval must be positive")
    if args.deep_computation_cost < 0.0:
        raise ValueError("--deep-computation-cost must be non-negative")
    if not 0.0 < args.deep_prior_probability < 1.0:
        raise ValueError(
            "--deep-prior-probability must lie in (0, 1)"
        )
    if args.deep_prior_weight < 0.0:
        raise ValueError("--deep-prior-weight must be non-negative")
    if args.topology_inertia_strength < 0.0:
        raise ValueError("--topology-inertia-strength must be non-negative")
    if args.topology_inertia_tau <= 0.0:
        raise ValueError("--topology-inertia-tau must be positive")
    if args.merge_min_replay < 0:
        raise ValueError("--merge-min-replay must be non-negative")
    if args.merge_stale_weight < 0.0:
        raise ValueError("--merge-stale-weight must be non-negative")
    if args.merge_dynamics_weight < 0.0:
        raise ValueError("--merge-dynamics-weight must be non-negative")
    if args.merge_loss_weight < 0.0:
        raise ValueError("--merge-loss-weight must be non-negative")
    if args.merge_gate_temperature <= 0.0:
        raise ValueError("--merge-gate-temperature must be positive")
    if not 0.0 < args.merge_budget_ratio <= 1.0:
        raise ValueError("--merge-budget-ratio must lie in (0, 1]")
    if args.merge_dual_lr < 0.0:
        raise ValueError("--merge-dual-lr must be non-negative")
    if args.merge_dual_initial < 0.0:
        raise ValueError("--merge-dual-initial must be non-negative")
    torch.manual_seed(args.seed)
    tree_init_depth = (
        args.tree_init_depth
        if args.tree_init_depth is not None
        else (0 if args.h_tree is not None else 1)
    )
    constructor = ConstructMemoryTree(
        data_path=args.data_path,
        num_basis=args.num_basis,
        decays=args.decays,
        z_dim=args.z_dim,
        node_dim=args.node_dim,
        memory_key_dim=args.memory_key_dim,
        tree_init_depth=tree_init_depth,
        device=args.device,
    )
    all_sequences = constructor.load_sequences()
    dataset = all_sequences
    validation_dataset = None
    manifest = None
    if args.validation_data_path is not None:
        if args.split_manifest is not None:
            raise ValueError(
                "--validation-data-path cannot be combined with --split-manifest; "
                "the continual protocol already declares task-local splits"
            )
        train_path = Path(args.data_path).expanduser().resolve()
        validation_path = Path(args.validation_data_path).expanduser().resolve()
        if train_path.name != "train.csv":
            raise ValueError(
                "continual training updates must read a file named train.csv"
            )
        if validation_path.name not in {"val.csv", "validation.csv"}:
            raise ValueError(
                "checkpoint selection must read a file named val.csv or validation.csv"
            )
        validation_constructor = ConstructMemoryTree(
            data_path=validation_path,
            num_basis=args.num_basis,
            decays=args.decays,
            z_dim=args.z_dim,
            node_dim=args.node_dim,
            memory_key_dim=args.memory_key_dim,
            tree_init_depth=tree_init_depth,
            device=args.device,
        )
        validation_dataset = validation_constructor.load_sequences(
            type_to_index=constructor.type_to_index
        )
        print(
            f"[Task-local validation] train={train_path} "
            f"validation={validation_path} "
            f"sequences={len(validation_dataset)}"
        )
    if args.split_manifest is not None:
        if args.split is None:
            raise ValueError("--split is required with --split-manifest")
        manifest = load_split_manifest(
            args.split_manifest, data_path=args.data_path
        )
        dataset = select_sequences(all_sequences, manifest, args.split)
        if args.split == "train":
            validation_all = select_sequences(
                all_sequences, manifest, "validation"
            )
            validation_dataset = []
            if validation_all and all("cluster_id" in row for row in validation_all):
                validation_groups = defaultdict(list)
                for sequence in validation_all:
                    cluster = int(torch.as_tensor(sequence["cluster_id"]).item())
                    validation_groups[cluster].append(sequence)
                for cluster in sorted(validation_groups):
                    rows = sorted(
                        validation_groups[cluster],
                        key=lambda row: int(
                            torch.as_tensor(row["source_index"]).item()
                        ),
                    )
                    validation_dataset.extend(rows[:5])
                expected = sum(min(5, len(rows)) for rows in validation_groups.values())
            else:
                # Real benchmarks do not have synthetic regime labels.  Use a
                # deterministic validation subset without manufacturing an
                # oracle cluster column or leaking test data into selection.
                validation_dataset = sorted(
                    validation_all,
                    key=lambda row: int(torch.as_tensor(row["source_index"]).item()),
                )
                expected = min(64, len(validation_dataset))
                validation_dataset = validation_dataset[:expected]
            if len(validation_dataset) != expected:
                raise ValueError(
                    "validation subset construction failed: "
                    f"selected {len(validation_dataset)} of {expected}"
                )
            print(
                f"[Controller validation] sequences={expected} "
                "(deterministic manifest validation subset)"
            )
        print(
            f"[DataSplit] split={args.split} sequences={len(dataset)} "
            f"manifest={args.split_manifest}"
        )
    elif args.split is not None:
        raise ValueError("--split-manifest is required with --split")
    if args.max_sequences is not None:
        if args.max_sequences <= 0:
            raise ValueError("--max-sequences must be positive")
        dataset = dataset[: args.max_sequences]
    if args.max_events_per_sequence is not None:
        if args.max_events_per_sequence <= 0:
            raise ValueError("--max-events-per-sequence must be positive")

    def _smoke_limit(sequences):
        if sequences is None:
            return None
        limited = list(sequences)
        if args.max_sequences is not None:
            limited = limited[: args.max_sequences]
        if args.max_events_per_sequence is None:
            return limited
        prefixes = []
        for sequence in limited:
            event_count = min(
                args.max_events_per_sequence,
                int(sequence["times"].numel()),
            )
            times = sequence["times"][:event_count]
            prefixes.append({
                "times": times,
                "types": sequence["types"][:event_count],
                "T": times[-1],
                **{
                    key: sequence[key]
                    for key in ("source_index", "cluster_id")
                    if key in sequence
                },
            })
        return prefixes

    dataset = _smoke_limit(dataset)
    # Validation is the expensive part of a one-epoch HM smoke run because it
    # evaluates several frozen/online protocols.  Apply the same explicit
    # smoke limits here; formal runs do not pass either option and are unchanged.
    validation_dataset = _smoke_limit(validation_dataset)
    if args.max_sequences is not None or args.max_events_per_sequence is not None:
        print(
            f"[Smoke limit] sequences={len(dataset)}, "
            f"events={sum(int(item['times'].numel()) for item in dataset)}, "
            f"validation_sequences={len(validation_dataset or [])}, "
            f"validation_events={sum(int(item['times'].numel()) for item in (validation_dataset or []))}"
        )
    if args.controller_base_checkpoint is not None and (
        args.resume is not None or args.controller_v4_fresh
    ):
        raise ValueError(
            "--controller-base-checkpoint is mutually exclusive with "
            "--resume and --controller-v4-fresh"
        )
    if args.controller_base_checkpoint is not None:
        trainer = MemoryTreeTrainer.from_checkpoint(
            args.controller_base_checkpoint,
            device=constructor.device,
        )
        trainer.training_config.epochs = args.epochs
        trainer.training_config.checkpoint_path = args.checkpoint
        trainer.training_config.best_checkpoint_path = args.best_checkpoint
        trainer.training_config.validation_history_path = (
            args.validation_history_path
        )
        trainer.training_config.controller_diagnostics_path = (
            args.controller_diagnostics_path
        )
        trainer.training_config.unified_topology_log_path = (
            args.unified_topology_log_path
        )
        trainer.training_config.training_metrics_path = args.training_metrics_path
        trainer.training_config.training_plot_path = args.training_plot_path
        trainer.training_config.plot_after_training = not args.no_training_plots
        trainer.training_config.seed = args.seed
        trainer.training_config.learning_rate = args.learning_rate
        trainer.training_config.weight_decay = args.weight_decay
        trainer.training_config.grad_clip = args.grad_clip
        trainer.training_config.sleep_every = args.sleep_every
        trainer.training_config.evaluation_ablation = args.evaluation_ablation
        heads = tuple(
            value.strip()
            for value in args.controller_heads.split(",")
            if value.strip()
        )
        allowed_heads = {"adapt", "retrieve", "write"}
        if not heads or set(heads).difference(allowed_heads):
            raise ValueError(
                "--controller-heads must contain only adapt,retrieve,write"
            )
        if args.controller_write_ranking and set(heads) != {"write"}:
            raise ValueError(
                "--controller-write-ranking requires --controller-heads write"
            )
        trainer.training_config.controller_write_ranking = bool(
            args.controller_write_ranking
        )
        trainer.prepare_controller_only_finetune(
            base_checkpoint=args.controller_base_checkpoint,
            target_version=args.controller_target_version,
            train_heads=heads,
        )
        trainer.controller_utility_replay.write_ranking_enabled = bool(
            args.controller_write_ranking
        )
        trainable_names = [
            f"{module_name}.{name}"
            for module_name, module in {
                "hawkes": trainer.hawkes,
                "encoder": trainer.encoder,
                "tree": trainer.tree,
                "controller": trainer.controller,
            }.items()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable_names or any(
            not name.startswith("controller.") for name in trainable_names
        ):
            raise RuntimeError(
                "Controller-only training exposed non-controller parameters: "
                f"{trainable_names}"
            )
        print(
            "[Controller] strict controller-only warm start; "
            f"trainable_tensors={len(trainable_names)} "
            f"frozen_sha256={trainer.training_config.frozen_state_sha256}"
        )
        _apply_cl_metadata(trainer, args)
        _apply_evaluation_ablation(trainer, args.evaluation_ablation)
        trainer.train(dataset, validation_dataset=validation_dataset)
        return
    if args.controller_v4_fresh and args.resume is not None:
        raise ValueError(
            "Controller v4 training must start from scratch; --resume is forbidden"
        )
    if args.resume is not None:
        trainer = MemoryTreeTrainer.from_checkpoint(
            args.resume,
            device=constructor.device,
        )
        _start_cl_stage(trainer, args, resume_payload)
        trainer.tree.configure_frontier_routing(
            config=FrontierRoutingConfig(
                frontier_budget=args.frontier_budget,
                frontier_min_experts=args.frontier_min_experts,
                routing_temperature=(
                    args.frontier_routing_temperature
                ),
                exploration_epsilon=args.frontier_exploration,
                confidence_weight=args.frontier_confidence_weight,
                expansion_compute_cost=args.frontier_compute_cost,
                posterior_temperature=(
                    args.frontier_posterior_temperature
                ),
                credible_mass=args.frontier_credible_mass,
                owner_confidence_threshold=(
                    args.frontier_owner_confidence
                ),
                max_writes_per_sequence=args.max_writes_per_sequence,
            ),
        )
        trainer.training_config.epochs = args.epochs
        trainer.training_config.checkpoint_path = args.checkpoint
        trainer.training_config.best_checkpoint_path = args.best_checkpoint
        trainer.training_config.validation_history_path = (
            args.validation_history_path
        )
        trainer.training_config.controller_diagnostics_path = (
            args.controller_diagnostics_path
        )
        trainer.training_config.unified_topology_log_path = (
            args.unified_topology_log_path
        )
        trainer.training_config.router_lr_scale = args.router_lr_scale
        trainer.training_config.controller_write_ranking = bool(
            args.controller_write_ranking
        )
        trainer.controller_utility_replay.write_ranking_enabled = bool(
            args.controller_write_ranking
        )
        trainer.training_config.seed = args.seed
        trainer.training_config.learning_rate = args.learning_rate
        trainer.training_config.weight_decay = args.weight_decay
        trainer.training_config.grad_clip = args.grad_clip
        trainer.training_config.sleep_every = args.sleep_every
        trainer.training_config.evaluation_ablation = args.evaluation_ablation
        trainer.training_config.plot_after_training = (
            not args.no_training_plots
        )
        trainer.training_config.training_metrics_path = (
            args.training_metrics_path
        )
        trainer.training_config.training_plot_path = args.training_plot_path
        trainer.wake_config.prototype_duplicate_threshold = (
            args.prototype_duplicate_threshold
        )
        trainer.wake_config.prototype_mode_threshold = args.prototype_mode_threshold
        trainer.wake_config.prototype_mode_capacity = args.prototype_mode_capacity
        trainer.wake_config.prototype_context_alias_capacity = (
            args.prototype_context_alias_capacity
        )
        trainer.wake_config.prototype_duplicate_quantile = (
            args.prototype_duplicate_quantile
        )
        trainer.tree.episodic_memory.configure_prototype_memory(
            duplicate_threshold=trainer.wake_config.prototype_duplicate_threshold,
            mode_threshold=trainer.wake_config.prototype_mode_threshold,
            duplicate_quantile=trainer.wake_config.prototype_duplicate_quantile,
            mode_capacity=trainer.wake_config.prototype_mode_capacity,
            context_alias_capacity=(
                trainer.wake_config.prototype_context_alias_capacity
            ),
        )
        trainer.wake_config.lambda_route_mi = args.route_mi_weight
        trainer.wake_config.lambda_route_posterior = (
            args.route_posterior_weight
        )
        trainer.wake_config.lambda_route_distill = (
            args.route_distill_weight
        )
        trainer.wake_config.lambda_route_mix = args.route_mix_weight
        trainer.wake_config.route_energy_temperature = (
            args.route_energy_temperature
        )
        trainer.wake_config.route_encoder_warmup_epochs = (
            args.route_encoder_warmup_epochs
        )
        trainer.wake_config.route_encoder_grad_scale = (
            args.route_encoder_grad_scale
        )
        trainer.wake_config.route_encoder_reliability_decay = (
            args.route_encoder_reliability_decay
        )
        trainer.wake_config.route_teacher_temperature = (
            args.route_teacher_temperature
        )
        trainer.wake_config.lambda_route_probe = args.route_probe_weight
        trainer.wake_config.route_probe_leaves = args.route_probe_leaves
        trainer.wake_config.route_probe_leaf_smoothing = (
            args.route_probe_leaf_smoothing
        )
        trainer.wake_config.route_probe_residual_temperature = (
            args.route_probe_residual_temperature
        )
        trainer.wake_config.route_probe_gain_temperature = (
            args.route_probe_gain_temperature
        )
        trainer.wake_config.route_probe_complexity_weight = (
            args.route_probe_complexity_weight
        )
        trainer.wake_config.route_probe_residual_rank = (
            args.residual_init_rank
        )
        trainer.wake_config.route_probe_residual_grad_clip = (
            args.residual_init_grad_clip
        )
        trainer.wake_config.lambda_route_balance = (
            args.route_balance_weight
        )
        trainer.wake_config.route_balance_batch_size = (
            args.route_balance_batch_size
        )
        trainer.wake_config.wake_wavefront_batch_size = (
            args.wake_wavefront_batch_size
        )
        trainer.wake_config.retrieval_microbatch = (
            args.retrieval_microbatch
        )
        trainer.wake_config.route_balance_max_steps = (
            args.route_balance_max_steps
        )
        trainer.wake_config.route_balance_target_kl = (
            args.route_balance_target_kl
        )
        trainer.wake_config.count_similarity_low = (
            args.count_similarity_low
        )
        trainer.wake_config.count_similarity_high = (
            args.count_similarity_high
        )
        trainer.wake_config.count_exponent = args.count_exponent
        trainer.wake_config.count_saturation = args.count_saturation
        trainer.wake_config.count_topk = args.count_topk
        trainer.controller.count_similarity_low = args.count_similarity_low
        trainer.controller.count_similarity_high = args.count_similarity_high
        trainer.controller.count_exponent = args.count_exponent
        trainer.controller.count_saturation = args.count_saturation
        trainer.controller.count_topk = args.count_topk
        trainer.sleep_config.light_replay_budget = args.light_replay_budget
        trainer.sleep_config.split_min_structural_strength = (
            args.split_min_structural_strength
        )
        trainer.sleep_config.split_min_effective_sample_size = (
            args.split_min_effective_sample_size
        )
        trainer.sleep_config.split_route_loss_weight = args.split_route_loss_weight
        trainer.sleep_config.split_anchor_weight = args.split_anchor_weight
        trainer.sleep_config.deep_availability_tau = (
            args.deep_availability_tau
        )
        trainer.sleep_config.deep_probe_interval = args.deep_probe_interval
        trainer.sleep_config.deep_computation_cost = (
            args.deep_computation_cost
        )
        trainer.sleep_config.deep_prior_probability = (
            args.deep_prior_probability
        )
        trainer.sleep_config.deep_prior_weight = args.deep_prior_weight
        trainer.sleep_config.deep_evidence_budget = args.deep_evidence_budget
        trainer.sleep_config.topology_inertia_strength = (
            args.topology_inertia_strength
        )
        trainer.sleep_config.topology_inertia_tau = args.topology_inertia_tau
        trainer.structure_config.prune_warmup_epochs = (
            args.prune_warmup_epochs
        )
        trainer.structure_config.merge_kwargs = {
            "min_replay": args.merge_min_replay,
            "gate_temperature": args.merge_gate_temperature,
            "loss_weight": args.merge_loss_weight,
            "budget_ratio": args.merge_budget_ratio,
            "dual_lr": args.merge_dual_lr,
            "dual_initial": args.merge_dual_initial,
            "stale_weight": args.merge_stale_weight,
            "dynamics_weight": args.merge_dynamics_weight,
            "normalize_by_events": True,
        }
        _apply_cl_metadata(trainer, args)
        trainer._reconcile_optimizer_parameters()
        _apply_evaluation_ablation(trainer, args.evaluation_ablation)
        trainer.train(dataset, validation_dataset=validation_dataset)
        return
    if args.hawkes_checkpoint is not None:
        hawkes, cold_start_payload = HawkesFamily.from_cold_start_checkpoint(
            args.hawkes_checkpoint,
            device=constructor.device,
        )
        if hawkes.num_types != len(constructor.event_types):
            raise ValueError(
                "Hawkes checkpoint/data event dimensions differ: "
                f"{hawkes.num_types} != {len(constructor.event_types)}"
            )
        if hawkes.num_basis != constructor.num_basis:
            raise ValueError(
                "Hawkes checkpoint/CLI num_basis differ: "
                f"{hawkes.num_basis} != {constructor.num_basis}"
            )
        constructor.hawkes_backbone = hawkes
        tree = constructor._build_memory_tree()
        result = cold_start_payload.get("training_result", {})
        print(
            f"[Hawkes] loaded cold-start checkpoint: {args.hawkes_checkpoint}"
        )
        if result:
            print(
                "[Hawkes] "
                f"best_val_nll/event={result.get('best_validation_nll', 'n/a')}, "
                f"spectral_radius={result.get('spectral_radius', 'n/a')}"
            )
    else:
        hawkes = constructor._build_hawkes_backbone()
        tree = constructor.hawkes_tree
    if args.hawkes_checkpoint is None and args.cold_start_epochs > 0:
        hawkes.cold_start(
            dataset,
            num_epochs=args.cold_start_epochs,
            checkpoint_path="checkpoints/hawkes_backbone_init.pt",
        )
    if tree is None:
        raise RuntimeError("tree construction failed")
    if args.h_tree is not None:
        from AttentionEncoderAdapter import initialize_tree_from_h_tree_file

        h_tree_payload = torch.load(
            args.h_tree, map_location="cpu", weights_only=False
        )
        h_tree_provenance = h_tree_payload.get("data_provenance")
        if h_tree_provenance is None:
            h_tree_provenance = {
                "evaluation_regime": h_tree_payload.get(
                    "evaluation_regime", "transductive"
                )
            }
        tree.data_provenance = dict(h_tree_provenance)
        if h_tree_provenance.get("evaluation_regime") == "strict_inductive":
            if manifest is None or args.split != "train":
                raise ValueError(
                    "strict_inductive H-tree training requires the same "
                    "--split-manifest with --split train"
                )
            if args.max_sequences is not None:
                raise ValueError(
                    "strict_inductive training cannot truncate manifest train IDs"
                )
            manifest_sha = file_sha256(args.split_manifest)
            if h_tree_provenance.get("manifest_sha256") != manifest_sha:
                raise ValueError("H-tree and Memory split manifest SHA-256 differ")
            expected = {
                "train": set(map(int, h_tree_provenance.get("train_source_ids", ()))),
                "validation": set(map(int, h_tree_provenance.get("validation_source_ids", ()))),
                "test": set(map(int, h_tree_provenance.get("test_source_ids", ()))),
            }
            actual = {
                name: set(map(int, manifest["splits"][name]))
                for name in ("train", "validation", "test")
            }
            if expected != actual:
                raise ValueError("H-tree provenance source IDs differ from Memory split")

        mapped = initialize_tree_from_h_tree_file(
            tree,
            args.h_tree,
            strict_coverage=True,
            synchronize_topology=True,
        )
        print(
            f"[H_tree] initialized {len(mapped)}/{len(tree.all_node_ids)} "
            f"node_emb from {args.h_tree}"
        )
    tree.configure_frontier_routing(
        config=FrontierRoutingConfig(
            frontier_budget=args.frontier_budget,
            frontier_min_experts=args.frontier_min_experts,
            routing_temperature=args.frontier_routing_temperature,
            exploration_epsilon=args.frontier_exploration,
            confidence_weight=args.frontier_confidence_weight,
            expansion_compute_cost=args.frontier_compute_cost,
            posterior_temperature=args.frontier_posterior_temperature,
            credible_mass=args.frontier_credible_mass,
            owner_confidence_threshold=args.frontier_owner_confidence,
            max_writes_per_sequence=args.max_writes_per_sequence,
        ),
    )
    print(
        "[Routing] "
        "mode=active_frontier "
        f"frontier={args.frontier_min_experts}..{args.frontier_budget} "
        f"temperature={args.frontier_routing_temperature:.3f} "
        f"exploration={args.frontier_exploration:.3f}"
    )
    # Residual initialization must be centered around the exact shared Hawkes
    # solution theta_0. Legacy runs without a sequence summary may still opt
    # into the older semantic blend.
    effective_semantic_blend = (
        0.0 if residual_initialization else args.semantic_blend
    )
    cold_target = tree.initialize_semantics_from_hawkes(
        hawkes,
        semantic_blend=effective_semantic_blend,
    )
    router_norms = tree.initialize_router_weights(
        gain=args.router_init_gain,
        seed=args.seed,
    )
    if args.semantic_smoke_only:
        _run_semantic_smoke_test(
            tree,
            hawkes,
            dataset,
            semantic_blend=effective_semantic_blend,
        )
        return

    if residual_initialization:
        initialization_mode = "residual_signature"
        initialization_stats = initialize_tree_from_residual_signatures(
            tree,
            hawkes,
            dataset,
            summary_path=args.sequence_summary,
            init_scale=args.residual_init_scale,
            lowrank_rank=args.residual_init_rank,
            grad_clip=args.residual_init_grad_clip,
        )
        print(
            "[Residual Initialization] "
            f"sequences={int(initialization_stats['sequence_count'])} "
            f"rank={initialization_stats['lowrank_rank']} "
            f"grad_norm="
            f"{initialization_stats['gradient_norm_min']:.3e}/"
            f"{initialization_stats['gradient_norm_mean']:.3e}/"
            f"{initialization_stats['gradient_norm_max']:.3e} "
            f"clipped={initialization_stats['gradient_clipped_fraction']:.3f} "
            f"membership={initialization_stats['leaf_membership_mass']} "
            f"target_mass={initialization_stats['target_leaf_mass']}"
        )
        leaf_delta_mean = initialization_stats["leaf_delta_mean_abs"]
        mean_error = initialization_stats["weighted_mean_error"]
    else:
        initialization_mode = (
            "legacy_random"
            if args.leaf_symmetry_scale > 0.0
            else "none"
        )
        initialization_stats = tree.break_initial_leaf_symmetry(
            relative_scale=args.leaf_symmetry_scale,
            seed=args.seed,
        )
        leaf_delta_mean = initialization_stats["mean_abs"]
        mean_error = initialization_stats["mean_error"]

    tree.initialization_metadata = {
        "mode": initialization_mode,
        "semantic_blend": float(effective_semantic_blend),
        "sequence_summary": (
            str(Path(args.sequence_summary).resolve())
            if args.sequence_summary is not None
            else None
        ),
        "residual_init_scale": float(args.residual_init_scale),
        "residual_init_rank": int(args.residual_init_rank),
        "residual_init_grad_clip": float(args.residual_init_grad_clip),
        "legacy_leaf_symmetry_scale": float(args.leaf_symmetry_scale),
        "stats": initialization_stats,
    }

    node_theta = torch.stack(
        [tree.semantic_theta(node_id) for node_id in tree.all_node_ids],
        dim=0,
    )
    semantic_delta = (node_theta - cold_target).detach().abs()
    print(
        "[Initialization] "
        f"mode={initialization_mode} "
        f"semantic_blend={effective_semantic_blend:.4f} "
        f"semantic_delta_mean={float(semantic_delta.mean().cpu()):.4e} "
        f"semantic_delta_max={float(semantic_delta.max().cpu()):.4e} "
        "offset_mode=path_additive "
        f"route_balance={args.route_balance_weight:.4f}"
    )
    stability = _leaf_spectral_radius_summary(tree, hawkes)
    if stability["max"] >= 1.0:
        raise RuntimeError(
            "semantic initialization produced an unstable Hawkes expert "
            f"(max spectral radius={stability['max']:.6f}); reduce the active "
            "semantic/residual initialization scale"
        )
    print(
        "[Symmetry] "
        f"mode={initialization_mode} "
        f"router_gain={args.router_init_gain:.4f} "
        f"router_norm_mean="
        f"{sum(router_norms.values()) / max(len(router_norms), 1):.4e} "
        f"leaf_delta_mean={leaf_delta_mean:.4e} "
        f"leaf_mean_error={mean_error:.3e} "
        f"spectral_radius=[{stability['min']:.6f},"
        f"{stability['max']:.6f}]"
    )
    encoder = CausalPrefixEncoder(
        num_event_types=hawkes.num_types,
        z_dim=tree.z_dim,
    ).to(tree._device_anchor.device)
    if (
        args.alignment_epochs > 0
        and args.allow_root_only_alignment
        and not tree.internal_ids
    ):
        alignment_stats = {
            "epochs": 0,
            "requested_epochs": int(args.alignment_epochs),
            "mode": "skipped_root_only_tree",
            "sequence_count": len(dataset),
            "internal_node_count": 0,
            "leaf_count": len(tree.leaf_ids),
        }
        tree.initialization_metadata["alignment"] = alignment_stats
        print(
            "[H-align Skipped] "
            "mode=skipped_root_only_tree "
            f"requested_epochs={args.alignment_epochs} "
            "reason=no_internal_routing_decision updated=none"
        )
    elif args.alignment_epochs > 0:
        alignment_membership = load_h_tree_leaf_membership(
            args.sequence_summary,
            dataset,
            tree.leaf_ids,
            dtype=hawkes.raw_mu.dtype,
        )
        alignment_stats = run_membership_alignment(
            tree,
            encoder,
            dataset,
            alignment_membership,
            epochs=args.alignment_epochs,
            batch_size=args.alignment_batch_size,
            learning_rate=args.alignment_lr,
            weight_decay=args.alignment_weight_decay,
            temperature=args.alignment_temperature,
            grad_clip=args.alignment_grad_clip,
            seed=args.seed,
        )
        tree.initialization_metadata["alignment"] = alignment_stats
        print(
            "[H-align Complete] "
            f"epochs={alignment_stats['epochs']} "
            f"loss={alignment_stats['initial_loss']:.6f}"
            f"->{alignment_stats['final_loss']:.6f} "
            f"accuracy={alignment_stats['final_weighted_accuracy']:.4f} "
            f"p_target={alignment_stats['final_target_probability']:.4f} "
            "updated=encoder+router_compat "
            "frozen=hawkes+node_emb+semantics+memory"
        )
    else:
        tree.initialization_metadata["alignment"] = {
            "epochs": 0,
            "mode": "disabled",
        }
    trainer = MemoryTreeTrainer(
        tree=tree,
        hawkes=hawkes,
        encoder=encoder,
        wake=WakeObjectiveConfig(
            prototype_duplicate_threshold=args.prototype_duplicate_threshold,
            prototype_mode_threshold=args.prototype_mode_threshold,
            prototype_duplicate_quantile=args.prototype_duplicate_quantile,
            prototype_mode_capacity=args.prototype_mode_capacity,
            prototype_context_alias_capacity=args.prototype_context_alias_capacity,
            lambda_route_mi=args.route_mi_weight,
            lambda_route_posterior=args.route_posterior_weight,
            lambda_route_distill=args.route_distill_weight,
            lambda_route_balance=args.route_balance_weight,
            lambda_route_mix=args.route_mix_weight,
            route_energy_temperature=args.route_energy_temperature,
            route_encoder_warmup_epochs=(
                args.route_encoder_warmup_epochs
            ),
            route_encoder_grad_scale=args.route_encoder_grad_scale,
            route_encoder_reliability_decay=(
                args.route_encoder_reliability_decay
            ),
            route_teacher_temperature=args.route_teacher_temperature,
            lambda_route_probe=args.route_probe_weight,
            route_probe_leaves=args.route_probe_leaves,
            route_probe_leaf_smoothing=args.route_probe_leaf_smoothing,
            route_probe_residual_temperature=(
                args.route_probe_residual_temperature
            ),
            route_probe_gain_temperature=(
                args.route_probe_gain_temperature
            ),
            route_probe_complexity_weight=(
                args.route_probe_complexity_weight
            ),
            route_probe_residual_rank=args.residual_init_rank,
            route_probe_residual_grad_clip=args.residual_init_grad_clip,
            route_balance_batch_size=args.route_balance_batch_size,
            wake_wavefront_batch_size=args.wake_wavefront_batch_size,
            retrieval_microbatch=args.retrieval_microbatch,
            route_balance_max_steps=args.route_balance_max_steps,
            route_balance_target_kl=args.route_balance_target_kl,
            count_similarity_low=args.count_similarity_low,
            count_similarity_high=args.count_similarity_high,
            count_exponent=args.count_exponent,
            count_saturation=args.count_saturation,
            count_topk=args.count_topk,
        ),
        sleep=SleepConfig(
            light_replay_budget=args.light_replay_budget,
            deep_availability_tau=args.deep_availability_tau,
            deep_probe_interval=args.deep_probe_interval,
            deep_computation_cost=args.deep_computation_cost,
            deep_prior_probability=args.deep_prior_probability,
            deep_prior_weight=args.deep_prior_weight,
            deep_evidence_budget=args.deep_evidence_budget,
            topology_inertia_strength=args.topology_inertia_strength,
            topology_inertia_tau=args.topology_inertia_tau,
            split_min_structural_strength=(
                args.split_min_structural_strength
            ),
            split_min_effective_sample_size=(
                args.split_min_effective_sample_size
            ),
            split_route_loss_weight=args.split_route_loss_weight,
            split_anchor_weight=split_anchor_weight,
        ),
        structure=StructureConfig(
            prune_warmup_epochs=args.prune_warmup_epochs,
            merge_kwargs={
                "min_replay": args.merge_min_replay,
                "gate_temperature": args.merge_gate_temperature,
                "loss_weight": args.merge_loss_weight,
                "budget_ratio": args.merge_budget_ratio,
                "dual_lr": args.merge_dual_lr,
                "dual_initial": args.merge_dual_initial,
                "stale_weight": args.merge_stale_weight,
                "dynamics_weight": args.merge_dynamics_weight,
                "normalize_by_events": True,
            },
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            sleep_every=args.sleep_every,
            checkpoint_path=args.checkpoint,
            best_checkpoint_path=args.best_checkpoint,
            validation_history_path=args.validation_history_path,
            controller_diagnostics_path=args.controller_diagnostics_path,
            unified_topology_log_path=args.unified_topology_log_path,
            router_lr_scale=args.router_lr_scale,
            seed=args.seed,
            plot_after_training=not args.no_training_plots,
            training_metrics_path=args.training_metrics_path,
            training_plot_path=args.training_plot_path,
            evaluation_ablation=args.evaluation_ablation,
            controller_target_version=args.controller_target_version,
            controller_train_heads=tuple(
                value.strip()
                for value in args.controller_heads.split(",")
                if value.strip()
            ),
            controller_write_ranking=bool(args.controller_write_ranking),
        ),
        device=constructor.device,
    )
    _apply_cl_metadata(trainer, args)
    _apply_evaluation_ablation(trainer, args.evaluation_ablation)
    if args.initial_checkpoint_output is not None and resume_payload is None:
        trainer.save_checkpoint(args.initial_checkpoint_output, epoch=0)
    trainer.train(dataset, validation_dataset=validation_dataset)
