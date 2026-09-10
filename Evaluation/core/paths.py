from __future__ import annotations

from pathlib import Path


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVALUATION_ROOT.parent
MODELS_ROOT = PROJECT_ROOT / "Models"
DATASETS_ROOT = PROJECT_ROOT / "Datasets"


def result_dir(spec, args) -> Path:
    root = (args.output_root or EVALUATION_ROOT / "results").expanduser().resolve()
    if spec.kind == "continual":
        pieces = ["continual", spec.model, spec.condition, f"seed_{args.seed}"]
    else:
        pieces = [spec.dataset, spec.model, spec.condition]
        if spec.dataset == "dws":
            pieces.append(f"variant_{args.variant}")
        if getattr(args, "rank", None) is not None:
            pieces.append(f"rank_{args.rank}")
        pieces.append(f"seed_{args.seed}")
    if args.run_id:
        pieces.append(args.run_id)
    return root.joinpath(*pieces)
