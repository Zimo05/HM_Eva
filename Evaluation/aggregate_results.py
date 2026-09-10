from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from core.io import write_csv, write_json
from core.metrics import bootstrap_mean_ci, mean, paired_permutation_test


ROOT = Path(__file__).resolve().parent


def completed_results(root: Path) -> list[dict]:
    rows = []
    for manifest_path in root.rglob("manifest.json"):
        run_dir = manifest_path.parent
        status_path, metrics_path = run_dir / "status.json", run_dir / "metrics.json"
        if not status_path.exists() or not metrics_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "complete":
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        resources_path = run_dir / "resources.json"
        resources = json.loads(resources_path.read_text(encoding="utf-8")) if resources_path.exists() else {}
        row = {"result_dir": str(run_dir), "job_key": manifest["job_key"], "dataset": manifest["dataset"], "model": manifest["model"], "condition": manifest["condition"], "seed": manifest["seed"], "variant": manifest.get("variant")}
        for key in ("nll_per_event", "accuracy", "macro_f1", "time_mae", "time_rmse", "num_events"):
            row[key] = metrics.get(key)
        for key in ("wall_seconds", "gpu_hours", "checkpoint_bytes", "peak_gpu_memory_bytes"):
            row[key] = resources.get(key)
        signature_payload = {
            "job": {key: manifest.get(key) for key in ("job_key", "seed", "variant", "rank", "task_start", "task_end")},
            "arguments": {key: value for key, value in manifest.get("arguments", {}).items() if key not in {"output_root", "run_id", "resume", "dry_run"}},
            "inputs": [item.get("sha256") for item in manifest.get("inputs", [])],
            "git_commit": manifest.get("git_commit"),
            "metrics": metrics,
        }
        row["result_hash"] = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()
        rows.append(row)
    return rows


def registry_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_plots(rows: list[dict], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    standard = [row for row in rows if row.get("condition") == "full" and row.get("nll_per_event") is not None]
    if standard:
        fig, ax = plt.subplots(figsize=(8, 5))
        for row in standard:
            x = row.get("checkpoint_bytes") or 0
            ax.scatter(x, row["nll_per_event"], label=f"{row['dataset']}/{row['model']}")
        ax.set_xscale("symlog")
        ax.set_xlabel("checkpoint bytes")
        ax.set_ylabel("NLL/event")
        ax.set_title("Performance-resource Pareto")
        fig.tight_layout(); fig.savefig(figures / "figure1_resource_pareto.png", dpi=180); plt.close(fig)
    continual = [row for row in rows if row.get("dataset") == "continual"]
    if continual:
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = 0
        for result in continual:
            path = Path(result["result_dir"]) / "sequence_metrics.csv"
            if not path.exists() or not path.stat().st_size:
                continue
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                stage_rows = list(csv.DictReader(handle))
            points = []
            for item in stage_rows:
                evaluation = item.get("evaluation", item.get("eval_kind", ""))
                task = item.get("task", item.get("checkpoint_task"))
                eval_task = item.get("eval_task")
                nll = item.get("nll_per_event")
                is_current = evaluation == "post_update" or (
                    evaluation in {"task", "current_task"}
                    and (eval_task in {None, "", task})
                )
                if is_current and task not in {None, ""} and nll not in {None, ""}:
                    points.append((int(task), float(nll)))
            if points:
                points.sort()
                ax.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=f"{result['model']}/{result['condition']}/s{result['seed']}")
                plotted += 1
        ax.set_xlabel("task")
        ax.set_ylabel("post-update NLL/event")
        ax.set_title("Continual-learning timeline")
        if plotted:
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(figures / "figure2_continual_timeline.png", dpi=180); plt.close(fig)
    ranks = [row for row in rows if str(row.get("condition", "")).startswith("residual_rank") and row.get("nll_per_event") is not None]
    if ranks:
        order = {"0": 0, "1": 1, "2": 2, "4": 4, "8": 8, "D": 9}
        ranks.sort(key=lambda row: order.get(row["condition"].split("_")[-1], 99))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(len(ranks)), [row["nll_per_event"] for row in ranks], marker="o")
        ax.set_xticks(range(len(ranks)), [row["condition"].split("_")[-1] for row in ranks])
        ax.set_xlabel("effective rank"); ax.set_ylabel("NLL/event")
        fig.tight_layout(); fig.savefig(figures / "figure3_residual_compression.png", dpi=180); plt.close(fig)


def statistical_tables(rows: list[dict], output: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("nll_per_event") is not None:
            grouped[(row["dataset"], row["model"], row["condition"], row.get("variant"))].append(row)
    summaries = []
    for key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        nll = [float(row["nll_per_event"]) for row in values]
        low, high = bootstrap_mean_ci(nll, seed=2024)
        summaries.append({"dataset": key[0], "model": key[1], "condition": key[2], "variant": key[3], "runs": len(nll), "nll_mean": mean(nll), "nll_ci_low": low, "nll_ci_high": high})
    write_csv(output / "summary_with_bootstrap_ci.csv", summaries)
    tests = []
    standard = [row for row in rows if row.get("condition") == "full" and row.get("nll_per_event") is not None]
    cells = defaultdict(dict)
    for row in standard:
        cells[(row["dataset"], row.get("variant"), row["model"])][int(row["seed"])] = float(row["nll_per_event"])
    datasets = sorted({(row["dataset"], row.get("variant")) for row in standard}, key=repr)
    for dataset, variant in datasets:
        hm = cells.get((dataset, variant, "HM"), {})
        for model in sorted({row["model"] for row in standard if row["dataset"] == dataset and row.get("variant") == variant and row["model"] != "HM"}):
            baseline = cells.get((dataset, variant, model), {})
            seeds = sorted(set(hm) & set(baseline))
            if seeds:
                tests.append({"dataset": dataset, "variant": variant, "comparison": f"HM_vs_{model}", "paired_seeds": len(seeds), "mean_nll_difference": mean([hm[s] - baseline[s] for s in seeds]), "permutation_p": paired_permutation_test([hm[s] for s in seeds], [baseline[s] for s in seeds], seed=2024)})
    write_csv(output / "paired_permutation_tests.csv", tests)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed results without launching experiments")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--registry", type=Path, default=ROOT / "experiment_registry.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "aggregate")
    args = parser.parse_args()
    rows = completed_results(args.results_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identities = defaultdict(list)
    for row in rows:
        identities[(row["job_key"], str(row["seed"]), str(row.get("variant")))].append(row)
    conflicts = []
    deduplicated = []
    for key, values in identities.items():
        by_hash = defaultdict(list)
        for item in values:
            by_hash[item["result_hash"]].append(item)
        deduplicated.append(values[0])
        if len(by_hash) > 1:
            conflicts.append({"identity": repr(key), "hashes": " | ".join(by_hash), "result_dirs": " | ".join(item["result_dir"] for item in values)})
    rows = deduplicated
    expected = registry_rows(args.registry)
    completed_ids = {f"{row['job_key']}:{row['seed']}:{row.get('variant') or ''}" for row in rows}
    missing = [row for row in expected if row.get("identity") not in completed_ids]
    write_csv(args.output_dir / "all_results.csv", rows)
    write_csv(args.output_dir / "missing_jobs.csv", missing)
    write_csv(args.output_dir / "conflicts.csv", conflicts)
    table1 = [row for row in rows if row["dataset"] in {"dws", "retweet", "taobao", "stackoverflow"} and row["condition"] == "full"]
    table2 = [row for row in rows if row["model"] == "HM" and row["condition"] != "full"]
    write_csv(args.output_dir / "table1_prediction.csv", table1)
    write_csv(args.output_dir / "table2_ablation.csv", table2)
    write_json(args.output_dir / "summary.json", {"completed": len(rows), "missing": len(missing), "conflicts": len(conflicts)})
    statistical_tables(rows, args.output_dir)
    make_plots(rows, args.output_dir)
    print(f"Completed={len(rows)} missing={len(missing)} conflicts={len(conflicts)}")


if __name__ == "__main__":
    main()
