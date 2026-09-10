from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-trials", type=int, default=8)
    parser.add_argument("--max-gpu-hours", type=float, default=24.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--python", dest="python_executable", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_runtime(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """Fail before creating a result directory when the runtime cannot run a job."""
    if sys.version_info < (3, 10):
        parser.error(
            "Evaluation 需要 Python 3.10+；当前入口由 "
            f"{sys.executable} (Python {sys.version.split()[0]}) 启动。"
            "请先激活正确的 conda/venv 环境，再重新运行命令。"
        )

    executable = args.python_executable or sys.executable
    probe = (
        "import importlib.util,json,sys; "
        "mods={n:importlib.util.find_spec(n) is not None for n in "
        "('numpy','pandas','torch')}; "
        "cuda=False; count=0; "
        "exec(\"if mods['torch']:\\n import torch\\n cuda=torch.cuda.is_available()\\n count=torch.cuda.device_count()\"); "
        "print(json.dumps({'version':list(sys.version_info[:3]),"
        "'modules':mods,'cuda':cuda,'cuda_device_count':count}))"
    )
    try:
        completed = subprocess.run(
            [executable, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        runtime = json.loads(completed.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as error:
        parser.error(f"无法检查实验 Python 环境 {executable!r}：{error}")

    if tuple(runtime["version"]) < (3, 10):
        version = ".".join(str(value) for value in runtime["version"])
        parser.error(f"底层实验环境 {executable!r} 是 Python {version}，需要 Python 3.10+。")
    missing = [name for name, present in runtime["modules"].items() if not present]
    if missing:
        parser.error(f"实验环境 {executable!r} 缺少依赖：{', '.join(missing)}。")
    if args.device.startswith("cuda"):
        if not runtime["cuda"]:
            parser.error(
                f"指定了 --device {args.device}，但 {executable!r} 中的 PyTorch 不支持可用 CUDA。"
                "可先用 --device cpu 做 smoke test，正式 GPU 实验需安装 CUDA 版 PyTorch。"
            )
        if ":" in args.device:
            try:
                index = int(args.device.split(":", 1)[1])
            except ValueError:
                parser.error("--device 应为 auto、cpu、cuda 或 cuda:N。")
            if index < 0 or index >= runtime["cuda_device_count"]:
                parser.error(
                    f"请求 GPU {index}，但实验环境只检测到 {runtime['cuda_device_count']} 张 GPU。"
                )
    elif args.device not in {"auto", "cpu"}:
        parser.error("--device 应为 auto、cpu、cuda 或 cuda:N。")
    return args


def _parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    return _validate_runtime(parser, parser.parse_args())


def stationary_args(*, dws: bool = False) -> argparse.Namespace:
    parser = common_parser("Run one stationary model/dataset evaluation cell")
    if dws:
        parser.add_argument("--variant", choices=("8", "13", "20"), default="13")
    return _parse(parser)


def continual_args(*, replay: bool = False) -> argparse.Namespace:
    parser = common_parser("Run one continual-learning evaluation cell")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=9)
    parser.add_argument("--data-root", type=Path, default=None)
    if replay:
        parser.add_argument("--hm-resource-root", type=Path, required=True)
    args = _parse(parser)
    if not 0 <= args.task_start <= args.task_end <= 9:
        parser.error("task range must satisfy 0 <= start <= end <= 9")
    return args


def diagnostic_args(kind: str) -> argparse.Namespace:
    parser = common_parser(f"Run HawkesMemory diagnostic: {kind}")
    if kind in {"law_recovery", "frontier", "residual_rank"}:
        parser.add_argument("--variant", choices=("8", "13", "20"), default="13")
    if kind == "residual_rank":
        parser.add_argument("--rank", required=True, choices=("0", "1", "2", "4", "8", "D"))
    if kind in {"timescales", "consolidation", "tree_growth"}:
        parser.add_argument("--task-start", type=int, default=0)
        parser.add_argument("--task-end", type=int, default=9)
        parser.add_argument("--data-root", type=Path, default=None)
    return _parse(parser)
