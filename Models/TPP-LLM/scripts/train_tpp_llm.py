"""
Train a TPP-LLM Model
"""
import argparse
import gzip
import json
import os.path
import sys
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig

from tpp_llm.data import TPPLLMDataset, collate_fn
from tpp_llm.model import TPPLLMModel
from tpp_llm.runner import TPPLLMRunner
from tpp_llm.utils import get_prompt

if __name__ == '__main__':
    # Set teh argument parser
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train and test the TPP-LLM model with event sequences.')
    parser.add_argument(
        '--model_path', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='llm path')
    parser.add_argument('--data_path', type=str, default=None, help='prepared dataset path')
    parser.add_argument('--evaluation_dataset', choices=['dws', 'retweet', 'taobao', 'stackoverflow'], default=None)
    parser.add_argument('--evaluation_variant', choices=['8', '13', '20'], default='13')
    parser.add_argument('--evaluation_output', type=Path, default=None)
    parser.add_argument('--anonymous_labels', action='store_true')
    parser.add_argument('--prepared_data', action='store_true')
    parser.add_argument('--initial_checkpoint', type=Path, default=None)
    parser.add_argument(
        '--num_event_types', type=int, default=None, help='number of event types')
    parser.add_argument(
        '--temporal_emb_type', type=str, default='positional', choices=['linear', 'positional', 'shifted'],
        help='temporal embedding type')
    parser.add_argument(
        '--temporal_emb_first', action='store_true', help='temporal embedding first or not')
    parser.add_argument(
        '--no_prompt', action='store_true', help='no prompt')
    parser.add_argument(
        '--num_integral_samples', type=int, default=20, help='number of samples during one integral step')
    parser.add_argument(
        '--quant_type', type=str, default=None, choices=['4bit', '8bit', None], help='quantization type')
    parser.add_argument(
        '--peft_type', type=str, default=None, choices=['lora', None], help='peft type')
    parser.add_argument(
        '--lora_rank', type=int, default=16, help='lora rank')
    parser.add_argument(
        '--lora_modules', type=str, nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        help='lora target modules')
    parser.add_argument(
        '--train_batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument(
        '--eval_batch_size', type=int, default=16, help='batch size for evaluation')
    parser.add_argument(
        '--learning_rate', type=float, default=5e-4, help='larning rate')
    parser.add_argument(
        '--lr_scheduler_type', type=str, default='constant', help='learning rate scheduler type')
    parser.add_argument(
        '--num_train_epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument(
        '--warmup_ratio', type=float, default=0, help='warmup ratio')
    parser.add_argument(
        '--beta_type', type=float, default=1, help='loss coefficient of the event type prediction')
    parser.add_argument(
        '--beta_time', type=float, default=1, help='loss coefficient of the event time prediction')
    parser.add_argument(
        '--device', type=str, default='cpu', help='cpu or cuda device')
    parser.add_argument(
        '--seed', type=int, default=2024, help='seed for reproducibility')

    # Parse arguments
    args = parser.parse_args()
    if args.evaluation_dataset and not args.prepared_data:
        model_root = Path(__file__).resolve().parents[1]
        project_root = model_root.parents[1]
        sys.path.insert(0, str(model_root))
        from data_configuration import DataConfiguration
        target = Path(args.data_path or model_root / 'data_adapted' / args.evaluation_dataset)
        adapter = DataConfiguration(dataset_root=project_root / 'Datasets', output_root=target.parent, seed=args.seed)
        labels = None
        if args.anonymous_labels:
            # The adapter falls back to deterministic event_N names when no
            # semantic label mapping is supplied.
            labels = None
        if args.evaluation_dataset == 'dws':
            adapter.dws(
                output_dir=target.parent,
                variants=[args.evaluation_variant],
                type_labels=labels,
                anonymous_labels=args.anonymous_labels,
            )
            target = target.parent / f'dws_{args.evaluation_variant}'
        else:
            getattr(adapter, args.evaluation_dataset)(
                output_dir=target,
                type_labels=labels,
                anonymous_labels=args.anonymous_labels,
            )
        args.data_path = str(target)
    if args.data_path is None:
        parser.error('--data_path is required unless --evaluation_dataset is used')
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_payload = json.loads((Path(args.data_path) / 'train.json').read_text(encoding='utf-8'))
    if not train_payload:
        parser.error('training data is empty')
    if args.num_event_types is None:
        args.num_event_types = max(int(v) for row in train_payload for v in row['type_event']) + 1
    print(f'args: {args}')
    transformers.set_seed(args.seed)
    base_dataset_name = os.path.basename(args.data_path).replace('_few_shot', '')
    prompt = get_prompt(
        dataset_name=base_dataset_name,
        event_time_first=args.temporal_emb_first,
        anonymous_labels=args.anonymous_labels,
    )
    if args.no_prompt:
        prompt = ''
    print(f'prompt: {prompt}')

    # Get the quantization config
    if args.quant_type == '4bit':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.quant_type == '8bit':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=False,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    # Get the PEFT config
    if args.peft_type == 'lora':
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=16,
            target_modules=args.lora_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
    else:
        peft_config = None

    # Load the model
    model = TPPLLMModel(
        model_name=args.model_path,
        num_event_types=args.num_event_types,
        num_integral_samples=args.num_integral_samples,
        temporal_emb_type=args.temporal_emb_type,
        temporal_emb_first=args.temporal_emb_first,
        prompt=prompt,
        bnb_config=bnb_config,
        peft_config=peft_config,
        device=args.device,
    )
    print(f'model: {model}')

    # Load the dataset
    dataset_train = TPPLLMDataset(f'{args.data_path}/train.json')
    dataset_val = TPPLLMDataset(f'{args.data_path}/dev.json')
    dataset_test = TPPLLMDataset(f'{args.data_path}/test.json')
    dataloader_train = DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    # Train and test the model
    runner = TPPLLMRunner(
        model=model,
        beta_type=args.beta_type,
        beta_time=args.beta_time,
        device=args.device,
    )
    if args.initial_checkpoint is not None:
        if not args.initial_checkpoint.is_file():
            raise FileNotFoundError(args.initial_checkpoint)
        runner.load(str(args.initial_checkpoint))
    output_dir = args.evaluation_output or Path('evaluation_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / 'best.pt'
    result = runner.run(
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        dataloader_test=dataloader_test,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        checkpoint_path=str(checkpoint),
    )
    event_rows = runner.event_predictions(dataloader_test)
    true = [row['true_type'] for row in event_rows]
    pred = [row['predicted_type'] for row in event_rows]
    labels = sorted(set(true) | set(pred))
    f1 = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(true, pred))
        fp = sum(a != label and b == label for a, b in zip(true, pred))
        fn = sum(a == label and b != label for a, b in zip(true, pred))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    errors = [row['predicted_delta_time'] - row['true_delta_time'] for row in event_rows]
    metrics = {
        'nll_per_event': sum(row['event_nll'] for row in event_rows) / len(event_rows),
        'accuracy': sum(a == b for a, b in zip(true, pred)) / len(true),
        'macro_f1': sum(f1) / len(f1),
        'time_mae': sum(abs(value) for value in errors) / len(errors),
        'time_rmse': (sum(value * value for value in errors) / len(errors)) ** 0.5,
        'num_events': len(event_rows),
        'best_epoch': result['best_epoch'],
        'validation': result['validation'],
    }
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    with gzip.open(output_dir / 'predictions.jsonl.gz', 'wt', encoding='utf-8') as handle:
        for row in event_rows:
            handle.write(json.dumps(row) + '\n')
