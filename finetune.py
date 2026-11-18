"""Fine-tune a ViT classifier on clane9/imagenet-100.

Example:

    uv run finetune.py --output-dir checkpoints/vit-imagenet100
    uv run finetune.py --epochs 1 --batch-size 8 --max-train-samples 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

HERE = Path(__file__).resolve().parent
DEFAULT_HUB = 'clane9/imagenet-100'
DEFAULT_PRETRAINED = 'google/vit-base-patch16-224'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hub-id', default=DEFAULT_HUB)
    parser.add_argument('--pretrained', default=DEFAULT_PRETRAINED)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=HERE / 'checkpoints' / 'vit-imagenet100',
    )
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--eval-split', default='validation')
    parser.add_argument('--epochs', type=float, default=5.0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument(
        '--max-train-samples',
        type=int,
        default=None,
        help='Optional cap for quick smoke runs',
    )
    parser.add_argument(
        '--max-eval-samples',
        type=int,
        default=None,
        help='Optional cap for quick smoke runs',
    )
    return parser.parse_args()


def example_image(example: dict):
    return example.get('image') or example.get('img')


def build_label_maps(label_feature) -> tuple[dict[int, str], dict[str, int]]:
    names = label_feature.names
    id2label = {index: name for index, name in enumerate(names)}
    label2id = {name: index for index, name in enumerate(names)}
    return id2label, label2id


def preprocess_factory(processor: ViTImageProcessor):
    def preprocess(example: dict) -> dict:
        image = example_image(example).convert('RGB')
        pixel_values = processor(images=image, return_tensors='pt')['pixel_values'][0]
        return {'pixel_values': pixel_values, 'labels': example['label']}

    return preprocess


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    return {'accuracy': accuracy}


def maybe_subset(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def load_split(raw_datasets, split: str):
    if split not in raw_datasets:
        return None
    return raw_datasets[split]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    raw_datasets = load_dataset(args.hub_id)
    train_raw = load_split(raw_datasets, args.train_split)
    if train_raw is None:
        raise ValueError(
            f'Split {args.train_split!r} not in {args.hub_id}; '
            f'available: {list(raw_datasets.keys())}',
        )

    id2label, label2id = build_label_maps(train_raw.features['label'])
    num_labels = len(id2label)

    processor = ViTImageProcessor.from_pretrained(args.pretrained)
    model = ViTForImageClassification.from_pretrained(
        args.pretrained,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    preprocess = preprocess_factory(processor)
    train_ds = train_raw.map(
        preprocess,
        remove_columns=train_raw.column_names,
        desc='Preprocess train',
    )
    train_ds = maybe_subset(train_ds, args.max_train_samples)

    eval_raw = load_split(raw_datasets, args.eval_split)
    eval_ds = None
    if eval_raw is not None:
        eval_ds = eval_raw.map(
            preprocess,
            remove_columns=eval_raw.column_names,
            desc='Preprocess eval',
        )
        eval_ds = maybe_subset(eval_ds, args.max_eval_samples)

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy='epoch' if eval_ds is not None else 'no',
        save_strategy='epoch' if eval_ds is not None else 'steps',
        save_steps=500,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        logging_steps=50,
        report_to='none',
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        fp16=use_cuda,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(args.output_dir)

    if eval_ds is not None:
        metrics = trainer.evaluate()
        print('Eval:', metrics, flush=True)

    print(f'Saved model and processor to {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()