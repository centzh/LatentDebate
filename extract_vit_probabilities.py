from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

HERE = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_label(dataset, label_name: str) -> int:
    return dataset.features.get("label").str2int(label_name)


def get_label_name(dataset, label_idx: int) -> str:
    return dataset.features.get("label").int2str(label_idx)


def select_dataset(dataset, include_classes: list[str] | None):
    if not include_classes:
        return dataset
    allowed = {get_label(dataset, name) for name in include_classes}
    labels = np.asarray(dataset["label"])
    indices = np.nonzero(np.isin(labels, list(allowed)))[0]
    return dataset.select(indices)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ViT token-layer probabilities to JSONL.")
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "config" / "cifar10" / "cat_dog.yaml",
        help="YAML configuration file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    hub_id = cfg["dataset"]["hub_id"]
    split = cfg["dataset"]["split"]
    include_classes = cfg["dataset"].get("include_classes")

    full_dataset = load_dataset(hub_id, split=split)
    dataset = select_dataset(full_dataset, include_classes)
    logger.info(
        "Loaded %s split=%s (%s instances after filter)",
        hub_id,
        split,
        len(dataset),
    )

    positive_class = cfg["strength"]["positive_class"]
    positive_label = get_label(full_dataset, positive_class)

    processor = AutoImageProcessor.from_pretrained(cfg["model"]["pretrained"])
    model = AutoModelForImageClassification.from_pretrained(
        cfg["model"]["pretrained"],
        device_map=cfg["model"]["device_map"],
    )
    model.eval()

    output_path = Path(cfg["output"]["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for image_number, instance in enumerate(tqdm(dataset, desc="Calculating initial strengths")):
            logger.info("Processing image %s/%s", image_number, len(dataset))

            image_path = instance.get("img")
            label = instance.get("label")

            inputs = processor(images=image_path, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)

            pred_label = outputs.logits.argmax(-1).item()
            pred_label_name = get_label_name(full_dataset, pred_label)

            hidden_states = [layer.detach().numpy() for layer in outputs.hidden_states]
            hidden_states = torch.from_numpy(np.squeeze(np.asarray(hidden_states)))

            interpreted_activations = model.classifier(hidden_states)
            interpreted_softmaxes = torch.softmax(interpreted_activations, dim=-1)
            initial_strengths = interpreted_softmaxes[:, :, positive_label]

            data = {
                "label": label,
                "result": {
                    "query": {
                        "query": str(image_path),
                        "query_probs": initial_strengths.tolist(),
                    },
                    "answer": {
                        "answer": pred_label_name,
                    },
                },
            }
            f.write(json.dumps(data) + "\n")

    logger.info("Completed writing initial strengths to %s", output_path.resolve())


if __name__ == "__main__":
    main()
