from __future__ import annotations

import argparse
import json
import logging
import math
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


def _reshape_for_heads(tensor: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """(batch, seq, hidden) -> (batch, num_heads, seq, head_dim)."""
    seq_len, _ = tensor.shape
    return tensor.view(seq_len, num_heads, head_dim)


def weigh_initial_strengths(
    model: AutoModelForImageClassification,
    initial_strengths: torch.Tensor,
    hidden_states: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """
    Scale token-layer strengths using CLS→patch attention.

    Pre-softmax logits are computed per head; softmax is over patch keys only (CLS excluded).
    Head weights are averaged; embedding row uses all ones.
    """
    vit = model.vit
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_tokens = initial_strengths.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    weight_rows: list[torch.Tensor] = []

    for layer_idx, layer in enumerate(vit.layers):
        h = hidden_states[layer_idx]
        h_norm = layer.layernorm_before(h)

        q = _reshape_for_heads(layer.attention.q_proj(h_norm), num_heads, head_dim)
        k = _reshape_for_heads(layer.attention.k_proj(h_norm), num_heads, head_dim)
        
        attn_logits = q.permute(1, 0, 2) @ k.permute(1, 2, 0) * scale
        cls_patch_logits = attn_logits[:, 0, 1:].mean(dim=0)
        patch_weights = torch.softmax(cls_patch_logits, dim=0)
        patch_weights = (patch_weights - patch_weights.min()) / (patch_weights.max() - patch_weights.min())
        weight_rows.append(patch_weights)

    weight_rows.append(
        torch.ones(num_tokens-1, device=initial_strengths.device, dtype=initial_strengths.dtype)
    )

    weights = torch.stack(weight_rows, dim=0)
    cls_column = torch.ones(
        weights.shape[0],
        1,
        device=weights.device,
        dtype=weights.dtype,
    )
    weights = torch.cat([cls_column, weights], dim=1)

    if weights.shape != initial_strengths.shape:
        raise ValueError(
            f'Attention weight shape {tuple(weights.shape)} does not match '
            f'initial_strengths {tuple(initial_strengths.shape)}'
        )
    return initial_strengths * weights


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
    negative_class = cfg["strength"]["negative_class"]
    weight_by_cls_attention = bool(cfg["strength"].get("weight_by_cls_attention", False))
    positive_label = get_label(full_dataset, positive_class)
    negative_label = get_label(full_dataset, negative_class)

    processor = AutoImageProcessor.from_pretrained(cfg["model"]["pretrained"])
    model = AutoModelForImageClassification.from_pretrained(
        cfg["model"]["pretrained"],
        device_map=cfg["model"]["device_map"],
    )
    model.eval()
    device = next(model.parameters()).device

    output_path = Path(cfg["output"]["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for image_number, instance in enumerate(tqdm(dataset, desc="Calculating initial strengths")):
            logger.info("Processing image %s/%s", image_number, len(dataset))

            image_path = instance.get("img")
            label = instance.get("label")

            inputs = processor(images=image_path, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True)

            pred_label = outputs.logits.argmax(-1).item()
            pred_label_name = get_label_name(full_dataset, pred_label)

            hidden_states = torch.stack(outputs.hidden_states).squeeze()
            # hidden_states[0] refers to activations after embeddings (before first encoder block)
            # so take hidden_states[1:] to get activations after first encoder block
            hidden_states = hidden_states[1:, :, :]
         
            interpreted_activations = model.classifier(hidden_states)
            pair_logits = interpreted_activations[..., [positive_label, negative_label]]
            pair_probs = torch.softmax(pair_logits, dim=-1)
            initial_strengths = pair_probs[..., 0]

            if weight_by_cls_attention:
                initial_strengths = weigh_initial_strengths(
                    model,
                    initial_strengths,
                    hidden_states,
                )

            data = {
                "label": label,
                "result": {
                    "query": {
                        "query": str(image_path),
                        "query_probs": initial_strengths.detach().cpu().tolist(),
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
