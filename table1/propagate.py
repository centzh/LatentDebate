"""Token-layer strength propagation used by Table 1 predictions and plots."""
from __future__ import annotations

import numpy as np


def propagate(initial, quadratic=False, cls_patch_only=False, cls_only=False, weighted=False):
    """Original topology or within-layer quadratic topology, probability semantics."""
    num_records, tokens, layers = initial.shape
    final = np.zeros_like(initial)

    for layer in range(0, layers):
        for token in range(tokens - 2, -1, -1):
            energy = np.zeros(num_records, dtype=np.float64)

            if cls_only:
                if token == 0:
                    energy += final[:, token, layer - 1]
                    influence = 1 - np.tanh(np.abs(energy))
                    final[:, token, layer] = np.tanh(energy) + influence * initial[:, token, layer]

            elif weighted:
                near_zero = np.abs(initial[:, token, layer]) < 0.01
                energy += final[:, :, layer].sum(axis=1)
                influence = 1 - np.tanh(np.abs(energy))
                propagated = np.tanh(energy) + influence * initial[:, token, layer]
                final[:, token, layer] = np.where(
                    near_zero,
                    initial[:, token, layer],
                    propagated,
                )

            else:
                if quadratic:
                    # Aggregate from all preceding tokens on the same layer.
                    energy += final[:, token + 1:, layer].sum(axis=1)

                elif cls_patch_only:
                    if token == 0:
                        energy += final[:, token + 1:, layer].sum(axis=1)

                else:
                    # Linear topology: previous token on the same layer.
                    energy += final[:, token + 1, layer]

                # Last-position tokens also receive the last-position node from the layer below.
                if token == 0 and layer:
                    energy += final[:, token, layer - 1]

                influence = 1 - np.tanh(np.abs(energy))
                final[:, token, layer] = np.tanh(energy) + influence * initial[:, token, layer]

    return final
