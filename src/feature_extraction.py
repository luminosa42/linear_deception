import numpy as np
import torch


def build_features_multilayer(model, dataset, layers):
    features = {layer: [] for layer in layers}
    labels = []

    for index, entry in enumerate(dataset):
        full_text = entry["prompt"] + entry["response"]
        tokens = model.to_tokens(full_text)
        prompt_length = model.to_tokens(entry["prompt"]).shape[1]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: any(
                    f"blocks.{layer}.hook_resid_post" == name for layer in layers
                ),
            )

        for layer in layers:
            residual = cache[f"blocks.{layer}.hook_resid_post"][0]
            response_residual = residual[prompt_length:]
            features[layer].append(
                response_residual.mean(dim=0).detach().float().cpu().numpy()
            )

        labels.append(1 if entry["label"] == "deceptive" else 0)
        print(f"Processed entry {index + 1}/{len(dataset)}")

    return {layer: np.array(values) for layer, values in features.items()}, np.array(labels)