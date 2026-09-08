import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer
from wandb import summary

torch.set_num_threads(os.cpu_count())

def get_activation(model, prompt_text, response_text, layer):
    full_text = prompt_text + response_text
    tokens = model.to_tokens(full_text)
    prompt_tokens = model.to_tokens(prompt_text)
    response_start = prompt_tokens.shape[1]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name == f"blocks.{layer}.hook_resid_post"
        )

    resid = cache[f"blocks.{layer}.hook_resid_post"][0]
    response_resid = resid[response_start:]
    return response_resid.mean(dim=0)

def build_features_single(model, dataset, layer):
    # Single layer
    # Use for pre-test
    X, y = [], []
    for i, entry in enumerate(dataset):
        print("Processing entry:", i)
        vec = get_activation(model, entry["prompt"], entry["response"], layer)
        X.append(vec.float().numpy())
        y.append(1 if entry["label"] == "deceptive" else 0)
        print("Processed entry:", i)
    return np.array(X), np.array(y)

def build_features_multilayer(model, dataset, layers):
    features = {layer: [] for layer in layers}
    y = []
    for i, entry in enumerate(dataset):
        print(f"Processing entry {i+1}/{len(dataset)}")
        full_text = entry["prompt"] + entry["response"]
        tokens = model.to_tokens(full_text)
        prompt_tokens = model.to_tokens(entry["prompt"])
        response_start = prompt_tokens.shape[1]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: any(f"blocks.{l}.hook_resid_post" == name for l in layers)
            )

        for layer in layers:
            resid = cache[f"blocks.{layer}.hook_resid_post"][0]
            response_resid = resid[response_start:]
            features[layer].append(response_resid.mean(dim=0).float().numpy())

        y.append(1 if entry["label"] == "deceptive" else 0)
        if i % 10 == 0:
            print(f"[{i+1}/{len(dataset)}] extracted")

        print(f"Processed entry {i+1}/{len(dataset)}")

    y = np.array(y)
    return {layer: np.array(features[layer]) for layer in layers}, y

def build_features_batched(model, dataset, layers, batch_size=8):
    features = {layer: [] for layer in layers}
    y = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(dataset) - 1)//batch_size + 1}")
        full_texts = [e["prompt"] + e["response"] for e in batch]
        tokens = model.to_tokens(full_texts, padding_side="right")  # pads to longest in batch

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: any(f"blocks.{l}.hook_resid_post" == name for l in layers)
            )

        for j, entry in enumerate(batch):
            print(f"Processing entry {i+j+1}/{len(dataset)}")
            prompt_len = model.to_tokens(entry["prompt"]).shape[1]
            for layer in layers:
                resid = cache[f"blocks.{layer}.hook_resid_post"][j]
                features[layer].append(resid[prompt_len:].mean(dim=0).float().numpy())
            y.append(1 if entry["label"] == "deceptive" else 0)
            print(f"Processed entry {i+j+1}/{len(dataset)}")

        print(f"[{i+len(batch)}/{len(dataset)}] done")

    return {l: np.array(v) for l, v in features.items()}, np.array(y)

def train_and_eval_probe(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_scores)
    return clf, auroc

def learning_curve(X, y, sizes, n_repeats=5):
    results = {}
    for n in sizes:
        aurocs = []
        for seed in range(n_repeats):
            idx = np.random.RandomState(seed).choice(len(y), size=min(n, len(y)), replace=False)
            X_sub, y_sub = X[idx], y[idx]
            try:
                _, auroc = train_and_eval_probe(X_sub, y_sub, test_size=0.3)
                aurocs.append(auroc)
            except ValueError:
                continue  # skip if a class is missing in a split
        results[n] = aurocs
        print(f"n={n}: mean AUROC={np.mean(aurocs):.3f}, std={np.std(aurocs):.3f}")
    return results

def plot_learning_curve(results):

    sizes = sorted(results.keys())
    means = [np.mean(results[n]) for n in sizes]
    stds = [np.std(results[n]) for n in sizes]

    plt.figure(figsize=(7, 5))
    plt.errorbar(sizes, means, yerr=stds, marker='o', capsize=4)
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance (AUROC=0.5)')
    plt.xlabel("Number of training examples")
    plt.ylabel("AUROC")
    plt.title("Probe AUROC vs. dataset size")
    plt.ylim(0.4, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/learning_curve.png", dpi=150)

def small_test(model):

    # Load a small dataset
    with open("data/instructed_generated.json") as f:
        pilot_instructed_results_as_list = json.load(f)
    
    X, y = build_features_single(model, pilot_instructed_results_as_list, layer=model.cfg.n_layers // 4)
    if len(X) > 0 and len(y) > 0:
        print("Building complete.")
    else:
        print("Wrong data, please check")
        return
        
    # Save features
    np.savez("data/features_layer{}.npz".format(model.cfg.n_layers // 4), X=X, y=y)
    print(f"Saved {X.shape[0]} examples, dim={X.shape[1]}")
    
    # Estimate converging size
    loaded = np.load("data/features_layer{}.npz".format(model.cfg.n_layers // 4))
    X_load, y_load = loaded["X"], loaded["y"]
    sizes = [20, 40, 60, 80, 100, 120, 150]
    results = learning_curve(X_load, y_load, sizes)
    plot_learning_curve(results)
    
    # Calculate the actual AUROC
    #clf, auroc = train_and_eval_probe(X, y, test_size=0.3)
    #print(f"AUROC: {auroc:.3f}")

def production_test(model):

    n_layers = model.cfg.n_layers
    candidate_layers = list(range(2, n_layers, 3))

    
    # Part 1: Load the full dataset
    with open("data/probe_train.json") as f:
        train_data = json.load(f)
    with open("data/probe_eval.json") as f:
        eval_data = json.load(f)

    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    X_train_by_layer, y_train = build_features_multilayer(model, train_data, candidate_layers)
    X_eval_by_layer, y_eval = build_features_multilayer(model, eval_data, candidate_layers)

    np.savez("data/features_train.npz", y=y_train,
            **{f"X_layer{l}": X_train_by_layer[l] for l in candidate_layers})
    np.savez("data/features_eval.npz", y=y_eval,
            **{f"X_layer{l}": X_eval_by_layer[l] for l in candidate_layers})
    

    # Part 2: Load features and train probes
    loaded_train = np.load("data/features_train.npz")
    loaded_eval = np.load("data/features_eval.npz")

    y_train = loaded_train["y"]
    y_eval = loaded_eval["y"]

    X_train_by_layer = {layer: loaded_train[f"X_layer{layer}"] for layer in candidate_layers}
    X_eval_by_layer = {layer: loaded_eval[f"X_layer{layer}"] for layer in candidate_layers}

    results_per_layer = {}

    for layer in candidate_layers:
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train_by_layer[layer], y_train)

        y_scores = clf.predict_proba(X_eval_by_layer[layer])[:, 1]
        auroc = roc_auc_score(y_eval, y_scores)
        results_per_layer[layer] = auroc
        print(f"Layer {layer}: AUROC = {auroc:.3f}") 

    summary = {
        "n_layers_total": n_layers,
        "train_size": len(y_train),
        "eval_size": len(y_eval),
        "train_deceptive": int(sum(y_train)),
        "eval_deceptive": int(sum(y_eval)),
        "results_per_layer": {str(layer): auroc for layer, auroc in results_per_layer.items()}
    }

    with open("data/baseline_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)   

def main():

    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    model = HookedTransformer.from_pretrained_no_processing(model_name, device="cpu", dtype=torch.bfloat16)

    #small_test(model)
    production_test(model)


if __name__ == "__main__":
    sys.exit(main())

