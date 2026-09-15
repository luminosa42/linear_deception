# Plotting routines for comparison with the paper
# role-playing scenarios and instructed pairs only.

import sys
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from transformer_lens import HookedTransformer

def recall_at_fpr(clf, X_eval, y_eval, target_fpr=0.01):
    y_scores = clf.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_eval, y_scores)
    idx = np.argmin(np.abs(fpr - target_fpr))

    # find the smallest fpr that is >= target_fpr (first threshold that meets your constraint)
    valid = np.where(fpr >= target_fpr)[0]
    if len(valid) == 0:
        idx = len(fpr) - 1  # fallback: highest fpr available
        print(f"No FPR >= {target_fpr:.3f} found; using highest available FPR={fpr[idx]:.3f}")
    else:
        idx = valid[0]

    return tpr[idx], fpr[idx]

# ROC curve + recall-at-FPR
def plot_roc_with_recall_at_fpr(clf, X_eval, y_eval, target_fpr=0.01, layer=None, smooth=True):

    recall_at_target, actual_fpr = recall_at_fpr(clf, X_eval, y_eval, target_fpr)

    fig, ax = plt.subplots(figsize=(6, 6))

    y_scores = clf.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_eval, y_scores)
    auroc = roc_auc_score(y_eval, y_scores)
    fpr_pct = fpr * 100
    tpr_pct = tpr * 100

    if smooth: # Overlay
        # Deduplicate fpr for interpolation (roc_curve can have repeated fpr values)
        fpr_unique, idx = np.unique(fpr_pct, return_index=True)
        tpr_unique = tpr_pct[idx]

        interp_fn = interp1d(fpr_unique, tpr_unique, kind='linear')
        fpr_smooth = np.linspace(fpr_unique.min(), fpr_unique.max(), 100)
        tpr_smooth = interp_fn(fpr_smooth)

        ax.plot(fpr_smooth, tpr_smooth, linewidth=2, label=f"Interpolated ROC (AUROC={auroc:.3f})")
    else:
        ax.plot(fpr_pct, tpr_pct, linewidth=2, label=f"ROC (AUROC={auroc:.3f})")

    ax.scatter([actual_fpr], [recall_at_target], color="red", zorder=5,
                label=f"Recall={recall_at_target:.2f} @ FPR≈{actual_fpr:.3f}")
    ax.plot([0, 100], [0, 100], linestyle="--", color="lightgray", linewidth=1, label="Chance")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 110)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("FPR on Honest")
    ax.set_ylabel("Recall (TPR)")
    title = f"ROC Curve — Layer {layer}" if layer is not None else "ROC Curve"
    ax.axvline(1, color="red", linestyle=":", linewidth=1, alpha=0.7, label="1% FPR")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/roc_curve_layer{layer}.png", dpi=150)
    
    print(f"AUROC: {auroc:.3f}")
    print(f"At {actual_fpr:.3f} FPR (target {target_fpr}), recall = {recall_at_target:.3f}")
    return auroc, recall_at_target

# Token-level probe score trajectory
def get_token_level_scores(model, prompt_text, response_text, clf, layer):

    """
    Returns the probe's per-token deception score across the response,
    rather than a single mean-pooled score.
    """

    full_text = prompt_text + response_text
    tokens = model.to_tokens(full_text)
    prompt_tokens = model.to_tokens(prompt_text)
    response_start = prompt_tokens.shape[1]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name == f"blocks.{layer}.hook_resid_post"
        )

    resid = cache[f"blocks.{layer}.hook_resid_post"][0]  # [seq_len, d_model]
    response_resid = resid[response_start:].float().numpy()  # [n_response_tokens, d_model]

    # Score every token individually using the trained probe
    token_scores = clf.predict_proba(response_resid)[:, 1]  # P(deceptive) per token

    response_token_ids = tokens[0, response_start:]
    response_token_strs = [model.to_string(t.unsqueeze(0)) for t in response_token_ids]

    return response_token_strs, token_scores

def plot_token_trajectory(model, eval_data_path, clf, layer):

    with open(eval_data_path) as f:
        eval_data = json.load(f)

    # For demonstration, just take the first example
    example = eval_data[0]
    label = label=f"{example['label']}_example0"

    print(f"Obtaining token-level scores for {label} (layer {layer})...")
    tokens_str, scores = get_token_level_scores(model, example["prompt"],example["response"] , clf, layer)

    print("Plotting token-level trajectory...")

    plt.figure(figsize=(10, 3))
    plt.plot(scores, marker='o', markersize=3)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Token position in response")
    plt.ylabel("P(deceptive)")
    plt.title(f"Token-level probe score — {label}")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"plots/token_trajectory_{label.replace(' ', '_')}.png", dpi=150)

    # print tokens with scores for a quick manual read
    for tok, score in zip(tokens_str, scores):
        print(f"{score:.2f}  {tok!r}")

def get_training_data(model, layer):

    train_loaded = np.load("data/features_train.npz")
    eval_loaded = np.load("data/features_eval.npz")

    y_train = train_loaded["y"]
    y_eval = eval_loaded["y"]

    print(train_loaded.files)  # confirm which X_layer{N} keys are actually saved

    # reconstruct candidate_layers from whatever's actually in the file
    candidate_layers = [int(k.replace("X_layer", "")) for k in train_loaded.files if k.startswith("X_layer")]
    print("Available layers:", candidate_layers)

    X_train_by_layer = {l: train_loaded[f"X_layer{l}"] for l in candidate_layers}
    X_eval_by_layer = {l: eval_loaded[f"X_layer{l}"] for l in candidate_layers}

    return [X_train_by_layer, y_train, X_eval_by_layer, y_eval]

def plot_roc_and_token_trajectory(model, data, layer):

    X_train_by_layer, y_train, X_eval_by_layer, y_eval = data

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_by_layer[layer], y_train)

    auroc, recall = plot_roc_with_recall_at_fpr(
        clf, X_eval_by_layer[layer], y_eval, layer=layer
    )
    print("Plotting ROC curve done. Now plotting token-level trajectory for the first example in the eval set...")

    plot_token_trajectory(model, eval_data_path="data/probe_eval.json", clf=clf, layer=layer)


def sweep_recall_by_layer(X_train_by_layer, y_train, X_eval_by_layer, y_eval,
                           layers, target_fpr=0.01):
    results = {}
    for layer in layers:
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train_by_layer[layer], y_train)
        recall, actual_fpr = recall_at_fpr(clf, X_eval_by_layer[layer], y_eval, target_fpr)
        results[layer] = {"recall": recall, "actual_fpr": actual_fpr}
        print(f"Layer {layer}: recall={recall:.3f} at FPR≈{actual_fpr:.3f}")
    return results

def plot_auroc_and_recall(training_data_path, data, target_fpr=0.01):

    with open(training_data_path) as f:
        training_data = json.load(f)

    results_per_layer = training_data["results_per_layer"]  # {"7": 0.7x, "14": 0.8x, "21": 0.9x}

    layers = sorted(int(l) for l in results_per_layer.keys())
    aurocs = [results_per_layer[str(l)] for l in layers]

    X_train_by_layer, y_train, X_eval_by_layer, y_eval = data
    layers = sorted(X_train_by_layer.keys())
    recall_results = sweep_recall_by_layer(X_train_by_layer, y_train, X_eval_by_layer, y_eval,
                                        layers, target_fpr=target_fpr)
    recalls_pct = [recall_results[l]["recall"] * 100 for l in layers]

    returned_fpr = recall_results[layers[0]]["actual_fpr"] 

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.1}
    )

    # Top panel — AUROC
    ax1.plot(layers, aurocs, marker='o', linewidth=2, markersize=8, color="#2b6cb0")
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(0.45, 1.05)
    ax1.grid(alpha=0.3)
    ax1.set_title("AUROC and Recall by Layer")

    # Bottom panel — Recall at fixed FPR
    ax2.plot(layers, recalls_pct, marker='s', linewidth=2, markersize=8, color="#c0392b")
    ax2.set_ylabel(f"Recall @ ~{returned_fpr*100:.0f}% FPR")
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.grid(alpha=0.3)

    # Shared x-axis
    ax2.set_xlabel("Probe layer")
    ax2.set_xticks(layers)

    plt.tight_layout()
    plt.savefig("plots/auroc_recall_stacked.png", dpi=150)

def sweep_regularization(X_train, y_train, X_eval, y_eval, C_values):
    results = {}
    for C in C_values:
        clf = LogisticRegression(max_iter=1000, C=C)
        clf.fit(X_train, y_train)
        y_scores = clf.predict_proba(X_eval)[:, 1]
        auroc = roc_auc_score(y_eval, y_scores)
        results[C] = auroc
        print(f"C={C:.4f} (λ≈{1/C:.2f}): AUROC={auroc:.3f}")
    return results

def plot_regularization_sweep(data, layer=1):

    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    X_train_by_layer, y_train, X_eval_by_layer, y_eval = data
    reg_results = sweep_regularization(
        X_train_by_layer[layer], y_train,
        X_eval_by_layer[layer], y_eval,
        C_values
    )
    C_vals = sorted(reg_results.keys())
    aurocs = [reg_results[c] for c in C_vals]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(C_vals, aurocs, marker='o', linewidth=2, markersize=8, color="#2b6cb0")
    ax.set_xscale('log')
    ax.set_xlabel("Regularization Coefficient (= 1/λ, log scale)")
    ax.set_ylabel("AUROC")
    title = f"AUROC vs. Regularization — Layer {layer}" if layer else "AUROC vs. Regularization"
    ax.set_title(title)
    ax.set_ylim(0.9, 1.05)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f"plots/regularization_sweep_layer{layer}.png", dpi=150)

def main():
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    model = HookedTransformer.from_pretrained_no_processing(model_name, device="cpu", dtype=torch.bfloat16)

    best_layer = 17 # The best perfoming
    training_data = get_training_data(model, layer=best_layer)
    print("Plotting AUROC and recall by layer...")
    plot_auroc_and_recall(training_data_path="data/baseline_training_summary.json", data=training_data)
    print("Plotting regularization sweep...")
    plot_regularization_sweep(training_data, layer=best_layer)
    print("Plotting ROC curve and token-level trajectory for the best layer...")
    plot_roc_and_token_trajectory(model, training_data, layer=best_layer)
    print("All plots saved in the 'plots/' directory.")

if __name__ == "__main__":
    sys.exit(main())
