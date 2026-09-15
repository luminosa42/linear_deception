"""
Phase 3 — Causal-Use Extension via Activation Patching
Depends on: data/features_train.npz, data/features_eval.npz, data/probe_eval.json
"""
import fractions
import os
import json
import sys
import numpy as np
import torch
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

# Prefer the newer TransformerBridge when available; fall back to HookedTransformer
bridge_cls = None
try:
    from transformer_lens.model_bridge import TransformerBridge as _TransformerBridge
    bridge_cls = _TransformerBridge
except Exception:
    try:
        from transformer_lens import TransformerBridge as _TransformerBridge
        bridge_cls = _TransformerBridge
    except Exception:
        try:
            from transformer_lens import HookedTransformer as _Hooked
            bridge_cls = _Hooked
        except Exception:
            bridge_cls = None

if bridge_cls is None:
    raise ImportError("Neither TransformerBridge nor HookedTransformer found in transformer_lens; please install a compatible version of transformer_lens.")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from probe_pipeline_main import get_activation

BEST_LAYER = 17  # The layer with the best probe performance (from Phase 2)

# JSON serialization helper for numpy / torch objects
def _json_default(o):
        # numpy / torch -> native python
        try:
            import numpy as _np
        except Exception:
            _np = None
        if _np is not None and isinstance(o, _np.generic):
            return o.item()
        try:
            import torch as _torch
        except Exception:
            _torch = None
        if _torch is not None and isinstance(o, _torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, (list, tuple)):
            return list(o)
        return str(o)

def get_probe_direction(X_train_by_layer, y_train, layer=17):
    """
    Retrain (or reload) the best layer probe and return the probe direction.
    """
    # Retrain (or reload) the layer-17 probe
    clf = LogisticRegression(max_iter=1000, C=10.0)  # C=10 was your best regularization point
    clf.fit(X_train_by_layer[layer], y_train)

    # The probe direction is the weight vector, normalized to unit length
    v = clf.coef_[0]  # shape: [d_model]
    v = v / np.linalg.norm(v)
    v_torch = torch.tensor(v, dtype=torch.float32)
    
    return clf, v_torch

def random_control(vector):
    """
    Generate a random control direction of the same dimension as the model's hidden size.
    """
    random_v = np.random.randn(*vector.shape)
    random_v = random_v / np.linalg.norm(random_v)
    random_v_torch = torch.tensor(random_v, dtype=torch.float32)

    return random_v_torch

def generate_baseline(model, prompt_text, max_new_tokens=60):
    tokens = model.to_tokens(prompt_text)
    output = model.generate(tokens, max_new_tokens=max_new_tokens, verbose=False)
    return model.to_string(output[0, tokens.shape[1]:])

# Hook function for intervention (TransformerLens)
def make_intervention_hook(direction, mode="ablate", alpha=1.0):
    """
    mode: "ablate" -> project out the direction
          "inject" -> add alpha * direction
    """
    def hook_fn(resid, hook):
        d = direction.to(resid.dtype)
        if mode == "ablate":
            proj = (resid @ d).unsqueeze(-1) * d  # [batch, seq, d_model]
            return resid - proj
        elif mode == "inject":
            return resid + alpha * d
    return hook_fn

# Generation with intervention active
def generate_with_intervention(model, prompt_text, direction, layer, mode, alpha=1.0, max_new_tokens=60):
    hook_name = f"blocks.{layer}.hook_resid_post"
    hook_fn = make_intervention_hook(direction, mode=mode, alpha=alpha)

    tokens = model.to_tokens(prompt_text)
    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        output = model.generate(tokens, max_new_tokens=max_new_tokens, verbose=False)

    generated = output[0, tokens.shape[1]:]
    return model.to_string(generated)

def score_with_probe(model,text, prompt, clf, layer):

    vec = get_activation(model, prompt, text, layer) 
    return clf.predict_proba(vec.numpy().reshape(1, -1))[:, 1][0]

# Load evals data
def load_eval_data(feature_train_path, feature_eval_path,eval_data_path):

    # Cached activation features (skip re-extraction entirely)
    train_loaded = np.load(feature_train_path)
    eval_loaded = np.load(feature_eval_path)
    y_train = train_loaded["y"]
    y_eval = eval_loaded["y"]

    layers = [int(k.replace("X_layer", "")) for k in train_loaded.files if k.startswith("X_layer")]
    X_train_by_layer = {l: train_loaded[f"X_layer{l}"] for l in layers}
    X_eval_by_layer = {l: eval_loaded[f"X_layer{l}"] for l in layers}

    # Raw prompt/response text — needed for generation + patching (features alone aren't enough here)
    with open(eval_data_path) as f:
        eval_data = json.load(f)

    return X_train_by_layer, y_train, X_eval_by_layer, y_eval, eval_data

def run_intervention_experiment(model, X_train_by_layer, y_train, eval_data, mode="ablate", alpha=1.0):

    results = []
    clf, v_torch = get_probe_direction(X_train_by_layer, y_train, layer=BEST_LAYER)
    random_v_torch = random_control(v_torch)
    deceptive_prompts = [e for e in eval_data if e["label"] == "deceptive"]
    print("Running {0} intervention experiment on {1} deceptive prompts...".format(mode, len(deceptive_prompts)))

    for i, entry in enumerate(deceptive_prompts):
        prompt = entry["prompt"]
        print(f"Working on prompt: {i}")

        baseline_text = generate_baseline(model, prompt)
        print("Generated baseline.")
        intervention_text = generate_with_intervention(model, prompt, v_torch, BEST_LAYER, mode=mode, alpha=alpha)
        print("Generated intervention.")
        control_text = generate_with_intervention(model, prompt, random_v_torch, BEST_LAYER, mode=mode)
        print("Generated control.")

        results.append({
            "prompt": prompt,
            "baseline_text": baseline_text,
            "intervention_text": intervention_text,
            "control_text": control_text,
            "baseline_score": score_with_probe(model, baseline_text, prompt, clf, BEST_LAYER),
            "intervention_score": score_with_probe(model, intervention_text, prompt, clf, BEST_LAYER),
            "control_score": score_with_probe(model, control_text, prompt, clf, BEST_LAYER),
        })
        print(f"baseline={results[-1]['baseline_score']:.3f}  "
            f"intervention={results[-1]['intervention_score']:.3f}  "
          f"control={results[-1]['control_score']:.3f}")

    return results

def is_refusal(text):
    """
    Check if the model refuses to answer the prompt, which is a form of intervention.
    """
    refusal_markers = [
            "i'm sorry", "i can't assist", "i cannot assist", "i can't help",
            "i cannot help", "as an ai", "i'm not able to", "i am not able to"
    ]
    text_lower = text.lower()
    return any(marker in text_lower for marker in refusal_markers)

def test_alpha_level(results_at_alpha):
    true_effect = np.array([r["injected_score"] - r["baseline_score"] for r in results_at_alpha])
    control_effect = np.array([r["control_score"] - r["baseline_score"] for r in results_at_alpha])
    diff = true_effect - control_effect
    stat, p = wilcoxon(diff)
    return true_effect.mean(), control_effect.mean(), diff.mean(), p

def run_injection_alpha_sweep(model, X_train_by_layer, y_train, eval_data, layer, alphas):
    """
    Calibration run: run a sweep of injection experiments over a range of alpha values.
    """
    sweep_results = {}
    clf, v_torch = get_probe_direction(X_train_by_layer, y_train, layer=BEST_LAYER)
    random_v_torch = random_control(v_torch)
    honest_prompts = [e for e in eval_data if e["label"] == "honest"]
    for alpha in alphas:
        scores_true, scores_control, refusals_true, refusals_control = [], [], [], []
        for entry in honest_prompts:
            prompt = entry["prompt"]
            baseline_text = generate_baseline(model,prompt)
            injected_text = generate_with_intervention(model,prompt, v_torch, layer, mode="inject", alpha=alpha)
            control_text = generate_with_intervention(model, prompt, random_v_torch, layer, mode="inject", alpha=alpha)

            scores_true.append(score_with_probe(model, injected_text, prompt, clf, layer) - score_with_probe(model, baseline_text, prompt, clf, layer))
            scores_control.append(score_with_probe(model, control_text, prompt, clf, layer) - score_with_probe(model, baseline_text, prompt, clf, layer))
            refusals_true.append(is_refusal(injected_text))
            refusals_control.append(is_refusal(control_text))

        sweep_results[alpha] = {
            "mean_true_effect": np.mean(scores_true),
            "mean_control_effect": np.mean(scores_control),
            "refusal_rate_true": np.mean(refusals_true),
            "refusal_rate_control": np.mean(refusals_control),
        }
        print(f"α={alpha:.1f}: true={sweep_results[alpha]['mean_true_effect']:.3f}, "
              f"control={sweep_results[alpha]['mean_control_effect']:.3f}, "
              f"refusals(true/control)={sweep_results[alpha]['refusal_rate_true']:.2f}/{sweep_results[alpha]['refusal_rate_control']:.2f}")

    # write sweep results to json for later analysis
    with open("data/injection_alpha_sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2, default=_json_default)

def select_best_alpha(sweep_results_path):

    with open(sweep_results_path) as f:
        sweep_results = json.load(f)

    # Find true vs control gap and significance for each alpha
    for alpha, results_at_alpha in sweep_results.items():
        # Aggregated format: mapping alpha -> {mean_true_effect, mean_control_effect, ...}
        if isinstance(results_at_alpha, dict) and "mean_true_effect" in results_at_alpha:
            true_m = results_at_alpha.get("mean_true_effect", np.nan)
            control_m = results_at_alpha.get("mean_control_effect", np.nan)
            gap = true_m - control_m
            p = np.nan
            try:
                alpha_f = float(alpha)
            except Exception:
                alpha_f = alpha
            print(f"α={alpha_f}: true={true_m:.3f}, control={control_m:.3f}, gap={gap:.3f}, p={p}")
        else:
            true_m, control_m, gap, p = test_alpha_level(results_at_alpha)
            try:
                alpha_f = float(alpha)
            except Exception:
                alpha_f = alpha
            print(f"α={alpha_f:.3f}: true={true_m:.3f}, control={control_m:.3f}, gap={gap:.3f}, p={p:.4f}")
    
    # Find the alpha that gives the largest mean true effect while keeping the refusal rate below 0.2
    best_alpha = None
    best_mean_true_effect = -np.inf
    for alpha, results in sweep_results.items():
        if results["refusal_rate_true"] < 0.2 and results["mean_true_effect"] > best_mean_true_effect:
                best_mean_true_effect = results["mean_true_effect"]
                try:
                    best_alpha = float(alpha)
                except Exception:
                    best_alpha = alpha
    if isinstance(best_alpha, (int, float)):
        print(f"Best alpha for injection: {best_alpha:.3f} (mean true effect: {best_mean_true_effect:.3f})")
    else:
        print(f"Best alpha for injection: {best_alpha} (mean true effect: {best_mean_true_effect:.3f})")

    return best_alpha    

def plot_alpha_sweep_results(sweep_result_path):

    with open(sweep_result_path) as f:
        sweep_results = json.load(f)

    # sweep_results may be a mapping from stringified alpha -> aggregated stats
    # Convert keys to numeric alphas and extract aggregated fields
    items = sorted((float(k), v) for k, v in sweep_results.items())
    alphas = [a for a, _ in items]
    true_means = [v.get("mean_true_effect", np.nan) for _, v in items]
    control_means = [v.get("mean_control_effect", np.nan) for _, v in items]

    plt.figure(figsize=(7,5))
    plt.plot(alphas, true_means, marker='o', label="True direction", color="#2b6cb0")
    plt.plot(alphas, control_means, marker='s', label="Random control", color="#c0392b")
    # Use log scale only if alphas are all positive
    if all(a > 0 for a in alphas):
        plt.xscale('log')
    plt.xlabel("Injection magnitude α")
    plt.ylabel("Mean P(deceptive) after injection")
    plt.title("Injection dose-response: true direction vs. random control")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/injection_dose_response.png", dpi=150)


def generate_intervention_results():
    # Load the model
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Prefer reduced precision on GPU; use float32 on CPU to avoid unsupported dtypes
    if device == "cpu":
        dtype = torch.float32
    else:
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        dtype = torch.bfloat16 if bf16_ok else torch.float16

    # Use whichever loader method the detected class exposes
    if hasattr(bridge_cls, "from_pretrained_no_processing"):
        model = bridge_cls.from_pretrained_no_processing(model_name, device=device, dtype=dtype)
    elif hasattr(bridge_cls, "boot_transformers"):
        model = bridge_cls.boot_transformers(model_name, device=device, dtype=dtype)
    else:
        raise RuntimeError("The transformer_lens class found does not expose a supported loader method (from_pretrained_no_processing or boot_transformers).")

    # Load eval data
    feature_train_path = "data/features_train.npz"
    feature_eval_path = "data/features_eval.npz"
    eval_data_path = "data/probe_eval.json"
    X_train_by_layer, y_train, _, _, eval_data = load_eval_data(feature_train_path, feature_eval_path, eval_data_path)

    print(f"Loaded {len(eval_data)} eval prompts, {len(y_train)} training examples.")

    # Calibrate alpha for injection experiments
    typical_norm = np.linalg.norm(X_train_by_layer[17], axis=1).mean()
    fractions = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    alphas = [f * typical_norm for f in fractions]
    #run_injection_alpha_sweep(model, X_train_by_layer, y_train, eval_data, layer=BEST_LAYER, alphas=alphas)  
    plot_alpha_sweep_results("data/injection_alpha_sweep_results.json")
    best_alpha = select_best_alpha("data/injection_alpha_sweep_results.json")

    """
    # Run intervention experiment (ablation) and save result to json
    ablate_results = run_intervention_experiment(model, X_train_by_layer, y_train, eval_data, mode="ablate", alpha=4.0)

    with open("data/intervention_results_ablate.json", "w") as f:
        json.dump(ablate_results, f, indent=2, default=_json_default)

    # Run intervention experiment (injection) and save result to json
    inject_results = run_intervention_experiment(model, X_train_by_layer, y_train,
                                                eval_data, mode="inject", alpha=4.0)
    with open("data/intervention_results_inject.json", "w") as f:
        json.dump(inject_results, f, indent=2, default=_json_default)

    print("Intervention experiments completed.")
    """

def plot_paired_intervention(abl, inj):

    # Input are ablation and injection scores
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (label, before, after) in zip(axes, [
        ("Ablation", abl["baseline_score"], abl["intervention_score"]),
        ("Injection", inj["baseline_score"], inj["intervention_score"]),
    ]):
        for b, a in zip(before, after):
            ax.plot([0, 1], [b, a], color="gray", alpha=0.4)
        ax.scatter(np.zeros_like(before), before, color="#2b6cb0", zorder=3, label="Baseline")
        ax.scatter(np.ones_like(after), after, color="#c0392b", zorder=3, label="After Intervention")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline", label])
        ax.set_ylabel("P(deceptive) — probe score")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()

    plt.tight_layout()
    plt.savefig("plots/paired_intervention_plot.png", dpi=150)

def check_refusal(inj_results):
    """
    Check if the model refuses to answer the prompt, which is a form of intervention.
    """
    
    inv_refusal_rate = np.mean([is_refusal(r["intervention_text"]) for r in inj_results])
    control_refusal_rate = np.mean([is_refusal(r["control_text"]) for r in inj_results])
    baseline_refusal_rate = np.mean([is_refusal(r["baseline_text"]) for r in inj_results])

    print(f"Baseline refusal rate: {baseline_refusal_rate:.2f}")
    print(f"True-direction injection refusal rate: {inv_refusal_rate:.2f}")
    print(f"Random-control injection refusal rate: {control_refusal_rate:.2f}")

    non_refusal_injection = [r for r in inj_results if not is_refusal(r["intervention_text"]) and not is_refusal(r["control_text"])]
    print(f"Usable pairs after excluding refusals: {len(non_refusal_injection)} / {len(inj_results)}")

    if len(non_refusal_injection) >= 5:  # only proceed if you have enough left
        true_effect_clean = np.array([r["baseline_score"] - r["intervention_score"] for r in non_refusal_injection])
        control_effect_clean = np.array([r["baseline_score"] - r["control_score"] for r in non_refusal_injection])
        print(f"Cleaned true effect: {true_effect_clean.mean():.3f}, control: {control_effect_clean.mean():.3f}")
        return non_refusal_injection
    else:
        print("Not enough non-refusal pairs to analyze.")
        return []



def analyze_intervention_results():

    def _extract_scores(results, keys):
        return {k: np.array([r[k] for r in results]) for k in keys}
    def _paired_cohens_d(effect):
        return effect.mean() / effect.std(ddof=1)

    # Load results
    with open("data/intervention_results_ablate.json") as f:
        ablate_results = json.load(f)
    with open("data/intervention_results_inject.json") as f:
        inject_results = json.load(f)

    # Check refusal rates and remove them if needed
    non_refusal_ablate = check_refusal(ablate_results)
    non_refusal_inject = check_refusal(inject_results)

    abl = _extract_scores(non_refusal_ablate, ["baseline_score", "intervention_score", "control_score"])
    inj = _extract_scores(non_refusal_inject, ["baseline_score", "intervention_score", "control_score"])

    # Basic checks
    abl["true_effect"] = abl["baseline_score"] - abl["intervention_score"]      # expect positive if causal
    abl["control_effect"] = abl["baseline_score"] - abl["control_score"]   # expect ~0

    inj["true_effect"] = inj["intervention_score"] - inj["baseline_score"]     # expect positive if causal
    inj["control_effect"] = inj["control_score"] - inj["baseline_score"]   # expect ~0

    print(f"Ablation — mean true effect: {abl['true_effect'].mean():.3f}, mean control effect: {abl['control_effect'].mean():.3f}")
    print(f"Injection — mean true effect: {inj['true_effect'].mean():.3f}, mean control effect: {inj['control_effect'].mean():.3f}")

    # Effect size (Cohen's d for paired samples)
    d_ablation = _paired_cohens_d(abl["true_effect"])
    d_ablation_control = _paired_cohens_d(abl["control_effect"])
    print(f"Ablation Cohen's d (true): {d_ablation:.2f}, (control): {d_ablation_control:.2f}")

    # Significance testing (Wilcoxon signed-rank test)
    stat, p_true = wilcoxon(abl["true_effect"])
    stat_ctrl, p_control = wilcoxon(abl["control_effect"])
    print(f"Ablation true effect: p={p_true:.4f}; control effect: p={p_control:.4f}")
    diff = abl["true_effect"] - abl["control_effect"]
    stat_diff, p_diff = wilcoxon(diff)
    print(f"True vs. control effect difference: mean={diff.mean():.3f}, p={p_diff:.4f}")

    # Visualisation
    plot_paired_intervention(abl, inj)
    print("Analysis done.")

if __name__ == "__main__":
    sys.exit(generate_intervention_results())
