"""
Shared helpers for the linear-deception notebooks.

The analysis itself lives in the sibling .ipynb notebooks; this module holds only
the pieces more than one of them needs: path setup, model loading, activation
extraction, cached-feature loading and probe fitting.
"""
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FEATURES_TRAIN_PATH = "data/features_train.npz"
FEATURES_EVAL_PATH = "data/features_eval.npz"


# ---------------------------------------------------------------- paths / env

def repo_root(start=None):
    """The directory containing .git, searching upward from `start` (default: cwd)."""
    start = Path(start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def ensure_dirs():
    """Create data/ and plots/ if missing — both are gitignored, so a fresh clone lacks them."""
    for sub in ("data", "plots"):
        Path(sub).mkdir(exist_ok=True)


def init_env(hf_token=None):
    """
    Load .env from the repo root and log in to Hugging Face if a token is available.

    Only needed to download a gated or not-yet-cached model; the imports are local
    so that python-dotenv / huggingface_hub stay optional for the other notebooks.
    """
    from dotenv import load_dotenv

    env_path = repo_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")
    else:
        print("No .env file found; relying on environment variables.")

    token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("No HUGGINGFACE_TOKEN provided; proceeding unauthenticated.")
        return
    from huggingface_hub import login

    try:
        login(token=token)
    except Exception as e:
        print("Warning: Hugging Face login failed:", e)


# -------------------------------------------------------------- model loading

def _model_class(prefer_bridge=False):
    """
    Resolve which transformer_lens class to load with.

    intervention.ipynb prefers the newer TransformerBridge where available; the
    other notebooks were written against HookedTransformer and stay on it so
    their cached features remain comparable.
    """
    if prefer_bridge:
        for module_name, attr in (
            ("transformer_lens.model_bridge", "TransformerBridge"),
            ("transformer_lens", "TransformerBridge"),
        ):
            try:
                return getattr(__import__(module_name, fromlist=[attr]), attr)
            except (ImportError, AttributeError):
                continue
    try:
        from transformer_lens import HookedTransformer

        return HookedTransformer
    except ImportError as e:
        raise ImportError(
            "Neither TransformerBridge nor HookedTransformer found in transformer_lens; "
            "please install a compatible version of transformer_lens."
        ) from e


def resolve_dtype(device):
    """float32 on CPU (float16 is unsupported in many CPU builds), bf16/fp16 on GPU."""
    if device == "cpu":
        return torch.float32
    try:
        bf16_ok = torch.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    return torch.bfloat16 if bf16_ok else torch.float16


def load_model(model_name=None, device=None, dtype=None, no_processing=True, prefer_bridge=False):
    """
    Load the model under test.

    Defaults match the original scripts: MODEL_NAME from the environment, and
    `from_pretrained_no_processing` so activations stay in the model's own basis.
    Pass device/dtype explicitly to pin them (the feature caches in data/ were
    extracted on CPU in bfloat16).
    """
    model_name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = resolve_dtype(device)

    cls = _model_class(prefer_bridge=prefer_bridge)
    loader_names = (
        ("from_pretrained_no_processing", "boot_transformers")
        if no_processing
        else ("from_pretrained",)
    )
    for loader_name in loader_names:
        loader = getattr(cls, loader_name, None)
        if loader is None:
            continue
        model = loader(model_name, device=device, dtype=dtype)
        print(f"Loaded {model_name} via {cls.__name__}.{loader_name} on {device} ({dtype}).")
        print(f"n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")
        return model
    raise RuntimeError(
        f"{cls.__name__} exposes none of the supported loaders {loader_names}."
    )


# ------------------------------------------------------- activations / probes

def get_activation(model, prompt_text, response_text, layer):
    """Mean-pooled residual-stream activation over the response tokens only."""
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


def load_features(train_path=FEATURES_TRAIN_PATH, eval_path=FEATURES_EVAL_PATH,
                  layers=None, verbose=True):
    """
    Load the cached per-layer activation features written by probe_pipeline_main.ipynb.

    Returns (X_train_by_layer, y_train, X_eval_by_layer, y_eval). `layers` defaults
    to whatever X_layer* keys the train file actually contains.
    """
    train_loaded = np.load(train_path)
    eval_loaded = np.load(eval_path)

    if layers is None:
        layers = sorted(
            int(k.removeprefix("X_layer")) for k in train_loaded.files if k.startswith("X_layer")
        )

    X_train_by_layer = {l: train_loaded[f"X_layer{l}"] for l in layers}
    X_eval_by_layer = {l: eval_loaded[f"X_layer{l}"] for l in layers}
    y_train, y_eval = train_loaded["y"], eval_loaded["y"]

    if verbose:
        print("Available layers:", layers)
        print(f"Train: {len(y_train)} examples ({int(y_train.sum())} deceptive)")
        print(f"Eval:  {len(y_eval)} examples ({int(y_eval.sum())} deceptive)")

    return X_train_by_layer, y_train, X_eval_by_layer, y_eval


def fit_probe(X, y, C=1.0, max_iter=1000):
    """Fit the linear probe used throughout: L2 logistic regression on activations."""
    clf = LogisticRegression(max_iter=max_iter, C=C)
    clf.fit(X, y)
    return clf


def json_default(o):
    """`json.dump(default=...)` hook for numpy scalars and torch tensors."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, (list, tuple)):
        return list(o)
    return str(o)
