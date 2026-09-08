"""
This is a smoke test for loading models and the proper environment variables. 
It is not meant to be run as part of the main codebase, 
but rather as a quick check that everything is set up correctly.
"""
from dotenv import load_dotenv, find_dotenv
import os
from transformer_lens import HookedTransformer
from huggingface_hub import login
import torch
import argparse
import sys

# Access the model through HF
def init_env(hf_token: str | None):
    dotenv_path = None
    try:
        dotenv_path = find_dotenv()
    except Exception:
        # find_dotenv can fail in some execution contexts; fall back to repo-root .env
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidate = os.path.join(repo_root, ".env")
        if os.path.exists(candidate):
            dotenv_path = candidate

    if dotenv_path:
        load_dotenv(dotenv_path)
        print(f"Loaded .env from: {dotenv_path}")
    else:
        print("No .env file found; relying on environment variables.")

    token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        try:
            login(token=token)
        except Exception as e:
            print("Warning: Hugging Face login failed:", e)
    else:
        print("No HUGGINGFACE_TOKEN provided; proceeding unauthenticated.")


def load_model(model_name: str, device: str = "cpu", dtype: torch.dtype | None = None):
    if device == "cpu" and dtype is None:
        # avoid float16 on CPU which is unsupported in many builds
        dtype = torch.float32
    try:
        model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)
        print("Loaded model:", model_name)
        print(model.cfg.n_layers, model.cfg.d_model)
        return model
    except Exception:
        print(f"Failed to load model {model_name}")
        raise


def check_generate(model, prompt: str = "The capital of France is"):
    tokens = model.to_tokens(prompt)
    output = model.generate(tokens, max_new_tokens=10)
    text = model.to_string(output)
    # Raw output may not make sense
    print("Raw generation result:", text)
    return tokens, output


def check_cache(model, tokens):
    logits, cache = model.run_with_cache(tokens, names_filter=lambda name: "resid_post" in name)
    # pick a sample key to show shape; keys may be tuples ("resid_post", idx) or strings containing 'resid_post'
    key = next(
        (
            k
            for k in cache.keys()
            if (isinstance(k, tuple) and k[0] == "resid_post") or (isinstance(k, str) and "resid_post" in k)
        ),
        None,
    )
    if key is None:
        print("No resid_post cache found; available keys:", list(cache.keys())[:5])
        # return cache and None shape so caller can handle assertion and errors
        return cache, None

    shape = cache[key].shape
    print("Cache sample shape:", shape)
    return cache, shape


def check_chat_template(model):
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    try:
        formatted_prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("Formatted chat prompt:", formatted_prompt)
        tokens = model.to_tokens(formatted_prompt)
        output = model.generate(tokens, max_new_tokens=20)
        text = model.to_string(output)
        print("Chat generation result for formatted prompt:", text)
        return text
    except Exception as e:
        print("Chat template check failed:", e)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run smoke checks for a pretrained model using transformer-lens")
    parser.add_argument("--model-name", "-m", help="HF model repo id to load", default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--device", "-d", help="Device to load model on", default="cpu")
    parser.add_argument("--dtype", help="dtype to use (float16|float32)", choices=["float16", "float32"], default=None)
    parser.add_argument("--hf-token", help="Hugging Face token (optional)", default=None)
    args = parser.parse_args(argv)

    init_env(args.hf_token)

    dtype = None
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32

    try:
        model = load_model(args.model_name, device=args.device, dtype=dtype)
    except Exception as e:
        print("ERROR: model load failed:", e)
        return 2

    try:
        tokens, out = check_generate(model)
    except Exception as e:
        print("Generation check failed:", e)
        return 3

    try:
        cache, shape = check_cache(model, tokens)

        # Assert shape is present and is a 3-tuple of ints: [batch, seq_len, d_model]
        if shape is None:
            raise AssertionError("resid_post cache not found")
        if not hasattr(shape, "__len__") or len(shape) != 3:
            raise AssertionError(f"Cache tensor shape is not 3D [batch, seq_len, d_model]: {shape}")
        for dim in shape:
            if not isinstance(dim, int):
                raise AssertionError(f"Cache tensor shape dimension is not an int: {shape}")

        cache = cache
    except AssertionError as e:
        print("Cache assertion failed:", e)
        return 4
    except Exception as e:
        print("Cache check failed:", e)
        return 4

    try:
        chat_text = check_chat_template(model)
        # Normalize chat_text to a single string for assertion
        if isinstance(chat_text, (list, tuple)):
            chat_str = " ".join(map(str, chat_text))
        else:
            chat_str = str(chat_text)

        assert "Paris" in chat_str, f"Expected 'Paris' in chat output, got: {chat_str}"
        print("Chat generation result contains 'Paris'.")
    except AssertionError as e:
        print("Chat assertion failed:", e)
        return 6
    except Exception as e:
        print("Chat template check failed:", e)
        return 5

    print("All checks completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
