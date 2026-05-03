"""
Swahili GPT — Inference Script
================================
Loads a trained MiniGPT checkpoint from an Orbax checkpoint directory
and generates Kiswahili text from a given prompt.

Usage:
    python inference.py --prompt "Habari za asubuhi" --steps 100

Requirements:
    pip install jax flax optax orbax-checkpoint tokenizers

Notes:
    - Point --ckpt_dir at your Orbax checkpoint folder.
    - Place kenya_tokenizer.json in the same folder or pass --tokenizer_path.
    - Adjust --temperature for creativity: lower = safer, higher = more creative.
"""

import argparse
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from tokenizers import Tokenizer

# ── Architecture (must match training config exactly) ──────────────────────────

class Block(nnx.Module):
    def __init__(self, emb_size, num_heads, rngs):
        self.ln_1 = nnx.LayerNorm(emb_size, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=emb_size, decode=False, rngs=rngs
        )
        self.ln_2 = nnx.LayerNorm(emb_size, rngs=rngs)
        self.mlp  = nnx.Sequential(
            nnx.Linear(emb_size, 4 * emb_size, rngs=rngs),
            nnx.gelu,
            nnx.Linear(4 * emb_size, emb_size, rngs=rngs)
        )

    def __call__(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nnx.Module):
    def __init__(self, vocab_size, seq_len, emb_size, num_heads, num_layers, rngs):
        self.token_emb = nnx.Embed(vocab_size, emb_size, rngs=rngs)
        self.pos_emb   = nnx.Embed(seq_len, emb_size, rngs=rngs)
        self.blocks    = nnx.Sequential(
            *[Block(emb_size, num_heads, rngs) for _ in range(num_layers)]
        )
        self.ln_f    = nnx.LayerNorm(emb_size, rngs=rngs)
        self.lm_head = nnx.Linear(emb_size, vocab_size, rngs=rngs)

    def __call__(self, idx):
        b, t = idx.shape
        pos  = jnp.arange(0, t, dtype=jnp.int32)[None, :]
        x    = self.token_emb(idx) + self.pos_emb(pos)
        mask = nnx.make_causal_mask(jnp.ones((b, t)))
        for block in self.blocks.layers:
            x = block(x, mask)
        return self.lm_head(self.ln_f(x))


# ── Generation ─────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100,
             temperature: float = 0.8, top_k: int = 50) -> str:
    """
    Generate text token-by-token using top-k sampling with temperature scaling.

    Args:
        model:          Loaded MiniGPT model.
        tokenizer:      Loaded HuggingFace tokenizers.Tokenizer.
        prompt:         Kiswahili text seed string.
        max_new_tokens: Number of new tokens to generate.
        temperature:    Sampling temperature. Lower = more conservative.
        top_k:          Only sample from the top-k most likely tokens.

    Returns:
        Full generated string (prompt + new text).
    """
    rng = jax.random.PRNGKey(42)
    tokens = tokenizer.encode(prompt).ids

    for _ in range(max_new_tokens):
        # Truncate to context window
        ctx = tokens[-1024:]
        idx = jnp.array([ctx])

        # Forward pass
        logits = model(idx)               # (1, t, vocab_size)
        logits = logits[0, -1, :]         # Last token's logits

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering: zero out everything outside top-k
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        sampled_idx = jax.random.categorical(rng, top_k_logits)
        next_token  = int(top_k_indices[sampled_idx])

        tokens.append(next_token)
        rng = jax.random.fold_in(rng, next_token)  # Advance RNG state

    return tokenizer.decode(tokens)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Swahili GPT Text Generation")
    parser.add_argument("--ckpt_dir",       type=str,   required=True,
                        help="Path to the Orbax checkpoint directory (e.g. ./MiniGPT_Kiswahili_Checkpoints)")
    parser.add_argument("--tokenizer_path", type=str,   default=None,
                        help="Path to kenya_tokenizer.json. Defaults to <ckpt_dir>/kenya_tokenizer.json")
    parser.add_argument("--prompt",         type=str,   default="Habari za",
                        help="Kiswahili prompt to complete")
    parser.add_argument("--steps",          type=int,   default=100,
                        help="Number of new tokens to generate")
    parser.add_argument("--temperature",    type=float, default=0.8,
                        help="Sampling temperature (0.1=conservative, 1.5=creative)")
    parser.add_argument("--top_k",          type=int,   default=50,
                        help="Top-k sampling pool size")
    # Model config — must match your training hyperparameters
    parser.add_argument("--seq_len",    type=int, default=1024)
    parser.add_argument("--emb_size",   type=int, default=512)
    parser.add_argument("--num_heads",  type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path or f"{args.ckpt_dir}/kenya_tokenizer.json"

    # 1. Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer  = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    # 2. Build an empty model with the same architecture as training
    print("Initializing model architecture...")
    rngs  = nnx.Rngs(0)
    model = MiniGPT(vocab_size, args.seq_len, args.emb_size,
                    args.num_heads, args.num_layers, rngs)

    # 3. Load weights from Orbax checkpoint
    print(f"Loading checkpoint from {args.ckpt_dir}...")
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    mngr    = ocp.CheckpointManager(args.ckpt_dir, options=options)

    if mngr.latest_step() is None:
        raise FileNotFoundError(
            f"No checkpoint found in {args.ckpt_dir}.\n"
            "Make sure you have trained the model and the folder exists."
        )

    _, model_state = nnx.split(model)
    state_tree     = {'model': model_state, 'opt': None}
    restored       = mngr.restore(mngr.latest_step(),
                                  args=ocp.args.StandardRestore({'model': model_state}))
    nnx.update(model, restored['model'])
    print(f"Weights loaded from step {mngr.latest_step()}.")

    # 4. Generate text
    print(f"\nPrompt: \"{args.prompt}\"")
    print("-" * 50)
    output = generate(
        model, tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.steps,
        temperature=args.temperature,
        top_k=args.top_k
    )
    print(output)
    print("-" * 50)


if __name__ == "__main__":
    main()
