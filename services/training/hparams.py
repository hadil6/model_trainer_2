"""Evidence-based default hyperparameters for SWIFT fine-tuning.

Design rationale:
  lora_rank    — larger models tolerate smaller rank (QLoRA paper, §4.2)
  lora_alpha   — community standard: alpha = 2 × rank (effective scale = 2)
  learning_rate — 2e-4 for 4-bit (QLoRA paper), 1e-4 for bf16 LoRA, 2e-5 full
  epochs       — LIMA principle: small, curated datasets benefit from more passes
  batch        — effective batch size fixed at 16 via gradient accumulation

Sources:
  QLoRA (Dettmers et al., 2023) — arXiv:2305.14314
  LIMA (Zhou et al., 2023)      — arXiv:2305.11206
  LoRA (Hu et al., 2021)        — arXiv:2106.09685
  SmolLM2 (HuggingFace, 2024)
  Mistral fine-tuning cookbook, 2024
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from data.model_registry import detect_size_b

# ── LoRA rank by model size ────────────────────────────────────────────────────
# QLoRA uses r=64 for 65B; for smaller models lower rank is efficient and sufficient.
_RANK_BY_SIZE: list[tuple[float, int]] = [
    (0.5,          4),   # <500M: minimal rank, few adapter params
    (1.5,          8),   # 500M–1.5B
    (7.0,         16),   # 1.5B–7B: r=16 is the sweet spot (Mistral cookbook)
    (15.0,        32),   # 7B–15B
    (float("inf"), 64),  # >15B: match QLoRA paper default
]

# ── Learning rate by PEFT method ───────────────────────────────────────────────
_LR_BY_PEFT: dict[str, float] = {
    "qlora":     2e-4,   # QLoRA paper default for 4-bit training
    "lora":      1e-4,   # bf16 LoRA — slightly lower (no quantization noise)
    "lora_int8": 1e-4,
    "dora":      1e-4,
    "full":      2e-5,   # full fine-tuning needs much lower LR
    "auto":      2e-4,
}

# ── Epochs by dataset size ─────────────────────────────────────────────────────
# Small curated datasets need more passes (LIMA principle).
_EPOCHS_BY_SIZE: list[tuple[int, int]] = [
    (300,      10),
    (700,       7),
    (1_500,     5),
    (3_000,     3),
    (8_000,     2),
    (999_999,   1),
]

_EFFECTIVE_BATCH = 16  # target effective batch size (industry-standard for fine-tuning)


def get_hparams(
    model_name: str,
    peft: str,
    n_pairs: int,
    vram_gb: int,
    task: str = "question-answering",
) -> dict:
    """Return evidence-based SWIFT training hyperparameters.

    All keys prefixed with '_' are metadata — not forwarded to the CLI.
    """
    size_b = detect_size_b(model_name)

    # ── LoRA rank & alpha ──────────────────────────────────────────────────────
    rank  = next(r for cap, r in _RANK_BY_SIZE if size_b <= cap)
    alpha = rank * 2  # alpha/rank = 2 is the standard effective scaling

    # ── Learning rate ──────────────────────────────────────────────────────────
    lr = _LR_BY_PEFT.get(peft, 2e-4)

    # ── Epochs ────────────────────────────────────────────────────────────────
    epochs = next(e for cap, e in _EPOCHS_BY_SIZE if n_pairs <= cap)

    # ── Physical batch size + gradient accumulation ────────────────────────────
    # Conservative batch sizes to stay within VRAM; accumulate to _EFFECTIVE_BATCH.
    if vram_gb >= 16:
        phys_batch = 4
    elif vram_gb >= 8:
        phys_batch = 2
    else:
        phys_batch = 1
    grad_accum = max(1, _EFFECTIVE_BATCH // phys_batch)

    # ── Context length ─────────────────────────────────────────────────────────
    max_length = 4096 if task == "code-generation" else 2048

    return {
        # LoRA adapter
        "lora_rank":                    rank,
        "lora_alpha":                   alpha,
        "lora_dropout":                 0.05,
        # Optimiser
        "learning_rate":                lr,
        "lr_scheduler_type":            "cosine",
        "warmup_ratio":                 0.05,
        "weight_decay":                 0.01,
        # Training loop
        "num_train_epochs":             epochs,
        "per_device_train_batch_size":  phys_batch,
        "gradient_accumulation_steps":  grad_accum,
        "gradient_checkpointing":       True,
        "max_length":                   max_length,
        # Checkpointing
        "save_strategy":                "epoch",
        "eval_strategy":                "epoch",
        "load_best_model_at_end":       True,
        # Metadata (not forwarded to CLI)
        "_model_size_b":    size_b,
        "_effective_batch": phys_batch * grad_accum,
        "_n_pairs":         n_pairs,
    }


# ── Adaptive max_length ───────────────────────────────────────────────────────

def adaptive_max_length(train_file: Path, task: str = "question-answering") -> int:
    """Compute max_length from actual dataset token statistics.

    Formula: next_power_of_2( (max_chars_per_sample / 4 + 50) × 1.2 )
      - chars/4  : standard token approximation for Latin-script text
      - +50      : overhead for chat-template special tokens (im_start, im_end, roles)
      - ×1.2     : 20% safety margin for tokenizer subword splits
      - power-of-2: GPU tensor-core alignment (Vaswani et al., 2017; O(n²) attention)

    Clamps to [256, 2048] for standard tasks, 4096 for code-generation.
    Falls back to 2048 if the file cannot be read.
    """
    if task == "code-generation":
        return 4096

    try:
        max_chars = 0
        with train_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    chars = sum(
                        len(m.get("content", ""))
                        for m in obj.get("messages", [])
                    )
                    max_chars = max(max_chars, chars)
                except Exception:
                    continue

        if max_chars == 0:
            return 2048

        estimated = int((max_chars / 4 + 50) * 1.2)
        power = math.ceil(math.log2(max(256, estimated)))
        return min(2048, 2 ** power)

    except Exception:
        return 2048


# ── Phase 2: Optuna search space ──────────────────────────────────────────────

def optuna_search_space(trial, peft_method: str = "qlora", n_pairs: int = 500) -> dict:
    """Optuna search space for automated HPO (Phase 2).

    Intended usage:
        import optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: objective(t, ...), n_trials=20)
        best = study.best_params
    """
    rank        = trial.suggest_categorical("lora_rank",        [4, 8, 16, 32, 64])
    alpha_ratio = trial.suggest_categorical("lora_alpha_ratio", [1, 2, 4])

    lr = trial.suggest_float(
        "learning_rate",
        5e-6 if peft_method == "full" else 1e-5,
        1e-4 if peft_method == "full" else 5e-4,
        log=True,
    )

    # Epoch bounds tuned for overfitting risk vs. dataset size (LIMA principle)
    if n_pairs < 300:
        epochs = trial.suggest_int("num_train_epochs", 5, 12)
    elif n_pairs < 700:
        epochs = trial.suggest_int("num_train_epochs", 4, 8)
    elif n_pairs < 2000:
        epochs = trial.suggest_int("num_train_epochs", 2, 6)
    else:
        epochs = trial.suggest_int("num_train_epochs", 1, 3)

    return {
        "lora_rank":        rank,
        "lora_alpha":       alpha_ratio * rank,   # derived; not an Optuna dimension
        "lora_dropout":     trial.suggest_float("lora_dropout",  0.0,  0.10),
        "learning_rate":    lr,
        "warmup_ratio":     trial.suggest_float("warmup_ratio",  0.02, 0.20),
        "weight_decay":     trial.suggest_float("weight_decay",  0.0,  0.10),
        "num_train_epochs": epochs,
    }
