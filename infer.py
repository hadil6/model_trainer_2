"""Interactive inference with a fine-tuned LoRA adapter.

Usage:
    python infer.py <job_id>
    python infer.py <job_id> --question "Ta question ici"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_best_checkpoint(output_dir: str) -> str | None:
    base = Path(output_dir)
    if not base.exists():
        return None

    best_link = base / "best_model"
    if best_link.exists():
        return str(best_link.resolve())

    checkpoints = sorted(
        base.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        if (base / "adapter_config.json").exists():
            return str(base)
        return None

    best_path, best_loss = None, float("inf")
    for cp in checkpoints:
        trainer_state = cp / "trainer_state.json"
        if trainer_state.exists():
            try:
                data = json.loads(trainer_state.read_text(encoding="utf-8"))
                for entry in data.get("log_history", []):
                    if "eval_loss" in entry and entry["eval_loss"] < best_loss:
                        best_loss = entry["eval_loss"]
                        best_path = str(cp)
            except Exception:
                pass

    return best_path or str(checkpoints[-1])


def load_model(job_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from core.config import get_settings

    artifacts_dir = Path(get_settings().artifacts_dir)
    report_path = artifacts_dir / job_id / "training_report.json"

    if not report_path.exists():
        print(f"[ERROR] training_report.json not found for job {job_id}")
        sys.exit(1)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    model_id = report.get("model_id", "")
    output_dir = (report.get("result") or {}).get("output_dir", "")

    # Try direct output_dir, then scan subdirs
    adapter_path = _find_best_checkpoint(output_dir)
    if not adapter_path:
        base = Path(output_dir)
        for sub in sorted(base.iterdir()) if base.exists() else []:
            if sub.is_dir():
                adapter_path = _find_best_checkpoint(str(sub))
                if adapter_path:
                    break

    if not adapter_path:
        print(f"[ERROR] No checkpoint found in {output_dir}")
        sys.exit(1)

    print(f"  model   : {model_id}")
    print(f"  adapter : {adapter_path}")
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print("Ready.\n", flush=True)
    return model, tokenizer


def generate(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    import torch

    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )

    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False)
    # Strip trailing chatml tokens
    import re
    generated = re.sub(r"<\|im_end\|>.*", "", generated, flags=re.DOTALL)
    return generated.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id", help="Job ID (12-char hex)")
    parser.add_argument("--question", "-q", default=None, help="Single question (non-interactive)")
    args = parser.parse_args()

    print(f"\nLoading model for job {args.job_id}...")
    model, tokenizer = load_model(args.job_id)

    if args.question:
        answer = generate(model, tokenizer, args.question)
        print(f"Question : {args.question}")
        print(f"Réponse  : {answer}")
        return

    # Interactive loop
    print("Mode interactif — tape 'exit' pour quitter.\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break
        answer = generate(model, tokenizer, question)
        print(f"Réponse : {answer}\n")


if __name__ == "__main__":
    main()
