"""Evaluate a fine-tuned LoRA adapter on the held-out test set.

Metrics computed (task-adaptive — état de l'art 2024-2025)
-----------------------------------------------------------
ALL generation tasks (always):
  ROUGE-1, ROUGE-2, ROUGE-L  — lexical recall        (Lin 2004)
  BLEU-4                      — n-gram precision      (Papineni 2002)
  F1-Token Overlap            — SQuAD-style token F1  (Rajpurkar 2016)
  Exact Match (EM)            — strict equality

OPTIONAL (if bert-score installed):
  BERTScore P/R/F1            — semantic similarity via BERT (Zhang 2020)

SUMMARIZATION:
  Faithfulness (LLM)          — hallucination detection
  G-Eval: coherence, consistency, fluency, relevance  (Liu 2023)

QUESTION-ANSWERING:
  Faithfulness (LLM)          — answer grounded in source knowledge
  G-Eval: correctness, faithfulness, completeness, conciseness

CHATBOT:
  G-Eval: coherence, helpfulness, fluency, engagement

CODE-GENERATION:
  CodeBLEU (if codebleu installed)  — structural code quality (Ren 2020)
  G-Eval: correctness, completeness, readability, syntax

CLASSIFICATION:
  Accuracy, F1-macro, Precision, Recall

NER:
  Entity F1, Precision, Recall

Public API
----------
evaluate(job_id, adapter_path, base_model_id, n_samples, use_llm_judge,
         eval_strategy) -> dict
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── ROUGE ─────────────────────────────────────────────────────────────────────
from rouge_score import rouge_scorer as _rouge_scorer_mod

# ── BLEU ──────────────────────────────────────────────────────────────────────
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from data.artifact_store import test_path, eval_qa_path, read_jsonl, write_json, eval_report_path


# ── Prompt formatting (chatml — matches SWIFT training template) ───────────────
def _format_prompt(instruction: str) -> str:
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _strip_template(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        text = text[len(prompt):]
    text = re.sub(r"<\|im_end\|>.*", "", text, flags=re.DOTALL)
    return text.strip()


# ── ROUGE + BLEU ──────────────────────────────────────────────────────────────
_ROUGE = _rouge_scorer_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
_SMOOTH = SmoothingFunction().method1


def _rouge_scores(reference: str, hypothesis: str) -> dict[str, float]:
    scores = _ROUGE.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def _bleu4(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0
    return round(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=_SMOOTH), 4)


# ── F1 Token Overlap + Exact Match (SQuAD standard) ──────────────────────────

def _f1_token(reference: str, hypothesis: str) -> float:
    """Token-level F1 — standard SQuAD metric (Rajpurkar et al. 2016)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    common = set(ref_tokens) & set(hyp_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(hyp_tokens)
    recall    = len(common) / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def _exact_match(reference: str, hypothesis: str) -> float:
    """Exact Match — 1.0 if strings are identical after lowercasing and stripping."""
    return 1.0 if reference.strip().lower() == hypothesis.strip().lower() else 0.0


# ── BERTScore (optional — pip install bert-score) ─────────────────────────────

def _compute_bertscore(
    references: list[str],
    hypotheses: list[str],
    lang: str = "en",
) -> dict[str, float]:
    """Semantic similarity via BERT embeddings (Zhang et al. 2020).
    Returns empty dict if bert-score is not installed.
    """
    try:
        from bert_score import score as bs_score
        import torch
        import platform
        # On low-VRAM GPUs (< 6 GB) Windows hangs both on GPU (silent CUDA
        # deadlock) and on CPU (RoBERTa-large loading + swap thrashing kills
        # the process for 20+ min). Skip entirely on those configs.
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            free_gb  = torch.cuda.mem_get_info()[0] / 1e9
            if platform.system() == "Windows" and total_gb < 6.0:
                logger.info(
                    "bertscore SKIPPED: Windows + small GPU (%.1f GB) — set EVAL_FORCE_BERTSCORE=1 to override",
                    total_gb,
                )
                if os.environ.get("EVAL_FORCE_BERTSCORE", "").lower() not in ("1", "true", "yes"):
                    return {}
                device = "cpu"
            elif total_gb >= 6.0 and free_gb >= 2.0:
                device = "cuda"
            else:
                device = "cpu"
                logger.info(
                    "bertscore: forcing CPU (total=%.1fGB free=%.1fGB)",
                    total_gb, free_gb,
                )
        else:
            device = "cpu"
        bs_lang = lang if lang in ("en", "fr", "de", "zh", "ar", "es", "pt") else "en"
        P, R, F1 = bs_score(
            hypotheses, references,
            lang=bs_lang,
            device=device,
            verbose=False,
            batch_size=8,
        )
        result = {
            "bertscore_precision": round(float(P.mean()), 4),
            "bertscore_recall":    round(float(R.mean()), 4),
            "bertscore_f1":        round(float(F1.mean()), 4),
        }
        logger.info(
            "bertscore lang=%s P=%.3f R=%.3f F1=%.3f",
            bs_lang, result["bertscore_precision"],
            result["bertscore_recall"], result["bertscore_f1"],
        )
        return result
    except ImportError:
        logger.info("bert-score not installed (pip install bert-score) — BERTScore skipped")
        return {}
    except Exception as exc:
        logger.warning("bertscore_failed: %s", exc)
        return {}


# ── CodeBLEU (optional — pip install codebleu) ────────────────────────────────

def _compute_codebleu(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """CodeBLEU — structural code quality metric (Ren et al. 2020).
    Returns empty dict if codebleu is not installed.
    """
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu(
            [[r] for r in references],
            hypotheses,
            lang="python",
            weights=(0.25, 0.25, 0.25, 0.25),
        )
        score = round(float(result.get("codebleu", 0.0)), 4)
        logger.info("codebleu=%.3f", score)
        return {"codebleu": score}
    except ImportError:
        logger.info("codebleu not installed (pip install codebleu) — CodeBLEU skipped")
        return {}
    except Exception as exc:
        logger.warning("codebleu_failed: %s", exc)
        return {}


# ── G-Eval: task-specific LLM evaluation (Liu et al. 2023) ───────────────────

_G_EVAL_DIMS: dict[str, list[tuple[str, str]]] = {
    "summarization": [
        ("coherence",    "Is the summary coherent and well-structured? Rate 1-5."),
        ("consistency",  "Does the summary contain ONLY facts from the reference? Rate 1-5."),
        ("fluency",      "Is the summary grammatically correct and fluent? Rate 1-5."),
        ("relevance",    "Does the summary cover the key points of the source? Rate 1-5."),
    ],
    "question-answering": [
        ("correctness",   "Is the answer factually correct based on the reference? Rate 1-5."),
        ("faithfulness",  "Is the answer grounded in the source knowledge without hallucination? Rate 1-5."),
        ("completeness",  "Does the answer cover all important aspects of the question? Rate 1-5."),
        ("conciseness",   "Is the answer concise without irrelevant information? Rate 1-5."),
    ],
    "chatbot": [
        ("coherence",    "Is the response coherent and on-topic? Rate 1-5."),
        ("helpfulness",  "Is the response helpful and relevant to the user's message? Rate 1-5."),
        ("fluency",      "Is the response grammatically correct and natural? Rate 1-5."),
        ("engagement",   "Is the response engaging and appropriate in tone? Rate 1-5."),
    ],
    "code-generation": [
        ("correctness",  "Is the generated code logically correct for the task? Rate 1-5."),
        ("completeness", "Does the code implement all required functionality? Rate 1-5."),
        ("readability",  "Is the code clean, readable, and well-structured? Rate 1-5."),
        ("syntax",       "Is the syntax valid with no obvious errors? Rate 1-5."),
    ],
}

_G_EVAL_SYSTEM = (
    "You are an expert NLP evaluator applying the G-Eval framework. "
    "Rate each criterion from 1 (very poor) to 5 (excellent). Be objective and consistent."
)


def _g_eval(
    results: list[dict],
    task: str,
    llm_client,
    n: int = 10,
) -> dict[str, float]:
    """G-Eval: task-specific LLM evaluation with multiple quality dimensions.
    Scores are normalized to [0, 1] by dividing by 5.
    """
    dims = _G_EVAL_DIMS.get(task, _G_EVAL_DIMS["question-answering"])
    dim_names       = [d[0] for d in dims]
    dim_descriptions = "\n".join(f"- {d[0]}: {d[1]}" for d in dims)

    scores: dict[str, list[float]] = {d: [] for d in dim_names}
    sample = random.sample(results, min(n, len(results)))

    for r in sample:
        prompt = (
            f"Instruction/Question: {r['instruction'][:300]}\n"
            f"Reference answer: {r['reference'][:400]}\n"
            f"Generated answer: {r['generated'][:400]}\n\n"
            f"Evaluate on these criteria:\n{dim_descriptions}\n\n"
            f"Return ONLY JSON: {{{', '.join(f'\"g_{d}\": <int 1-5>' for d in dim_names)}}}"
        )
        try:
            raw = llm_client.complete(system=_G_EVAL_SYSTEM, user=prompt)
            m = re.search(r"\{[\s\S]*?\}", raw)
            if m:
                j = json.loads(m.group(0))
                for d in dim_names:
                    val = j.get(f"g_{d}") or j.get(d)
                    if val is not None:
                        scores[d].append(max(1.0, min(5.0, float(val))))
        except Exception as exc:
            logger.warning("g_eval_sample_failed: %s", exc)

    out: dict[str, float] = {}
    for d in dim_names:
        if scores[d]:
            out[f"g_eval_{d}"] = round(sum(scores[d]) / len(scores[d]) / 5.0, 4)

    if out:
        out["g_eval_overall"] = round(sum(out.values()) / len(out), 4)
        logger.info("g_eval task=%s overall=%.3f n=%d", task, out["g_eval_overall"], len(sample))

    return out


# ── Faithfulness via LLM (summarization + QA) ────────────────────────────────

_FAITHFULNESS_SYSTEM = (
    "You are a factual consistency evaluator. "
    "Determine whether the generated text contains ONLY information "
    "that can be inferred from the reference. "
    "Penalize any fact that is not supported by the reference (hallucination)."
)


def _faithfulness_score(
    results: list[dict],
    llm_client,
    n: int = 10,
) -> dict[str, float]:
    """LLM-based faithfulness: measures whether the model hallucinated.
    Score 0 = fully hallucinated, 1 = fully faithful.
    """
    scores: list[float] = []
    sample = random.sample(results, min(n, len(results)))

    for r in sample:
        prompt = (
            f"Reference (ground truth): {r['reference'][:500]}\n"
            f"Generated text: {r['generated'][:500]}\n\n"
            "Score how faithful the generated text is to the reference.\n"
            "1.0 = every fact in the generated text is in the reference\n"
            "0.0 = the generated text invents facts not in the reference\n"
            'Return ONLY JSON: {"faithfulness": <float 0.0-1.0>, "reason": "one sentence"}'
        )
        try:
            raw = llm_client.complete(system=_FAITHFULNESS_SYSTEM, user=prompt)
            m = re.search(r"\{[\s\S]*?\}", raw)
            if m:
                j = json.loads(m.group(0))
                scores.append(max(0.0, min(1.0, float(j.get("faithfulness", 0.5)))))
        except Exception as exc:
            logger.warning("faithfulness_failed: %s", exc)

    if not scores:
        return {}

    result = {"faithfulness": round(sum(scores) / len(scores), 4)}
    logger.info("faithfulness score=%.3f n=%d", result["faithfulness"], len(scores))
    return result


# ── Model loading ─────────────────────────────────────────────────────────────
def _flush_gpu_cache() -> None:
    import time
    import torch
    from data.model_zoo.embeddings import release_sentence_transformer

    release_sentence_transformer()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        time.sleep(3)  # wait for CUDA driver to release SWIFT subprocess pages
        free_mb = torch.cuda.mem_get_info()[0] / 1024 ** 2
        logger.info("eval: GPU cache flushed — free VRAM %.0f MB", free_mb)


# Minimum free VRAM (GB) below which the evaluator falls back to CPU mode.
# Even 4-bit quantization needs ~800 MB for the weights plus KV cache and
# activations; below this threshold, accelerate will offload modules to CPU/disk
# which crashes with BitsAndBytes ("Some modules are dispatched on the CPU…").
# Forcing CPU mode up front avoids that crash — evaluation is slower but always
# completes, which is what matters for the user.
_MIN_VRAM_FOR_GPU_EVAL_GB = 1.2


def _load_model(base_model_id: str, adapter_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    _flush_gpu_cache()
    logger.info("eval: loading tokenizer %s", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    # ── Pre-flight check: is there enough VRAM to safely attempt GPU eval? ────
    use_gpu = False
    free_gb = 0.0
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
        if free_gb >= _MIN_VRAM_FOR_GPU_EVAL_GB:
            use_gpu = True
        else:
            logger.warning(
                "eval: insufficient VRAM (%.2f GB free, need >= %.2f GB) — "
                "falling back to CPU mode to guarantee evaluation completes",
                free_gb, _MIN_VRAM_FOR_GPU_EVAL_GB,
            )

    if use_gpu:
        load_kwargs: dict = {"device_map": "auto", "trust_remote_code": True}
        if free_gb < 1.5:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            logger.info("eval: 4-bit quantization (free VRAM %.2f GB)", free_gb)
        elif free_gb < 2.5:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("eval: 8-bit quantization (free VRAM %.2f GB)", free_gb)
        else:
            load_kwargs["torch_dtype"] = torch.float16
            logger.info("eval: float16 (free VRAM %.2f GB)", free_gb)
    else:
        # CPU fallback — always works but slower. BitsAndBytes quantization
        # requires CUDA kernels and cannot be used on CPU.
        load_kwargs = {
            "device_map":         "cpu",
            "torch_dtype":        torch.float32,
            "trust_remote_code":  True,
            "low_cpu_mem_usage":  True,
        }
        logger.info(
            "eval: CPU fallback (free VRAM %.2f GB < %.2f GB threshold) — "
            "evaluation will be slower but guaranteed to complete",
            free_gb, _MIN_VRAM_FOR_GPU_EVAL_GB,
        )

    logger.info("eval: loading base model %s", base_model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        # Defensive last-resort fallback: if GPU loading still fails despite
        # the pre-flight check, retry on CPU before giving up.
        if not use_gpu:
            raise
        logger.warning(
            "eval: GPU load failed (%s) — retrying on CPU as last resort",
            type(exc).__name__,
        )
        _flush_gpu_cache()
        load_kwargs = {
            "device_map":         "cpu",
            "torch_dtype":        torch.float32,
            "trust_remote_code":  True,
            "low_cpu_mem_usage":  True,
        }
        model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)

    logger.info("eval: loading LoRA adapter %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, instruction: str, max_new_tokens: int = 256) -> str:
    import torch
    prompt = _format_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False)
    return _strip_template(generated, "")


# ── Classification helpers ────────────────────────────────────────────────────

def _extract_label(text: str) -> str:
    text = text.strip().lower()
    first_line = text.split("\n")[0].strip()
    first_token = first_line.split()[0] if first_line.split() else text[:20]
    return first_token.strip(".,!?;:'\"()[]{}").strip()


def _compute_classification_metrics(references: list[str], predictions: list[str]) -> dict[str, float]:
    if not references:
        return {"accuracy": 0.0, "f1_macro": 0.0, "precision": 0.0, "recall": 0.0}

    labels = sorted(set(references) | set(predictions))
    n      = len(references)
    correct = sum(r == p for r, p in zip(references, predictions))
    accuracy = round(correct / n, 4)

    p_scores, r_scores, f_scores = [], [], []
    for label in labels:
        tp = sum(r == label and p == label for r, p in zip(references, predictions))
        fp = sum(r != label and p == label for r, p in zip(references, predictions))
        fn = sum(r == label and p != label for r, p in zip(references, predictions))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        p_scores.append(prec)
        r_scores.append(rec)
        f_scores.append(f1)

    return {
        "accuracy":  accuracy,
        "f1_macro":  round(sum(f_scores) / len(f_scores), 4),
        "precision": round(sum(p_scores) / len(p_scores), 4),
        "recall":    round(sum(r_scores) / len(r_scores), 4),
    }


# ── NER helpers ───────────────────────────────────────────────────────────────

def _parse_entities(text: str) -> set[tuple[str, str]]:
    entities: set[tuple[str, str]] = set()
    for part in text.split("|"):
        part = part.strip()
        if ":" in part:
            etype, _, evalue = part.partition(":")
            entities.add((etype.strip().upper(), evalue.strip().lower()))
    return entities


def _compute_ner_metrics(ref_entities_list: list[set], pred_entities_list: list[set]) -> dict[str, float]:
    tp = fp = fn = 0
    for ref, pred in zip(ref_entities_list, pred_entities_list):
        tp += len(ref & pred)
        fp += len(pred - ref)
        fn += len(ref - pred)
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
    return {"entity_f1": f1, "entity_precision": precision, "entity_recall": recall}


# ── Public entry point ────────────────────────────────────────────────────────

def evaluate(
    job_id: str,
    adapter_path: str,
    base_model_id: str,
    n_samples: int = 50,
    use_llm_judge: bool = True,
    judge_n: int = 10,
    seed: int = 42,
    eval_strategy: dict | None = None,
) -> dict:
    """Run full evaluation and write eval_report.json. Returns the report dict."""
    strategy = (eval_strategy or {}).get("strategy", "generation")

    # Read task early — needed to route task-specific metrics
    _task = "question-answering"
    from data.artifact_store import training_report_path, domain_path
    _rp = training_report_path(job_id)
    if _rp.exists():
        try:
            _task = json.loads(_rp.read_text(encoding="utf-8")).get("task", "question-answering")
        except Exception:
            pass

    # Detect language for BERTScore
    _lang = "fr"
    _dp = domain_path(job_id)
    if _dp.exists():
        try:
            _lang = json.loads(_dp.read_text(encoding="utf-8")).get("language", "fr")
        except Exception:
            pass

    # ── 1. Load test set ──────────────────────────────────────────────────────
    test_records = read_jsonl(test_path(job_id))
    if not test_records:
        return {"error": f"test.jsonl not found for job {job_id}"}

    random.seed(seed)
    sample = random.sample(test_records, min(n_samples, len(test_records)))
    logger.info("eval: strategy=%s task=%s %d test samples", strategy, _task, len(sample))

    # ── 2. Load model ─────────────────────────────────────────────────────────
    model, tokenizer = _load_model(base_model_id, adapter_path)

    # ── 3. Generate responses ─────────────────────────────────────────────────
    results: list[dict] = []
    for rec in sample:
        instruction = rec.get("instruction", "")
        reference   = rec.get("output", "")
        try:
            generated = _generate(model, tokenizer, instruction)
        except Exception as exc:
            logger.warning("generation failed: %s", exc)
            generated = ""
        results.append({
            "instruction": instruction,
            "reference":   reference,
            "generated":   generated,
        })

    # Free model from GPU before computing BERTScore (also needs memory)
    del model, tokenizer
    _flush_gpu_cache()

    # ── 4. Compute metrics according to strategy ──────────────────────────────
    metrics: dict[str, Any]

    if strategy == "classification":
        references  = [_extract_label(r["reference"]) for r in results]
        predictions = [_extract_label(r["generated"]) for r in results]
        metrics = _compute_classification_metrics(references, predictions)
        metrics["n_test"] = len(results)
        for r, ref, pred in zip(results, references, predictions):
            r["reference_label"] = ref
            r["predicted_label"] = pred
        logger.info(
            "eval_classification accuracy=%.3f f1_macro=%.3f",
            metrics["accuracy"], metrics["f1_macro"],
        )

    elif strategy == "ner":
        ref_entities_list  = [_parse_entities(r["reference"]) for r in results]
        pred_entities_list = [_parse_entities(r["generated"]) for r in results]
        metrics = _compute_ner_metrics(ref_entities_list, pred_entities_list)
        metrics["n_test"] = len(results)
        logger.info(
            "eval_ner entity_f1=%.3f precision=%.3f recall=%.3f",
            metrics["entity_f1"], metrics["entity_precision"], metrics["entity_recall"],
        )

    else:
        # ── Generation: ROUGE + BLEU + F1-Token + EM (always) ────────────────
        def _mean(key: str) -> float:
            vals = [r[key] for r in results if key in r]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        for r in results:
            rouge = _rouge_scores(r["reference"], r["generated"])
            bleu  = _bleu4(r["reference"], r["generated"])
            f1t   = _f1_token(r["reference"], r["generated"])
            em    = _exact_match(r["reference"], r["generated"])
            r.update({**rouge, "bleu4": bleu, "f1_token": f1t, "exact_match": em})

        metrics = {
            "rouge1":      _mean("rouge1"),
            "rouge2":      _mean("rouge2"),
            "rougeL":      _mean("rougeL"),
            "bleu4":       _mean("bleu4"),
            "f1_token":    _mean("f1_token"),
            "exact_match": _mean("exact_match"),
            "n_test":      len(results),
        }
        logger.info(
            "eval_generation rouge1=%.3f rougeL=%.3f bleu4=%.3f f1_token=%.3f em=%.3f",
            metrics["rouge1"], metrics["rougeL"], metrics["bleu4"],
            metrics["f1_token"], metrics["exact_match"],
        )

        # ── BERTScore (optional) ──────────────────────────────────────────────
        refs = [r["reference"] for r in results]
        hyps = [r["generated"] for r in results]
        bert = _compute_bertscore(refs, hyps, lang=_lang)
        metrics.update(bert)

        # ── CodeBLEU for code-generation (optional) ───────────────────────────
        if _task == "code-generation":
            cb = _compute_codebleu(refs, hyps)
            metrics.update(cb)

    metrics["eval_strategy"] = strategy
    metrics["task"] = _task

    # ── 5. LLM client (shared for G-Eval, faithfulness, and metric thresholds) ──
    _llm_client = None
    try:
        from agents.llm_client import get_llm_client
        _llm_client = get_llm_client()
    except Exception as exc:
        logger.warning("llm_client_unavailable: %s", exc)

    # ── 5b. Extended LLM-based metrics (generation tasks only) ────────────────
    # Skip G-Eval + Faithfulness when:
    #   - env var EVAL_SKIP_LLM_JUDGE=1 is set (manual opt-out), OR
    #   - we are on Windows with a small GPU (< 6 GB total) — these LLM calls
    #     have historically hung indefinitely on this configuration with no
    #     timeout. BERTScore + ROUGE + BLEU + F1 are sufficient as quality
    #     signals; the primary metric (bertscore_f1) does not depend on judges.
    _skip_judge = os.environ.get("EVAL_SKIP_LLM_JUDGE", "").lower() in ("1", "true", "yes")
    if not _skip_judge:
        try:
            import platform, torch as _torch
            if platform.system() == "Windows" and _torch.cuda.is_available():
                _total_gb = _torch.cuda.get_device_properties(0).total_memory / 1e9
                if _total_gb < 6.0:
                    _skip_judge = True
                    logger.info(
                        "llm_judge auto-skipped: Windows + small GPU (%.1f GB) — relying on lexical + BERTScore",
                        _total_gb,
                    )
        except Exception:
            pass

    if use_llm_judge and not _skip_judge and results and strategy == "generation" and _llm_client:
        try:
            g_eval_scores = _g_eval(results, _task, _llm_client, n=judge_n)
            metrics.update(g_eval_scores)

            if _task in ("summarization", "question-answering", "chatbot"):
                faith = _faithfulness_score(results, _llm_client, n=judge_n)
                metrics.update(faith)

        except Exception as exc:
            logger.warning("llm_judge_skipped: %s", exc)
    elif _skip_judge:
        logger.info("llm_judge skipped (EVAL_SKIP_LLM_JUDGE or Windows+small-GPU policy)")

    # ── 6. Domain-aware quality summary ──────────────────────────────────────
    _domain = "general"
    if _dp.exists():
        try:
            _domain = json.loads(_dp.read_text(encoding="utf-8")).get("domain", "general")
        except Exception:
            pass

    from services.metric_thresholds import get_thresholds

    good_t       = (eval_strategy or {}).get("good_threshold")
    acceptable_t = (eval_strategy or {}).get("acceptable_threshold")
    primary      = (eval_strategy or {}).get("primary_metric", "rouge1")
    primary_val  = metrics.get(primary, 0.0)
    strat_rationale = (eval_strategy or {}).get("rationale", "")

    if good_t is None:
        thresholds   = get_thresholds(task=_task, domain=_domain, llm_client=_llm_client)
        good_t       = thresholds.rouge1_good
        acceptable_t = thresholds.rouge1_acceptable
        metrics["threshold_rouge1_good"]       = thresholds.rouge1_good
        metrics["threshold_rouge1_acceptable"] = thresholds.rouge1_acceptable
        metrics["threshold_rouge2_good"]       = thresholds.rouge2_good
        metrics["threshold_rougeL_good"]       = thresholds.rougeL_good
        metrics["threshold_bleu4_good"]        = thresholds.bleu4_good

    if primary_val >= good_t:
        tier = "good"
    elif primary_val >= acceptable_t:
        tier = "acceptable"
    else:
        tier = "poor"

    suspicious_bound = good_t * 1.9
    if primary_val > suspicious_bound:
        quality = (
            f"⚠ Score anormal — {primary}={primary_val:.3f} dépasse le seuil de suspicion "
            f"{suspicious_bound:.2f} (risque de data leakage). {strat_rationale}"
        )
    elif tier == "good":
        quality = (
            f"Bonne qualité — {primary}={primary_val:.3f} atteint l'objectif {good_t} "
            f"(minimum requis : {acceptable_t}). {strat_rationale}"
        )
    elif tier == "acceptable":
        quality = (
            f"Qualité acceptable — {primary}={primary_val:.3f} dépasse le minimum requis "
            f"{acceptable_t} (objectif optimal : {good_t}). {strat_rationale}"
        )
    else:
        quality = (
            f"Qualité insuffisante — {primary}={primary_val:.3f} inférieur au minimum requis "
            f"{acceptable_t} (score à améliorer). {strat_rationale}"
        )

    metrics["quality_summary"]       = quality
    metrics["quality_tier"]          = tier
    metrics["primary_metric"]        = primary
    metrics["good_threshold"]        = good_t
    metrics["acceptable_threshold"]  = acceptable_t

    # ── 7. Write report ───────────────────────────────────────────────────────
    report = {
        "job_id":        job_id,
        "adapter_path":  adapter_path,
        "base_model_id": base_model_id,
        "eval_strategy": strategy,
        "task":          _task,
        "metrics":       metrics,
        "samples":       results[:10],
    }
    write_json(eval_report_path(job_id), report)
    logger.info(
        "eval_done job=%s strategy=%s task=%s %s=%.3f tier=%s",
        job_id, strategy, _task, primary, primary_val, tier,
    )
    return report
