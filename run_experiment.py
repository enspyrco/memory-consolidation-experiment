#!/usr/bin/env python3
"""
Memory Consolidation Gating Experiment
=======================================
Tests whether surprise-gated memory writes outperform importance-gated writes
and write-everything baselines. Wraps D-MEM's existing infrastructure.

Usage:
  # Smoke test (1 sample, fast)
  python run_experiment.py --ratio 0.1

  # Full run (all 10 LoCoMo samples)
  python run_experiment.py --ratio 1.0

  # With noise injection
  python run_experiment.py --ratio 1.0 --noise 0.75

  # Single condition only
  python run_experiment.py --conditions surprise_gated
"""

import os
import sys
import json
import time
import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add D-MEM code to path
DMEM_CODE = str(Path(__file__).parent / "vendor" / "dmem" / "code")
sys.path.insert(0, DMEM_CODE)

# Import and auto-patch LLMController with claude-code backend BEFORE importing D-MEM
import claude_code_backend  # noqa: F401 — patches LLMController on import

from dmem import DopamineMemorySystem
from load_dataset import load_locomo_dataset
from utils import calculate_metrics, aggregate_metrics
from noise_generator import inject_noise
from memory_layer import token_tracker

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Experimental conditions
# ---------------------------------------------------------------------------

CONDITIONS = {
    "write_everything": {
        "description": "No gate. All turns fully processed (A-MEM equivalent).",
        "rpe_mode": "rpe",
        "theta_low": 0.0,
        "theta_high": 0.0,
        "alpha": 0.6,
    },
    "importance_gated": {
        "description": "LLM classifies utility (TRANSIENT/SHORT_TERM/PERSISTENT). Threshold filters.",
        "rpe_mode": "utility_only",
        "theta_low": 0.3,
        "theta_high": 0.7,
        "alpha": 0.0,
    },
    "surprise_gated": {
        "description": "Embedding cosine similarity — low similarity = high surprise. Zero LLM cost.",
        "rpe_mode": "surprise_only",
        "theta_low": 0.3,
        "theta_high": 0.7,
        "alpha": 1.0,
    },
    "combined_rpe": {
        "description": "D-MEM's best: utility * (surprise + 0.4). Multiplicative gating.",
        "rpe_mode": "rpe_v3",
        "theta_low": 0.3,
        "theta_high": 0.7,
        "alpha": 0.6,
    },
}


# ---------------------------------------------------------------------------
# Agent wrapper — minimal, delegates everything to D-MEM
# ---------------------------------------------------------------------------

class GatingAgent:
    """Thin wrapper around DopamineMemorySystem for our experiment."""

    def __init__(self, condition_cfg: dict, model: str, backend: str):
        from memory_layer import LLMController
        self.memory_system = DopamineMemorySystem(
            theta_low=condition_cfg["theta_low"],
            theta_high=condition_cfg["theta_high"],
            alpha=condition_cfg["alpha"],
            rpe_mode=condition_cfg["rpe_mode"],
            warmup_turns=10,
            model_name="all-MiniLM-L6-v2",
            llm_backend=backend,
            llm_model=model,
        )
        self.retriever_llm = LLMController(
            backend=backend, model=model, api_key=None
        )
        self.retrieve_k = 10

    def add_memory(self, content: str, time: str = None):
        self.memory_system.add_note(content, time=time)

    def answer_question(self, question: str, category: int, answer: str):
        """RAG-style QA: retrieve relevant memories, then answer with LLM."""
        # Retrieve
        memories = self.memory_system.find_related_memories_raw(
            question, k=self.retrieve_k
        )
        # Refine retrieval with LLM
        refine_prompt = (
            f"Given these memories and a question, select the most relevant parts.\n\n"
            f"Memories:\n{memories}\n\nQuestion: {question}\n\n"
            f"Return JSON: {{\"relevant_parts\": \"...\"}}"
        )
        try:
            refined = self.retriever_llm.llm.get_completion(
                refine_prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {"relevant_parts": {"type": "string"}},
                            "required": ["relevant_parts"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
            )
            context = json.loads(refined).get("relevant_parts", memories)
        except Exception:
            context = memories

        # Answer
        answer_prompt = (
            f"Based on the following conversation context, answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            f"Answer concisely. Return JSON: {{\"answer\": \"...\"}}"
        )
        try:
            raw = self.retriever_llm.llm.get_completion(
                answer_prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
            )
            return raw, question, context
        except Exception:
            return json.dumps({"answer": ""}), question, context


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def flatten_turns(sample) -> list:
    """Flatten a LoCoMo sample's sessions into a list of dicts."""
    turns = []
    for sess_id, sess in sample.conversation.sessions.items():
        for turn in sess.turns:
            turns.append({
                "text": f"Speaker {turn.speaker} says : {turn.text}",
                "time": sess.date_time,
                "session_id": sess_id,
            })
    return turns


def evaluate_qa(agent, qa_list, logger) -> tuple:
    """Run QA evaluation, return (results, aggregate_metrics)."""
    results = []
    all_metrics = []
    all_categories = []
    for qa in qa_list:
        if int(qa.category) not in [1, 2, 3, 4, 5]:
            continue
        prediction, _, raw_ctx = agent.answer_question(
            qa.question, qa.category, qa.final_answer
        )
        try:
            prediction = json.loads(prediction)["answer"]
        except Exception:
            pass
        metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {}
        all_metrics.append(metrics)
        all_categories.append(qa.category)
        results.append({
            "question": qa.question,
            "prediction": prediction,
            "reference": qa.final_answer,
            "category": qa.category,
            "metrics": metrics,
        })
        logger.info(
            f"  Q: {qa.question[:80]}... | "
            f"pred: {str(prediction)[:60]} | ref: {str(qa.final_answer)[:60]}"
        )
    agg = aggregate_metrics(all_metrics, all_categories) if all_metrics else {}
    return results, agg


def run_condition(
    condition_name: str,
    condition_cfg: dict,
    samples: list,
    model: str,
    backend: str,
    noise_ratio: float,
    logger: logging.Logger,
) -> dict:
    """Run a single experimental condition across all samples."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Condition: {condition_name}")
    logger.info(f"Config: {condition_cfg}")
    logger.info(f"Samples: {len(samples)}, Noise: {noise_ratio}")
    logger.info(f"{'='*60}")

    all_results = []
    all_metrics_list = []
    all_cats = []
    total_tokens = 0
    routing_totals = defaultdict(int)

    for sidx, sample in enumerate(tqdm(samples, desc=condition_name)):
        token_tracker.reset()
        agent = GatingAgent(condition_cfg, model, backend)

        # Ingest conversation turns
        turns = flatten_turns(sample)
        turn_texts = [t["text"] for t in turns]
        turn_times = [t["time"] for t in turns]

        if noise_ratio > 0:
            noisy_texts, _ = inject_noise(turn_texts, noise_ratio, seed=42 + sidx)
            while len(turn_times) < len(noisy_texts):
                turn_times.append(turn_times[-1] if turn_times else "")
            turn_texts = noisy_texts

        t0 = time.time()
        for i, text in enumerate(turn_texts):
            t_time = turn_times[i] if i < len(turn_times) else None
            agent.add_memory(text, time=t_time)
        ingest_time = time.time() - t0

        # Evaluate QA
        t0 = time.time()
        res, _ = evaluate_qa(agent, sample.qa, logger)
        eval_time = time.time() - t0

        metrics_list = [r["metrics"] for r in res if r["metrics"]]
        cats_list = [r["category"] for r in res if r["metrics"]]
        all_results.extend(res)
        all_metrics_list.extend(metrics_list)
        all_cats.extend(cats_list)

        # Routing stats
        routing = agent.memory_system.get_routing_summary()
        for k in ("skip", "construct", "evolve"):
            routing_totals[k] += routing[k]
        routing_totals["total_turns"] += routing["total_turns"]

        tokens = token_tracker.total_tokens
        total_tokens += tokens

        logger.info(
            f"[{sidx+1}/{len(samples)}] "
            f"ingest={ingest_time:.1f}s eval={eval_time:.1f}s "
            f"tokens={tokens} routing={routing['skip_pct']:.0f}% skip"
        )

    overall_agg = aggregate_metrics(all_metrics_list, all_cats) if all_metrics_list else {}
    total = routing_totals["total_turns"] or 1

    return {
        "condition": condition_name,
        "config": {k: v for k, v in condition_cfg.items() if k != "description"},
        "description": condition_cfg["description"],
        "aggregate": overall_agg,
        "total_tokens": total_tokens,
        "routing": {
            "total_turns": routing_totals["total_turns"],
            "skip": routing_totals["skip"],
            "construct": routing_totals["construct"],
            "evolve": routing_totals["evolve"],
            "skip_pct": routing_totals["skip"] / total * 100,
            "construct_pct": routing_totals["construct"] / total * 100,
            "evolve_pct": routing_totals["evolve"] / total * 100,
        },
        "num_samples": len(samples),
        "num_qa": len(all_results),
        "results": all_results,
    }


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _run_condition_worker(cond_name, cond_cfg, samples, model, backend, noise_ratio, output_dir):
    """Run a single condition in a subprocess. Re-imports to avoid shared state."""
    # Each subprocess needs fresh imports (SentenceTransformer, token_tracker, etc.)
    import importlib
    import claude_code_backend  # noqa: re-patch in subprocess
    importlib.reload(claude_code_backend)

    log_file = os.path.join(output_dir, f"{cond_name}.log")
    logger = logging.getLogger(cond_name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    result = run_condition(cond_name, cond_cfg, samples, model, backend, noise_ratio, logger)

    # Save results from within the worker
    cond_path = os.path.join(output_dir, f"{cond_name}.json")
    with open(cond_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved {cond_name} results to {cond_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Memory Consolidation Gating Experiment")
    parser.add_argument("--ratio", type=float, default=0.1,
                        help="Fraction of LoCoMo samples to use (default: 0.1 = 1 sample)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Noise injection ratio (default: 0.0, recommended: 0.75)")
    parser.add_argument("--conditions", nargs="+", default=["all"],
                        help="Which conditions to run (default: all)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="LLM model for utility scoring and QA (default: claude-haiku-4-5-20251001)")
    parser.add_argument("--backend", default="claude-code",
                        help="LLM backend (default: claude-code, uses Max plan)")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Run N conditions in parallel (default: 1 = sequential)")
    args = parser.parse_args()

    # Resolve conditions
    if "all" in args.conditions:
        conditions_to_run = list(CONDITIONS.keys())
    else:
        conditions_to_run = args.conditions
        for c in conditions_to_run:
            if c not in CONDITIONS:
                print(f"Unknown condition: {c}. Available: {list(CONDITIONS.keys())}")
                sys.exit(1)

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger("experiment")

    # Load dataset
    dataset_path = Path(__file__).parent / "vendor" / "dmem" / "code" / "data" / "locomo10.json"
    logger.info(f"Loading LoCoMo dataset from {dataset_path}")
    samples = load_locomo_dataset(str(dataset_path))
    if args.ratio < 1.0:
        samples = samples[: max(1, int(len(samples) * args.ratio))]
    logger.info(f"Using {len(samples)} of {10} samples (ratio={args.ratio})")

    # Run conditions
    all_outputs = {}
    start_time = datetime.now()

    if args.parallel > 1 and len(conditions_to_run) > 1:
        logger.info(f"Running {len(conditions_to_run)} conditions in parallel (max {args.parallel} workers)")
        # Each condition runs as a separate subprocess to avoid shared state
        with ProcessPoolExecutor(max_workers=min(args.parallel, len(conditions_to_run))) as executor:
            futures = {}
            for cond_name in conditions_to_run:
                future = executor.submit(
                    _run_condition_worker,
                    cond_name, CONDITIONS[cond_name], samples,
                    args.model, args.backend, args.noise, args.output_dir,
                )
                futures[future] = cond_name

            for future in as_completed(futures):
                cond_name = futures[future]
                try:
                    result = future.result()
                    all_outputs[cond_name] = result
                    logger.info(f"Completed {cond_name}: F1={result['aggregate'].get('overall', {}).get('f1', {}).get('mean', 0):.3f}")
                except Exception as e:
                    logger.error(f"Condition {cond_name} failed: {e}")
    else:
        for cond_name in conditions_to_run:
            cond_cfg = CONDITIONS[cond_name]
            result = run_condition(
                cond_name, cond_cfg, samples,
                args.model, args.backend, args.noise, logger,
            )
            all_outputs[cond_name] = result

            # Save per-condition results incrementally
            cond_path = os.path.join(args.output_dir, f"{cond_name}.json")
            with open(cond_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Saved {cond_name} results to {cond_path}")

    end_time = datetime.now()

    # Save combined results
    combined = {
        "experiment": "memory_consolidation_gating",
        "timestamp": start_time.isoformat(),
        "runtime_seconds": (end_time - start_time).total_seconds(),
        "config": {
            "model": args.model,
            "backend": args.backend,
            "ratio": args.ratio,
            "noise": args.noise,
            "num_samples": len(samples),
        },
        "conditions": {
            name: {
                "description": out["description"],
                "f1": out["aggregate"].get("overall", {}).get("f1", {}).get("mean", 0),
                "multi_hop_f1": out["aggregate"].get("category_1", {}).get("f1", {}).get("mean", 0),
                "total_tokens": out["total_tokens"],
                "skip_pct": out["routing"]["skip_pct"],
            }
            for name, out in all_outputs.items()
        },
    }
    combined_path = os.path.join(args.output_dir, "combined.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Condition':<22} {'F1':>6} {'Multi-hop F1':>13} {'Tokens':>10} {'Skip %':>8}")
    print("-" * 80)
    for name, out in all_outputs.items():
        overall = out["aggregate"].get("overall", {})
        cat1 = out["aggregate"].get("category_1", {})
        f1 = overall.get("f1", {}).get("mean", 0)
        mh_f1 = cat1.get("f1", {}).get("mean", 0)
        tokens = out["total_tokens"]
        skip = out["routing"]["skip_pct"]
        print(f"{name:<22} {f1:>6.3f} {mh_f1:>13.3f} {tokens:>10,} {skip:>7.1f}%")
    print("=" * 80)
    print(f"\nResults saved to {args.output_dir}/")
    print(f"Runtime: {(end_time - start_time).total_seconds():.0f}s")


if __name__ == "__main__":
    # Use 'spawn' to avoid shared ML model state between workers
    multiprocessing.set_start_method("spawn", force=True)
    main()
