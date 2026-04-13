# Memory Consolidation Gating Experiment

Testing whether surprise-gated memory writes outperform importance-gated writes and write-everything baselines for AI agent memory.

## Hypothesis

Gating memory writes by **surprise** (prediction error — "was this unexpected?") produces better downstream task performance than gating by **importance** ("is this useful?") or no gating ("write everything"). This is backed by convergent findings across neuroscience, information theory, cognitive psychology, organizational learning, and continual ML. Full research synthesis: [CONSOLIDATION_SCIENCE.md](../../orgs/enspyrco/engram/docs/CONSOLIDATION_SCIENCE.md).

## Experimental Design

Based on D-MEM's ablation methodology (arXiv:2603.14597). Uses the LoCoMo benchmark (Maharana et al., ACL 2024) — 10 multi-session conversations with 1,986 QA pairs.

### Conditions

| # | Strategy | D-MEM mode | Description |
|---|----------|-----------|-------------|
| 1 | Write-everything | (baseline) | No gate. All conversation turns stored. A-MEM baseline. |
| 2 | Importance-gated | `utility_only` | LLM classifies utility (TRANSIENT/SHORT_TERM/PERSISTENT). Threshold filters. |
| 3 | Surprise-gated | `surprise_only` | Embedding cosine similarity — low similarity to existing memory = high surprise. Zero LLM cost. |
| 4 | Combined RPE | `rpe_v3` | D-MEM's best: `utility * (surprise + 0.4)`. Multiplicative gating. |

### Metrics

- **F1 Score** (primary) — token-level, SQuAD-style, by QA category (single-hop, multi-hop, temporal, open-domain, adversarial)
- **Token consumption** — total API tokens across full pipeline
- **Skip rate** — % of conversation turns filtered out by gate
- **Multi-hop accuracy** — F1 specifically on category 1 (hardest, requires cross-session reasoning)
- **Noise robustness** — all conditions re-run with 75% noise injection (LoCoMo-Noise)

### Noise Injection

D-MEM's `noise_generator.py` injects three noise types at configurable ratios:
- Filler (40%): "hmm", "ok", "yeah"
- Off-topic (30%): "The weather is nice today"
- Repetition (30%): duplicate of random earlier turn

Tested at 75% noise ratio (for every 4 real turns, 3 noise turns added).

## Quick Start

```bash
# Setup
cd ~/git/experiments/memory-consolidation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Quick smoke test (10% of data, ~200 API calls, ~5 min)
python run_experiment.py --ratio 0.1 --conditions all

# Full run (all data, ~2000 API calls per condition, ~45 min)
python run_experiment.py --ratio 1.0 --conditions all

# With noise injection
python run_experiment.py --ratio 1.0 --conditions all --noise 0.75

# Analyze results
python analyze_results.py results/
```

## Project Structure

```
vendor/
  dmem/          # D-MEM codebase (git clone)
  locomo/        # LoCoMo dataset (git clone)
src/
  experiment.py  # Main experiment runner
  agents.py      # Agent implementations wrapping D-MEM's memory systems
  evaluate.py    # QA evaluation and F1 scoring
  report.py      # Results analysis and markdown report generation
run_experiment.py  # CLI entry point
analyze_results.py # Post-hoc analysis
requirements.txt
results/           # Output directory (gitignored)
```

## Dependencies

Extends D-MEM's requirements. Key additions:
- `anthropic` — for Claude API (optional, to test Claude as the gating/eval LLM)
- All D-MEM deps: `sentence-transformers`, `scikit-learn`, `numpy`, `nltk`, `rank-bm25`, etc.

## Status

- [x] Research synthesis complete (CONSOLIDATION_SCIENCE.md)
- [x] D-MEM and LoCoMo repos cloned
- [x] Experimental design documented
- [ ] Python environment and deps
- [ ] Experiment runner scaffolding
- [ ] Smoke test (--ratio 0.1)
- [ ] Full clean run
- [ ] Noise injection run (0.75)
- [ ] Results analysis and comparison table
- [ ] Blog post updated with numbers

## Context

This experiment supports a blog post: "Stop Automating Your AI's Memory. Talk to It Instead." (~/git/consolidation-blog-post.md). The surprise-gate ablation is the most self-contained testable claim. Results go directly into the post.
