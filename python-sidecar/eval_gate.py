"""Eval-and-promote gate — the safety mechanism of self-training.

A newly trained adapter is promoted ONLY if it scores at least
``min_improvement`` above the current active adapter (the baseline) on a
held-out set. Otherwise it is discarded. Without this gate a self-training loop
drifts and degrades; with it, the model can only stay the same or get better.

The scorer is pluggable:
  * ``EmbeddingSimilarityEvaluator`` (default, production): generate on each
    held-out instruction with the candidate adapter, embed the output and the
    preferred answer, average their cosine similarity. Real, engine-agnostic.
  * Any object implementing ``score(engine, adapter_path, held_out) -> float``
    can be injected — tests use a deterministic one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class HeldOutItem:
    instruction: str
    preferred_output: str
    context: Optional[str] = None


class Evaluator(Protocol):
    def score(self, engine, adapter_path: Optional[str],
              held_out: list[HeldOutItem]) -> float: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class EmbeddingSimilarityEvaluator:
    """Score = mean cosine similarity between generated and preferred outputs.

    Higher is better, in roughly [-1, 1] (usually [0, 1] for text embeddings).
    Engine-agnostic: works with MLX, llama.cpp, or the fake engine.
    """

    def __init__(self, max_tokens: int = 256, temperature: float = 0.0):
        self.max_tokens = max_tokens
        self.temperature = temperature

    def score(self, engine, adapter_path, held_out):
        if not held_out:
            return 0.0
        engine.set_adapter(adapter_path)
        gens, refs = [], []
        for item in held_out:
            msgs = []
            if item.context:
                msgs.append({"role": "system", "content": item.context})
            msgs.append({"role": "user", "content": item.instruction})
            out = "".join(engine.generate(
                msgs, max_tokens=self.max_tokens,
                temperature=self.temperature, stream=False))
            gens.append(out)
            refs.append(item.preferred_output)
        gvecs = engine.embed(gens)
        rvecs = engine.embed(refs)
        sims = [_cosine(g, r) for g, r in zip(gvecs, rvecs)]
        return sum(sims) / len(sims)


@dataclass
class GateDecision:
    promote: bool
    candidate_score: float
    baseline_score: float
    improvement: float
    reason: str


class EvalGate:
    """Decides whether a candidate adapter beats the baseline."""

    def __init__(self, evaluator: Evaluator, min_improvement: float = 0.0):
        self.evaluator = evaluator
        self.min_improvement = min_improvement

    def evaluate(self, engine, candidate_adapter: str,
                 baseline_adapter: Optional[str],
                 held_out: list[HeldOutItem]) -> GateDecision:
        """Score candidate and baseline on the same held-out set, then decide."""
        candidate_score = self.evaluator.score(engine, candidate_adapter, held_out)
        # Baseline is the current active adapter (or None = base model).
        baseline_score = self.evaluator.score(engine, baseline_adapter, held_out)
        improvement = candidate_score - baseline_score
        promote = improvement >= self.min_improvement
        reason = (
            f"candidate {candidate_score:.4f} vs baseline {baseline_score:.4f} "
            f"(Δ={improvement:+.4f}, threshold={self.min_improvement:+.4f}) "
            f"→ {'PROMOTE' if promote else 'REJECT'}"
        )
        return GateDecision(promote, candidate_score, baseline_score,
                            improvement, reason)
