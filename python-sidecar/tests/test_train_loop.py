"""Tests for the self-training loop + eval-and-promote gate.

The gate is safety-critical, so these tests inject a deterministic evaluator and
a controllable engine to prove: promotion when better, rejection when worse,
the single-active-adapter invariant, rollback, and correct DB state.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from store import Store
from feedback import FeedbackStore
from engine import make_engine, TrainConfig
from eval_gate import EvalGate, GateDecision, HeldOutItem
from trainer import Trainer, AdapterRegistry


# ---- a deterministic evaluator we fully control ---------------------------
class ScriptedEvaluator:
    """Returns pre-set scores keyed by adapter identity.

    score(engine, adapter_path, held_out): looks up adapter_path in self.scores;
    None (base model / no adapter) -> self.base_score.
    """
    def __init__(self, scores: dict, base_score: float = 0.5):
        self.scores = scores
        self.base_score = base_score
        self.calls = []

    def score(self, engine, adapter_path, held_out):
        self.calls.append(adapter_path)
        if adapter_path is None:
            return self.base_score
        # match by the version dir name (last path component)
        key = os.path.basename(adapter_path.rstrip("/"))
        return self.scores.get(key, self.base_score)


def seed_examples(fb, n=10, tag="a"):
    """Add n unique examples. `tag` keeps content distinct across rounds so the
    dedup layer doesn't drop a second batch."""
    for i in range(n):
        fb.capture(f"{tag} question number {i}",
                   f"the good answer {tag}{i}", "thumbs_up")


def make_trainer(evaluator, tmp, min_improvement=0.0, seed=0):
    store = Store(":memory:")
    fb = FeedbackStore(store, train_threshold=5)
    seed_examples(fb, 10)
    eng = make_engine("fake", os.path.join(tmp, "models"),
                      os.path.join(tmp, "adapters"), dim=128)
    eng.load("gemma-4-e4b")
    tr = Trainer(store, eng, fb, adapters_dir=os.path.join(tmp, "adapters"),
                 config=TrainConfig(iters=20), evaluator=evaluator,
                 min_improvement=min_improvement, seed=seed)
    return tr, store, fb, eng


# ---- gate unit tests ------------------------------------------------------
def test_gate_promotes_when_better():
    ev = ScriptedEvaluator({"v0001": 0.9}, base_score=0.5)
    gate = EvalGate(ev, min_improvement=0.0)
    d = gate.evaluate(engine=None, candidate_adapter="/x/v0001",
                      baseline_adapter=None, held_out=[HeldOutItem("q", "a")])
    assert d.promote is True
    assert abs(d.improvement - 0.4) < 1e-9


def test_gate_rejects_when_worse():
    ev = ScriptedEvaluator({"v0002": 0.3}, base_score=0.5)
    gate = EvalGate(ev, min_improvement=0.0)
    d = gate.evaluate(engine=None, candidate_adapter="/x/v0002",
                      baseline_adapter=None, held_out=[HeldOutItem("q", "a")])
    assert d.promote is False
    assert d.improvement < 0


def test_gate_respects_min_improvement():
    # candidate is better by 0.05 but threshold is 0.1 -> reject
    ev = ScriptedEvaluator({"v0003": 0.55}, base_score=0.5)
    gate = EvalGate(ev, min_improvement=0.1)
    d = gate.evaluate(engine=None, candidate_adapter="/x/v0003",
                      baseline_adapter=None, held_out=[HeldOutItem("q", "a")])
    assert d.promote is False


# ---- full-loop tests ------------------------------------------------------
def test_loop_promotes_first_adapter():
    with tempfile.TemporaryDirectory() as tmp:
        # first adapter (v0001) beats the base model (0.5)
        ev = ScriptedEvaluator({"v0001": 0.8}, base_score=0.5)
        tr, store, fb, eng = make_trainer(ev, tmp)
        summary = tr.run_once()
        assert summary["status"] == "promoted", summary
        active = tr.registry.active()
        assert active is not None and active["id"] == "v0001"
        assert eng.current_adapter and eng.current_adapter.endswith("v0001")
        # examples consumed
        assert fb.unused_count() == 0
        # run recorded as promoted
        runs = tr.list_runs()
        assert runs[0]["status"] == "promoted"


def test_loop_rejects_worse_adapter_and_keeps_baseline():
    with tempfile.TemporaryDirectory() as tmp:
        # v0001 promotes (0.8), then v0002 is worse (0.6 < 0.8) -> reject
        ev = ScriptedEvaluator({"v0001": 0.8, "v0002": 0.6}, base_score=0.5)
        tr, store, fb, eng = make_trainer(ev, tmp)
        s1 = tr.run_once()
        assert s1["status"] == "promoted"
        # add more examples so a second round can run
        seed_examples(fb, 6, tag="b")
        s2 = tr.run_once()
        assert s2["status"] == "rejected", s2
        # active adapter is STILL v0001 (the good one)
        active = tr.registry.active()
        assert active["id"] == "v0001"
        # engine restored to the baseline adapter, not the rejected one
        assert eng.current_adapter.endswith("v0001")
        # both adapters exist in the registry (audit trail)
        allad = tr.registry.list_all()
        ids = {a["id"] for a in allad}
        assert ids == {"v0001", "v0002"}
        rejected = [a for a in allad if a["id"] == "v0002"][0]
        assert rejected["promoted"] == 0 and rejected["is_active"] == 0


def test_single_active_invariant():
    with tempfile.TemporaryDirectory() as tmp:
        ev = ScriptedEvaluator({"v0001": 0.8, "v0002": 0.95}, base_score=0.5)
        tr, store, fb, eng = make_trainer(ev, tmp)
        tr.run_once()               # v0001 active
        seed_examples(fb, 6, tag="b")
        tr.run_once()               # v0002 better -> active
        n_active = store.query("SELECT COUNT(*) AS n FROM adapters WHERE is_active=1")[0]["n"]
        assert n_active == 1
        assert tr.registry.active()["id"] == "v0002"


def test_rollback():
    with tempfile.TemporaryDirectory() as tmp:
        ev = ScriptedEvaluator({"v0001": 0.8, "v0002": 0.95}, base_score=0.5)
        tr, store, fb, eng = make_trainer(ev, tmp)
        tr.run_once()
        seed_examples(fb, 6, tag="b")
        tr.run_once()
        assert tr.registry.active()["id"] == "v0002"
        # roll back to v0001
        tr.registry.rollback("v0001")
        assert tr.registry.active()["id"] == "v0001"
        n_active = store.query("SELECT COUNT(*) AS n FROM adapters WHERE is_active=1")[0]["n"]
        assert n_active == 1


def test_loop_skips_with_too_few_examples():
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(":memory:")
        fb = FeedbackStore(store, train_threshold=5)
        fb.capture("only one", "example", "thumbs_up")
        eng = make_engine("fake", tmp + "/m", tmp + "/a", dim=64)
        eng.load("gemma-4-e4b")
        tr = Trainer(store, eng, fb, adapters_dir=tmp + "/a",
                     evaluator=ScriptedEvaluator({}))
        s = tr.run_once()
        assert s["status"] == "skipped"


def test_train_failure_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        ev = ScriptedEvaluator({})
        tr, store, fb, eng = make_trainer(ev, tmp)
        # force the engine to fail training
        from engine.base import TrainResult
        eng.train_lora = lambda *a, **k: TrainResult(ok=False, error="boom")
        s = tr.run_once()
        assert s["status"] == "failed"
        assert "boom" in s["error"]
        runs = tr.list_runs()
        assert runs[0]["status"] == "failed"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
