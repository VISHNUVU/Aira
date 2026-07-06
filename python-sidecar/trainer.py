"""Self-training loop — the flagship module.

Orchestrates one round of the self-improvement cycle:

  1. Build a train / held-out split from the pending examples.
  2. Launch a QLoRA fine-tune via the engine (real on Mac; mocked in tests).
  3. Evaluate the new adapter against the held-out set with the eval gate.
  4. Promote ONLY if it beats the current active adapter; else discard.
  5. Record the run + decision in the adapter registry (training_runs +
     adapters tables), keeping every version for rollback.

Runs in a background thread so the UI stays responsive; the DB rows let the
Training dashboard show live status.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import threading
import uuid
from dataclasses import asdict
from typing import Optional

from store import Store, now_ms
from feedback import FeedbackStore
from eval_gate import EvalGate, EmbeddingSimilarityEvaluator, HeldOutItem
from engine import TrainConfig


class AdapterRegistry:
    """Owns the ``adapters`` table + the 'one active adapter' invariant."""

    def __init__(self, store: Store):
        self.store = store

    def next_version(self) -> str:
        row = self.store.query("SELECT COUNT(*) AS n FROM adapters")[0]
        return f"v{row['n'] + 1:04d}"

    def register(self, adapter_id, base_model, path, training_run, eval_score,
                 baseline_score, promoted, n_examples, notes="") -> None:
        self.store.conn.execute(
            "INSERT INTO adapters (id,base_model,path,training_run,eval_score,"
            "baseline_score,promoted,is_active,n_examples,created_at,notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (adapter_id, base_model, path, training_run, eval_score,
             baseline_score, int(promoted), 0, n_examples, now_ms(), notes),
        )
        self.store.conn.commit()

    def active(self) -> Optional[dict]:
        rows = self.store.query("SELECT * FROM adapters WHERE is_active=1")
        return dict(rows[0]) if rows else None

    def set_active(self, adapter_id: str) -> None:
        """Atomically make one adapter active (enforces single-active invariant)."""
        self.store.conn.execute("UPDATE adapters SET is_active=0 WHERE is_active=1")
        self.store.conn.execute(
            "UPDATE adapters SET is_active=1 WHERE id=?", (adapter_id,))
        self.store.conn.commit()

    def rollback(self, adapter_id: str) -> dict:
        """Manually activate a prior adapter version."""
        rows = self.store.query("SELECT * FROM adapters WHERE id=?", (adapter_id,))
        if not rows:
            raise ValueError(f"no such adapter {adapter_id!r}")
        self.set_active(adapter_id)
        return dict(rows[0])

    def list_all(self) -> list[dict]:
        return [dict(r) for r in self.store.query(
            "SELECT * FROM adapters ORDER BY created_at DESC")]


class Trainer:
    """Runs the split -> train -> eval -> promote loop."""

    def __init__(self, store: Store, engine, feedback: FeedbackStore,
                 adapters_dir: str, base_model: str = "gemma-4-e4b",
                 config: Optional[TrainConfig] = None,
                 held_out_frac: float = 0.15,
                 min_improvement: float = 0.0,
                 evaluator=None, seed: int = 0):
        self.store = store
        self.engine = engine
        self.feedback = feedback
        self.adapters_dir = adapters_dir
        self.base_model = base_model
        self.config = config or TrainConfig()
        self.held_out_frac = held_out_frac
        self.registry = AdapterRegistry(store)
        self.gate = EvalGate(
            evaluator or EmbeddingSimilarityEvaluator(),
            min_improvement=min_improvement,
        )
        self.seed = seed
        self._thread: Optional[threading.Thread] = None

    # ---- run lifecycle in DB --------------------------------------------
    def _create_run(self, n_train, n_held) -> str:
        run_id = str(uuid.uuid4())
        self.store.conn.execute(
            "INSERT INTO training_runs (id,status,base_model,n_train,n_held_out,"
            "config,started_at) VALUES (?,?,?,?,?,?,?)",
            (run_id, "running", self.base_model, n_train, n_held,
             json.dumps(asdict(self.config)), now_ms()),
        )
        self.store.conn.commit()
        return run_id

    def _update_run(self, run_id, **fields):
        if not fields:
            return
        cols = ",".join(f"{k}=?" for k in fields)
        self.store.conn.execute(
            f"UPDATE training_runs SET {cols} WHERE id=?",
            (*fields.values(), run_id),
        )
        self.store.conn.commit()

    # ---- the loop --------------------------------------------------------
    def run_once(self) -> dict:
        """Execute one full training round synchronously. Returns a summary."""
        # 1. Gather pending examples and split.
        rows = self.store.query(
            "SELECT id,instruction,context,preferred_output FROM examples "
            "WHERE used_in_run IS NULL ORDER BY created_at ASC")
        if len(rows) < 2:
            return {"status": "skipped", "reason": "not enough examples (<2)"}

        rng = random.Random(self.seed)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        n_held = max(1, int(len(rows) * self.held_out_frac))
        held_idx = set(idx[:n_held])
        train_rows = [rows[i] for i in idx if i not in held_idx]
        held_rows = [rows[i] for i in idx if i in held_idx]
        if not train_rows:
            return {"status": "skipped", "reason": "no training rows after split"}

        run_id = self._create_run(len(train_rows), len(held_rows))

        try:
            # 2. Export train JSONL and fine-tune.
            adapter_id = self.registry.next_version()
            out_dir = os.path.join(self.adapters_dir, adapter_id)
            data_path = os.path.join(out_dir, "train.jsonl")
            os.makedirs(out_dir, exist_ok=True)
            self._write_jsonl(train_rows, data_path)

            result = self.engine.train_lora(data_path, out_dir, self.config)
            if not result.ok:
                self._update_run(run_id, status="failed", error=result.error,
                                 finished_at=now_ms())
                shutil.rmtree(out_dir, ignore_errors=True)
                return {"status": "failed", "run_id": run_id, "error": result.error}

            self._update_run(run_id, status="evaluating",
                             train_loss=json.dumps(result.train_loss))

            # 3. Evaluate against held-out with the gate.
            held = [HeldOutItem(r["instruction"], r["preferred_output"],
                                r["context"]) for r in held_rows]
            baseline = self.registry.active()
            baseline_path = baseline["path"] if baseline else None
            decision = self.gate.evaluate(
                self.engine, out_dir, baseline_path, held)

            # 4/5. Promote or discard; record everything.
            self.registry.register(
                adapter_id, self.base_model, out_dir, run_id,
                decision.candidate_score, decision.baseline_score,
                decision.promote, len(train_rows), notes=decision.reason)

            self.feedback.mark_used(run_id)  # consume examples either way

            if decision.promote:
                self.registry.set_active(adapter_id)
                self.engine.set_adapter(out_dir)
                status = "promoted"
            else:
                status = "rejected"
                # keep the adapter files for audit but leave inactive; restore
                # the previously-active adapter on the engine.
                self.engine.set_adapter(baseline_path)

            self._update_run(
                run_id, status=status, eval_score=decision.candidate_score,
                baseline_score=decision.baseline_score, adapter_id=adapter_id,
                finished_at=now_ms())

            return {
                "status": status, "run_id": run_id, "adapter_id": adapter_id,
                "candidate_score": decision.candidate_score,
                "baseline_score": decision.baseline_score,
                "improvement": decision.improvement, "reason": decision.reason,
                "n_train": len(train_rows), "n_held_out": len(held_rows),
            }
        except Exception as e:  # pragma: no cover - defensive
            self._update_run(run_id, status="failed", error=str(e),
                             finished_at=now_ms())
            return {"status": "failed", "run_id": run_id, "error": str(e)}

    def run_async(self, on_done=None) -> None:
        """Run one round in a background thread (UI stays responsive)."""
        def _target():
            summary = self.run_once()
            if on_done:
                on_done(summary)
        self._thread = threading.Thread(target=_target, daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @staticmethod
    def _write_jsonl(rows, path):
        with open(path, "w") as f:
            for r in rows:
                user = r["instruction"]
                if r["context"]:
                    user = f"{r['context']}\n\n{user}"
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": r["preferred_output"]},
                ]}) + "\n")

    # ---- introspection ---------------------------------------------------
    def list_runs(self, limit: int = 50) -> list[dict]:
        return [dict(r) for r in self.store.query(
            "SELECT id,status,n_train,n_held_out,eval_score,baseline_score,"
            "adapter_id,started_at,finished_at,error FROM training_runs "
            "ORDER BY started_at DESC LIMIT ?", (limit,))]
