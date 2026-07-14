"""Tests for MLXDriver.train_lora()'s VLM dispatch and in-process execution —
covering two real, confirmed-live bugs that made training dead-on-arrival:

1. Every mlx-community Gemma 4 export actually downloaded here — gemma-4-e4b
   included, the designated training target, not just the 12B — ships as a
   unified checkpoint declaring vision_config/audio_config, which the plain
   mlx_lm.lora path can't load at all.
2. Even after routing unified checkpoints to mlx_vlm.lora, shelling out via
   subprocess ([sys.executable, "-m", ...]) is dead-on-arrival in the frozen,
   packaged app specifically — sys.executable there is this sidecar's own
   bootloader executable, not a real Python interpreter, and the bundle
   ships no standalone python3 binary to fall back to. Fixed by running the
   target module in-process via runpy instead (mirrors image_gen.py's own
   subprocess-vs-in-process fix earlier this session).

Mocks runpy.run_module throughout, so these never need real model weights,
a GPU, or a full training run — matching test_mlx_driver_filters.py's
pattern of testing this class's logic in isolation.
"""
import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.mlx_driver import MLXDriver
from engine.base import TrainConfig


def fresh_driver(vlm_mode: bool, model_id: str = "gemma-4-e4b") -> MLXDriver:
    tmp = tempfile.mkdtemp()
    d = MLXDriver(os.path.join(tmp, "models"), os.path.join(tmp, "adapters"))
    d._vlm_mode = vlm_mode
    d._model_id = model_id
    return d


def _write_jsonl(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write('{"messages": [{"role": "user", "content": "hi"}, '
                '{"role": "assistant", "content": "hello"}]}\n')


def test_strip_ansi_removes_color_codes():
    colored = "\033[95mIter 5: Train loss \033[92m4.86\033[0m, more text"
    assert MLXDriver._strip_ansi(colored) == "Iter 5: Train loss 4.86, more text"


def test_parse_losses_finds_nothing_in_raw_ansi_output():
    # Regression: mlx_vlm's progress lines are unconditionally ANSI-colored
    # (no TTY check) — parsing them without stripping first silently drops
    # every line, since the loss value fails float() with escape codes
    # embedded in it.
    raw = "\033[96mIter 5: Train loss \033[92m4.86358795\033[0m, Learning Rate 1e-04"
    assert MLXDriver._parse_losses(raw) == []


def test_parse_losses_finds_loss_after_stripping_ansi():
    raw = "\033[96mIter 5: Train loss \033[92m4.86358795\033[0m, Learning Rate 1e-04"
    stripped = MLXDriver._strip_ansi(raw)
    assert MLXDriver._parse_losses(stripped) == [(5, 4.86358795)]


def test_train_lora_dispatches_to_vlm_path_when_vlm_mode():
    d = fresh_driver(vlm_mode=True)
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "train.jsonl")
        _write_jsonl(data_path)
        out_dir = os.path.join(tmp, "out")
        with patch.object(d, "_train_lora_vlm") as mock_vlm, \
             patch("engine.mlx_driver._require_mlx") as mock_require:
            d.train_lora(data_path, out_dir, TrainConfig())
            mock_vlm.assert_called_once()
            mock_require.assert_not_called()  # the mlx_lm-only path must not run


def test_train_lora_vlm_uses_output_path_not_adapter_path():
    # Regression: mlx_vlm.lora's --adapter-path means "resume from an
    # existing adapter" (raises FileNotFoundError if passed with nothing to
    # resume), not "save to" — only --output-path is the save destination.
    d = fresh_driver(vlm_mode=True)
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "train.jsonl")
        _write_jsonl(data_path)
        out_dir = os.path.join(tmp, "out")

        def fake_run_module(module_name, run_name=None):
            print("Iter 1: Train loss 1.0, ")

        with patch("runpy.run_module", side_effect=fake_run_module) as mock_run:
            result = d.train_lora(data_path, out_dir, TrainConfig(iters=1))
        assert result.ok
        mock_run.assert_called_once_with("mlx_vlm.lora", run_name="__main__")


def test_train_lora_vlm_command_has_output_path_not_adapter_path():
    d = fresh_driver(vlm_mode=True)
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "train.jsonl")
        _write_jsonl(data_path)
        out_dir = os.path.join(tmp, "out")
        captured_argv = []

        def fake_run_module(module_name, run_name=None):
            captured_argv.extend(sys.argv)

        with patch("runpy.run_module", side_effect=fake_run_module):
            d.train_lora(data_path, out_dir, TrainConfig(iters=1))
        assert "--output-path" in captured_argv
        assert "--adapter-path" not in captured_argv
        assert "--dataset" in captured_argv
        assert "--model-path" in captured_argv


def test_train_lora_vlm_restores_argv_and_load_weights_after_success():
    d = fresh_driver(vlm_mode=True)
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "train.jsonl")
        _write_jsonl(data_path)
        out_dir = os.path.join(tmp, "out")
        import mlx.nn as nn
        original_load_weights = nn.Module.load_weights
        original_argv = list(sys.argv)
        with patch("runpy.run_module"):
            d.train_lora(data_path, out_dir, TrainConfig(iters=1))
        assert sys.argv == original_argv
        assert nn.Module.load_weights is original_load_weights


def test_train_lora_vlm_surfaces_system_exit_from_bad_args():
    # argparse calls sys.exit() on invalid arguments — since this now runs
    # in-process (see _run_lora_module_inprocess), an uncaught SystemExit
    # here would kill whichever thread called this, not just "a subprocess".
    d = fresh_driver(vlm_mode=True)
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "train.jsonl")
        _write_jsonl(data_path)
        out_dir = os.path.join(tmp, "out")

        def fake_run_module(module_name, run_name=None):
            print("usage: mlx_vlm.lora [-h] ...")
            raise SystemExit(2)

        with patch("runpy.run_module", side_effect=fake_run_module):
            result = d.train_lora(data_path, out_dir, TrainConfig(iters=1))
        assert not result.ok
        assert "exited with code 2" in result.error


def test_train_lora_vlm_surfaces_training_exception():
    d = fresh_driver(vlm_mode=True)
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "train.jsonl")
        _write_jsonl(data_path)
        out_dir = os.path.join(tmp, "out")
        with patch("runpy.run_module", side_effect=ValueError("boom")):
            result = d.train_lora(data_path, out_dir, TrainConfig(iters=1))
        assert not result.ok
        assert "boom" in result.error


def test_train_lora_vlm_missing_mlx_vlm_reports_cleanly():
    d = fresh_driver(vlm_mode=True)
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "train.jsonl")
        _write_jsonl(data_path)
        out_dir = os.path.join(tmp, "out")
        with patch.dict(sys.modules, {"mlx_vlm": None}):
            result = d.train_lora(data_path, out_dir, TrainConfig(iters=1))
        assert not result.ok
        assert "mlx_vlm" in result.error


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
