"""Tests for image_gen.py — mocks the mflux model/save calls throughout, so
these never need mflux installed, a GPU, or network access."""
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import image_gen
from image_gen import generate, is_available, ImageGenError


def _reset_cache():
    image_gen._available = None
    image_gen._model = None


def test_is_available_false_when_mflux_missing():
    _reset_cache()
    with patch("image_gen._available", None), \
         patch.dict(sys.modules, {"mflux": None}):
        assert is_available() is False
    _reset_cache()


def test_is_available_true_when_mflux_present():
    _reset_cache()
    with patch.dict(sys.modules, {"mflux": MagicMock()}):
        assert is_available() is True
    _reset_cache()


def test_generate_raises_when_unavailable():
    _reset_cache()
    with patch("image_gen.is_available", return_value=False):
        try:
            generate("a cat", tempfile.mkdtemp())
            assert False, "expected ImageGenError"
        except ImageGenError as e:
            assert "not installed" in str(e)


def test_generate_rejects_empty_prompt():
    with patch("image_gen.is_available", return_value=True):
        try:
            generate("   ", tempfile.mkdtemp())
            assert False, "expected ImageGenError"
        except ImageGenError as e:
            assert "empty" in str(e)


def test_generate_success_writes_file_and_returns_metadata():
    d = tempfile.mkdtemp()
    fake_image = object()
    fake_model = MagicMock()
    fake_model.generate_image.return_value = fake_image

    def fake_save(image, path):
        assert image is fake_image
        with open(path, "wb") as f:
            f.write(b"fake png bytes")

    with patch("image_gen.is_available", return_value=True), \
         patch("image_gen._get_model", return_value=fake_model), \
         patch("image_gen._save_image", side_effect=fake_save):
        result = generate("a cat", d, seed=42)
    assert result["seed"] == 42
    assert os.path.exists(result["path"])
    assert result["filename"].endswith(".png")
    assert result["seconds"] >= 0


def test_generate_passes_prompt_and_dimensions_to_model():
    d = tempfile.mkdtemp()
    fake_model = MagicMock()
    fake_model.generate_image.return_value = object()

    with patch("image_gen.is_available", return_value=True), \
         patch("image_gen._get_model", return_value=fake_model), \
         patch("image_gen._save_image"):
        generate("a dog", d, width=256, height=256, steps=2, seed=7)

    _, kwargs = fake_model.generate_image.call_args
    assert kwargs["prompt"] == "a dog"
    assert kwargs["width"] == 256
    assert kwargs["height"] == 256
    assert kwargs["num_inference_steps"] == 2
    assert kwargs["seed"] == 7


def test_generate_raises_when_model_generation_fails():
    d = tempfile.mkdtemp()
    fake_model = MagicMock()
    fake_model.generate_image.side_effect = RuntimeError("out of memory")

    with patch("image_gen.is_available", return_value=True), \
         patch("image_gen._get_model", return_value=fake_model):
        try:
            generate("a cat", d)
            assert False, "expected ImageGenError"
        except ImageGenError as e:
            assert "out of memory" in str(e)


def test_generate_raises_when_save_fails():
    d = tempfile.mkdtemp()
    fake_model = MagicMock()
    fake_model.generate_image.return_value = object()

    with patch("image_gen.is_available", return_value=True), \
         patch("image_gen._get_model", return_value=fake_model), \
         patch("image_gen._save_image", side_effect=OSError("disk full")):
        try:
            generate("a cat", d)
            assert False, "expected ImageGenError"
        except ImageGenError as e:
            assert "disk full" in str(e)


def test_generate_succeeds_across_repeated_calls():
    d = tempfile.mkdtemp()
    fake_model = MagicMock()
    fake_model.generate_image.return_value = object()

    with patch("image_gen.is_available", return_value=True), \
         patch("image_gen._get_model", return_value=fake_model), \
         patch("image_gen._save_image"):
        r1 = generate("a cat", d)
        r2 = generate("a dog", d)
    assert r1["filename"] != r2["filename"]


def test_get_model_caches_instance():
    # _get_model() does its mflux imports lazily inside the function body —
    # inject fake modules into sys.modules so this test needs neither mflux
    # nor torch actually installed.
    _reset_cache()
    fake_config_mod = MagicMock()
    fake_config_mod.ModelConfig.from_name.return_value = "cfg"
    fake_klein_mod = MagicMock()
    fake_klein_mod.Flux2Klein.return_value = "the-model"
    with patch.dict(sys.modules, {
        "mflux.models.common.config.model_config": fake_config_mod,
        "mflux.models.flux2.variants.txt2img.flux2_klein": fake_klein_mod,
    }):
        first = image_gen._get_model()
        second = image_gen._get_model()
        assert first == "the-model"
        assert second == "the-model"
        assert fake_klein_mod.Flux2Klein.call_count == 1  # only constructed once
    _reset_cache()


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
