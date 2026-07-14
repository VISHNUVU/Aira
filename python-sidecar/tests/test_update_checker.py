"""Tests for update_checker.py — pre-flight checks and the install_dmg
backup/restore safety net. Uses tempdirs and mocked subprocess calls so this
never touches a real /Applications or mounts a real .dmg."""
import os
import shutil
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from update_checker import UpdateChecker


def _make_app_bundle(path: str, marker: str) -> None:
    os.makedirs(os.path.join(path, "Contents", "MacOS"), exist_ok=True)
    with open(os.path.join(path, "Contents", "MacOS", "marker"), "w") as f:
        f.write(marker)


def test_check_target_writable_ok_for_writable_dir():
    with tempfile.TemporaryDirectory() as tmp:
        checker = UpdateChecker("0.1.0")
        target = os.path.join(tmp, "Aria.app")
        assert checker.check_target_writable(target) is None


def test_check_target_writable_reports_error_for_readonly_dir():
    with tempfile.TemporaryDirectory() as tmp:
        readonly = os.path.join(tmp, "locked")
        os.makedirs(readonly)
        os.chmod(readonly, 0o500)
        checker = UpdateChecker("0.1.0")
        target = os.path.join(readonly, "Aria.app")
        try:
            err = checker.check_target_writable(target)
            assert err is not None
            assert "locked" in err
        finally:
            os.chmod(readonly, 0o700)


def _fake_hdiutil_and_ditto(mount_point: str, new_app_src: str, fail_ditto: bool = False):
    """Patches subprocess.run so 'hdiutil attach' populates mount_point with
    a fake new .app (copied from new_app_src), 'ditto' does a real copy (or
    raises, to simulate a mid-swap failure), and 'hdiutil detach' no-ops."""
    import subprocess

    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[0] == "hdiutil" and cmd[1] == "attach":
            dest = os.path.join(mount_point, "NewApp.app")
            shutil.copytree(new_app_src, dest)
            class R: returncode = 0
            return R()
        if cmd[0] == "hdiutil" and cmd[1] == "detach":
            class R: returncode = 0
            return R()
        if cmd[0] == "ditto":
            if fail_ditto:
                raise RuntimeError("simulated ditto failure mid-copy")
            return real_run(cmd, **kwargs)
        return real_run(cmd, **kwargs)

    return fake_run


def test_install_dmg_swaps_bundle_on_success():
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "Aria.app")
        _make_app_bundle(target, "old-version")
        new_src = os.path.join(tmp, "staged-new.app")
        _make_app_bundle(new_src, "new-version")

        checker = UpdateChecker("0.1.0")
        with tempfile.TemporaryDirectory() as fake_mount:
            with patch("subprocess.run", side_effect=_fake_hdiutil_and_ditto(fake_mount, new_src)):
                with patch("tempfile.mkdtemp", return_value=fake_mount):
                    checker.install_dmg(os.path.join(tmp, "fake.dmg"), target_app=target)

        with open(os.path.join(target, "Contents", "MacOS", "marker")) as f:
            assert f.read() == "new-version"
        assert not os.path.exists(target.rstrip("/") + ".update-backup")


def test_install_dmg_restores_backup_on_failure():
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "Aria.app")
        _make_app_bundle(target, "old-version")
        new_src = os.path.join(tmp, "staged-new.app")
        _make_app_bundle(new_src, "new-version")

        checker = UpdateChecker("0.1.0")
        with tempfile.TemporaryDirectory() as fake_mount:
            with patch("subprocess.run", side_effect=_fake_hdiutil_and_ditto(fake_mount, new_src, fail_ditto=True)):
                with patch("tempfile.mkdtemp", return_value=fake_mount):
                    try:
                        checker.install_dmg(os.path.join(tmp, "fake.dmg"), target_app=target)
                        assert False, "expected install_dmg to raise"
                    except RuntimeError:
                        pass

        # The original bundle must be back in place, untouched, and no
        # leftover backup directory — a failed update must be invisible.
        assert os.path.exists(target)
        with open(os.path.join(target, "Contents", "MacOS", "marker")) as f:
            assert f.read() == "old-version"
        assert not os.path.exists(target.rstrip("/") + ".update-backup")


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
