"""XPU is a selectable device, and the device check scores a fixed prefix.

The check is not a study run. It reads the frozen Global-MMLU adaptation
manifest. Scoring the evaluation split here would put a device probe on
the questions the study reports.
"""

import json

from edge_slm_ace.utils import resolve_model


def _run_arm_argv(device=None):
    argv = [
        "--model",
        "tiny-gpt2",
        "--language",
        "en",
        "--arm",
        "baseline",
        "--output-dir",
        "unused",
    ]
    if device is not None:
        argv += ["--device", device]
    return argv


class TestDeviceChoices:
    def test_the_three_runners_accept_xpu(self):
        from scripts.run_arm import parse_args as arm
        from scripts.run_grid import parse_args as grid
        from scripts.screen import parse_args as screen

        assert arm(_run_arm_argv("xpu")).device == "xpu"
        assert screen(["--device", "xpu"]).device == "xpu"
        assert grid(["--device", "xpu"]).device == "xpu"

    def test_the_device_default_stays_unset(self):
        """None means the harness decides. xpu is an option, not the default."""
        from scripts.run_arm import parse_args as arm
        from scripts.run_grid import parse_args as grid
        from scripts.screen import parse_args as screen

        assert arm(_run_arm_argv()).device is None
        assert screen([]).device is None
        assert grid([]).device is None


class TestDeviceCheck:
    def test_the_path_is_named_after_the_device(self, tmp_path):
        from edge_slm_ace.reporting import device_check_path

        assert device_check_path(tmp_path, "cpu") == tmp_path / "device_check" / "cpu.json"
        assert device_check_path(tmp_path, "xpu") == tmp_path / "device_check" / "xpu.json"

    def _patch(self, monkeypatch):
        from scripts import check_device

        captured = {}

        def fake_eval(model_id, language, item_ids, **kwargs):
            captured["model_id"] = model_id
            captured["language"] = language
            captured["item_ids"] = list(item_ids)
            captured["kwargs"] = kwargs
            return {"sentinel": True}

        def fake_acc(results, language, benchmark="belebele"):
            assert results == {"sentinel": True}
            captured["accuracy_call"] = (language, benchmark)
            # Every per-item value the fake returns is 1. 0.42 cannot be that mean.
            return 0.42

        def fake_items(results, language, expected_ids=None, benchmark="belebele"):
            captured["items_call"] = (language, list(expected_ids), benchmark)
            return {item_id: 1 for item_id in expected_ids}

        monkeypatch.setattr(check_device, "run_frozen_eval", fake_eval)
        monkeypatch.setattr(check_device, "accuracy_of", fake_acc)
        monkeypatch.setattr(check_device, "per_item_correctness", fake_items)
        return captured

    def test_it_scores_the_global_mmlu_adaptation_prefix(self, tmp_path, monkeypatch):
        """Accuracy is the harness number. A mean of the per-item vector is a different one."""
        import torch

        from edge_slm_ace.data.splits import frozen_split
        from scripts import check_device

        adapt, evaluate = frozen_split("global_mmlu")
        expected = adapt[:20]
        assert len(expected) == 20
        assert set(expected).isdisjoint(evaluate)
        assert check_device.parse_args(["--device", "cpu"]).n == 20

        captured = self._patch(monkeypatch)
        assert check_device.main(["--device", "cpu", "--results-root", str(tmp_path)]) == 0

        spec = resolve_model("qwen3-1.7b")
        assert captured["model_id"] == spec.hf_id
        assert captured["language"] == "ne"
        assert captured["item_ids"] == expected
        assert captured["kwargs"]["device"] == "cpu"
        assert captured["kwargs"]["benchmark"] == "global_mmlu"
        assert captured["kwargs"]["seed"] == 42
        # Prompt wording, batch size and dtype stay on run_frozen_eval's defaults.
        assert "include_scaffold" not in captured["kwargs"]
        assert "batch_size" not in captured["kwargs"]
        assert "dtype" not in captured["kwargs"]
        assert captured["accuracy_call"] == ("ne", "global_mmlu")
        assert captured["items_call"] == ("ne", expected, "global_mmlu")

        payload = json.loads((tmp_path / "device_check" / "cpu.json").read_text(encoding="utf-8"))
        assert payload["accuracy"] == 0.42
        assert payload["per_item"] == {item_id: 1 for item_id in expected}
        assert list(payload["per_item"]) == expected
        assert payload["item_ids"] == expected
        assert payload["revision"] == spec.revision
        assert payload["hf_id"] == spec.hf_id
        assert payload["model"] == "qwen3-1.7b"
        assert payload["device"] == "cpu"
        assert payload["dtype"] is None
        assert payload["torch_version"] == torch.__version__
        assert payload["benchmark"] == "global_mmlu"
        assert payload["language"] == "ne"
        assert payload["n"] == 20
        assert "xpu_available" in payload["environment"]
        assert "xpu_device_name" in payload["environment"]

    def test_n_is_a_prefix_of_the_adaptation_manifest(self, tmp_path, monkeypatch):
        from edge_slm_ace.data.splits import frozen_split
        from scripts import check_device

        adapt, evaluate = frozen_split("global_mmlu")
        captured = self._patch(monkeypatch)
        assert (
            check_device.main(
                [
                    "--device",
                    "xpu",
                    "--n",
                    "7",
                    "--dtype",
                    "float32",
                    "--results-root",
                    str(tmp_path),
                ]
            )
            == 0
        )
        assert captured["item_ids"] == adapt[:7]
        assert set(captured["item_ids"]).isdisjoint(evaluate)
        assert captured["kwargs"]["dtype"] == "float32"
        assert captured["kwargs"]["device"] == "xpu"
        payload = json.loads((tmp_path / "device_check" / "xpu.json").read_text(encoding="utf-8"))
        assert payload["n"] == 7
        assert payload["device"] == "xpu"
        assert payload["dtype"] == "float32"
        assert payload["item_ids"] == adapt[:7]

    def test_a_missing_harness_accuracy_writes_nothing(self, tmp_path, monkeypatch):
        """A file with no harness accuracy would be read later as a reference."""
        from scripts import check_device

        monkeypatch.setattr(check_device, "run_frozen_eval", lambda *a, **k: {})
        monkeypatch.setattr(check_device, "accuracy_of", lambda *a, **k: None)

        assert check_device.main(["--device", "xpu", "--results-root", str(tmp_path)]) == 1
        assert not (tmp_path / "device_check" / "xpu.json").exists()
