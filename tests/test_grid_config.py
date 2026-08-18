"""Tests for the grid runner's config handling.

configs/experiment_grid.yaml carried a documented `scoring:` block that nothing
forwarded, so editing alpha/beta/gamma/delta changed no behaviour and reported
no error. These tests cover the forwarding and the validator that now makes an
unread key fail loudly.
"""

from pathlib import Path

import pytest
import yaml

from scripts.run_eval_grid import (
    SCORING_KEYS,
    build_experiment_command,
    validate_mode_configs,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment_grid.yaml"


@pytest.fixture
def grid_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def mode_named(config, name):
    return next(m for m in config["modes"] if m["name"] == name)


def build(config, mode_name, task_name="sciq_test", device="cuda"):
    task = next(t for t in config["tasks"] if t["task_name"] == task_name)
    return build_experiment_command(
        model_config=config["models"][0],
        task_config=task,
        mode_config=mode_named(config, mode_name),
        device=device,
        output_dir=Path("results/m/t/a/d"),
        defaults=config.get("defaults", {}),
        limit=None,
        seed=42,
        scoring=config.get("scoring"),
    )


class TestScoringIsForwarded:
    def test_every_scoring_key_reaches_the_command(self, grid_config):
        cmd = build(grid_config, "tinyace_wm_256")
        for key in SCORING_KEYS:
            assert f"--{key.replace('_', '-')}" in cmd

    def test_config_values_are_the_ones_forwarded(self, grid_config):
        cmd = build(grid_config, "tinyace_wm_256")
        assert cmd[cmd.index("--alpha") + 1] == str(grid_config["scoring"]["alpha"])

    def test_a_mode_can_override_one_weight(self, grid_config):
        mode_named(grid_config, "tinyace_wm_256")["gamma"] = 0.9
        cmd = build(grid_config, "tinyace_wm_256")
        assert cmd[cmd.index("--gamma") + 1] == "0.9"
        assert cmd[cmd.index("--alpha") + 1] == str(grid_config["scoring"]["alpha"])

    def test_baseline_takes_no_scoring_flags(self, grid_config):
        """Only the arms with a playbook have a retention score to weight."""
        cmd = build(grid_config, "baseline")
        assert "--alpha" not in cmd


class TestFrozenArm:
    def test_frozen_mode_forwards_an_init_playbook(self, grid_config):
        cmd = build(grid_config, "tinyace_wm_256_frozen")
        assert "--playbook-mode" in cmd and cmd[cmd.index("--playbook-mode") + 1] == "frozen"
        init = cmd[cmd.index("--init-playbook") + 1]
        assert "sciq_val" in init, "the frozen arm must read the adaptation split's playbook"
        assert "{" not in init, "the template must be fully substituted"

    def test_frozen_arm_is_scoped_off_its_own_adaptation_split(self, grid_config):
        """Freezing on sciq_val and scoring on sciq_val is the leak, not the fix."""
        mode = mode_named(grid_config, "tinyace_wm_256_frozen")
        assert mode["only_tasks"] == ["sciq_test"]

    def test_learning_arms_get_no_frozen_flags(self, grid_config):
        cmd = build(grid_config, "tinyace_wm_256")
        assert "--playbook-mode" not in cmd


class TestModeKeyValidation:
    def test_the_shipped_config_is_clean(self, grid_config):
        assert validate_mode_configs(grid_config["modes"]) == []

    def test_an_unread_key_is_reported(self):
        problems = validate_mode_configs([{"name": "x", "mode": "ace", "aplha": 1.0}])
        assert len(problems) == 1
        assert "aplha" in problems[0]
