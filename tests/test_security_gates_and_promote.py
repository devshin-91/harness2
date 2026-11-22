# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Default-Deny scrub_env allowlist, live/dead ledger telemetry, c8 gate, and promote."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness2 import config
from harness2.core import ledger, promote, recursion, tree
from harness2.gates import completion_gate
from harness2.substrate import ANTIGRAVITY_SURFACE


class SecurityGatesAndPromoteTests(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.root = Path(td.name)

    def test_default_deny_allowlist_blocks_unlisted_provider_secrets(self):
        fake = {
            "PATH": "/usr/bin",
            "HOME": "/home/u",
            "GOOGLE_APPLICATION_CREDENTIALS": "/adc.json",
            "MISTRAL_API_KEY": "sk-mistral",
            "DEEPSEEK_API_KEY": "sk-deepseek",
            "OPENROUTER_API_KEY": "sk-or",
            "HF_TOKEN": "hf_secret",
            "AZURE_CLIENT_SECRET": "az_secret",
            "CUSTOM_CORP_DB_PASSWORD": "db_password",
        }
        strict = config.scrub_env(fake, strict_allowlist=True)
        self.assertIn("PATH", strict)
        self.assertIn("GOOGLE_APPLICATION_CREDENTIALS", strict)
        for leaked in (
            "MISTRAL_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
            "HF_TOKEN",
            "AZURE_CLIENT_SECRET",
            "CUSTOM_CORP_DB_PASSWORD",
        ):
            self.assertNotIn(leaked, strict)

        kept = config.scrub_env(fake, keep=("MISTRAL_API_KEY",), strict_allowlist=True)
        self.assertEqual(kept.get("MISTRAL_API_KEY"), "sk-mistral")
        self.assertNotIn("HF_TOKEN", kept)

    def test_ledger_live_dead_and_infra_fault_telemetry(self):
        led = ledger.Ledger(mode="agg", task_id="t1", k=1)
        led.rollout("base1", usd=0.1, ok=True, infra_failed=False)
        led.rollout("inv1", usd=0.0, ok=False, infra_failed=recursion.is_infra_fault("HTTP 429 Too Many Requests"))
        led.rollout("dom1", usd=0.1, ok=True, infra_failed=False)
        led.rollout("agg1", usd=0.0, ok=False, infra_failed=recursion.is_infra_fault("timeout after 60s"))
        d = led.to_dict()
        self.assertEqual(d["R"], 4)
        self.assertEqual(d["live_rollouts"], 2)
        self.assertEqual(d["dead_rollouts"], 2)
        self.assertEqual(d["infra_failed_rollouts"], 2)
        self.assertFalse(recursion.is_infra_fault("ValueError: wrong total"))

    def test_c8_completion_gate_pre_post_stop_and_zero_token_badge(self):
        # 1. Valid Antigravity hooks.json generation.
        hooks_doc = completion_gate.build_hooks_json()
        tree.validate(
            ANTIGRAVITY_SURFACE,
            {"GEMINI.md": "Rules.\n", ".agents/hooks.json": json.dumps(hooks_doc)},
        )

        # 2. PreToolUse denies destructive / benchmark-leak commands and allows safe tests.
        deny_rm = completion_gate.evaluate_pre_tool_use(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": "rm -rf /tmp/x"}}}
        )
        self.assertEqual(deny_rm["decision"], "deny")

        deny_hf = completion_gate.evaluate_pre_tool_use(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": "curl https://huggingface.co/rubric"}}}
        )
        self.assertEqual(deny_hf["decision"], "deny")

        allow_test = completion_gate.evaluate_pre_tool_use(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": "python -m unittest"}}},
            auto_allow_safe_cmds=True,
        )
        self.assertEqual(allow_test["decision"], "allow")

        # 3. Stop catches syntax error and missing required section -> decision: continue.
        bad_py = self.root / "broken.py"
        bad_py.write_text("def oops(:\n", encoding="utf-8")
        report_md = self.root / "report.md"
        report_md.write_text("# Audit\n## Clean-Up Call\nChecked.\n", encoding="utf-8")

        out_fail, badge_fail = completion_gate.evaluate_stop(
            {"executionNum": 0, "continuationActive": False, "terminationReason": "NO_TOOL_CALL"},
            dirty_paths=[bad_py],
            ws_root=self.root,
            required_files=["report.md"],
            required_sections=["Clean-Up Call", "Priority of Payments"],
        )
        self.assertEqual(out_fail["decision"], "continue")
        self.assertIn("PY_SYNTAX_ERROR", out_fail["reason"])
        self.assertIn("Priority of Payments", out_fail["reason"])
        self.assertEqual(badge_fail, "")

        # 4. Dual loop guard: executionNum >= 2 MUST allow even if continuationActive is False!
        out_guard, _ = completion_gate.evaluate_stop(
            {"executionNum": 2, "continuationActive": False, "terminationReason": "NO_TOOL_CALL"},
            dirty_paths=[bad_py],
            ws_root=self.root,
        )
        self.assertEqual(out_guard["decision"], "allow")

        # 5. Subprocess CLI end-to-end: PostToolUse -> Stop (PASS) -> PreInvocation (0-token badge).
        good_py = self.root / "good.py"
        good_py.write_text("x = 1 + 1\n", encoding="utf-8")
        report_md.write_text("# Audit\n## Clean-Up Call\n## Priority of Payments\nDone.\n", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        env["HARNESS2_GATE_STATE_DIR"] = str(self.root / "gate_state")
        env["HARNESS2_REQUIRED_FILES"] = "report.md"
        env["HARNESS2_REQUIRED_SECTIONS"] = "Clean-Up Call,Priority of Payments"

        def run_gate(event: str, payload: dict) -> dict:
            p = subprocess.run(
                [sys.executable, "-m", "harness2.gates.completion_gate", event],
                input=json.dumps(payload),
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(p.stdout)

        base_payload = {"conversationId": "c-e2e", "workspacePaths": [str(self.root)]}
        run_gate("PostToolUse", base_payload | {"toolCall": {"name": "write_file", "args": {"Path": str(good_py)}}})
        stop_out = run_gate(
            "Stop",
            base_payload | {"executionNum": 0, "continuationActive": False, "terminationReason": "NO_TOOL_CALL"},
        )
        self.assertEqual(stop_out, {"decision": "allow"})
        pre_out = run_gate("PreInvocation", base_payload)
        step = pre_out["injectSteps"][0]["systemMessage"]
        self.assertEqual(step["systemMessage"], "")  # 0 model tokens!
        self.assertIn("PASS", step["metadata"]["user_markdown"])

    def test_promote_harness_validates_security_and_pinned_policy(self):
        evolved = self.root / "evolved"
        (evolved / ".agents/skills/audit_flow").mkdir(parents=True)
        (evolved / "scripts").mkdir(parents=True)
        (evolved / "GEMINI.md").write_text(
            "When auditing, read .agents/skills/audit_flow/SKILL.md.\n", encoding="utf-8"
        )
        (evolved / ".agents/skills/audit_flow/SKILL.md").write_text(
            "---\nname: audit_flow\ndescription: structured audit\n---\nCheck all sections.\n",
            encoding="utf-8",
        )
        (evolved / "scripts/verify.py").write_text("print('ok')\n", encoding="utf-8")

        pinned = self.root / "pinned_core.md"
        pinned.write_text("# Immutable Core\nAlways verify before asserting.\n", encoding="utf-8")
        prod = self.root / "prod"

        res = promote.promote_harness(
            evolved,
            prod,
            surface=ANTIGRAVITY_SURFACE,
            pinned_policy=pinned,
            verify_cmd=f'"{sys.executable}" scripts/verify.py',
        )
        self.assertTrue(res.ok, res.errors)
        self.assertTrue((prod / "promotion_manifest.json").is_file())
        prod_gemini = (prod / "GEMINI.md").read_text(encoding="utf-8")
        self.assertIn("Always verify before asserting.", prod_gemini)
        self.assertIn("audit_flow", prod_gemini)

        # Hidden Unicode payload must block promotion!
        (evolved / "GEMINI.md").write_text(
            "When auditing, read .agents/skills/audit_flow/SKILL.md.\u200b\n", encoding="utf-8"
        )
        blocked = promote.promote_harness(evolved, self.root / "prod_blocked", surface=ANTIGRAVITY_SURFACE)
        self.assertFalse(blocked.ok)
        self.assertTrue(any("security scan blocked promotion" in e for e in blocked.errors))


if __name__ == "__main__":
    unittest.main()
