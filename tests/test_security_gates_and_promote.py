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
        self.assertNotIn("GOOGLE_APPLICATION_CREDENTIALS", strict)
        with_adc = config.scrub_env(
            fake, keep=("GOOGLE_APPLICATION_CREDENTIALS",), strict_allowlist=True
        )
        self.assertIn("GOOGLE_APPLICATION_CREDENTIALS", with_adc)
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
        self.assertIn("gate_receipt.json", step["metadata"]["user_markdown"])

        # 6. Replayable receipt written to disk with command, input_digest, and artifact SHA-256.
        receipt_files = list((self.root / "gate_state").rglob("gate_receipt.json"))
        self.assertEqual(len(receipt_files), 1)
        receipt = json.loads(receipt_files[0].read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(len(receipt["input_digest"]), 64)
        self.assertIn("good.py", receipt["artifact_hashes"])
        self.assertIn("report.md", receipt["artifact_hashes"])

        # 7. Fail closed when require_artifacts=True and no verifiable artifact exists.
        empty_ws = self.root / "empty_ws"
        empty_ws.mkdir()
        fc_receipt: dict = {}
        out_fc, _ = completion_gate.evaluate_stop(
            {"executionNum": 0, "continuationActive": False},
            dirty_paths=[],
            ws_root=empty_ws,
            require_artifacts=True,
            receipt_out=fc_receipt,
        )
        self.assertEqual(out_fc["decision"], "continue")
        self.assertIn("MISSING_ARTIFACT", out_fc["reason"])
        self.assertEqual(fc_receipt["status"], "FAIL")

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

        verify_cmd = f'"{sys.executable}" scripts/verify.py'
        res = promote.promote_harness(
            evolved,
            prod,
            surface=ANTIGRAVITY_SURFACE,
            pinned_policy=pinned,
            verify_cmd=verify_cmd,
        )
        self.assertTrue(res.ok, res.errors)
        manifest_file = prod / "promotion_manifest.json"
        self.assertTrue(manifest_file.is_file())
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.assertEqual(manifest["command"], verify_cmd)
        self.assertEqual(len(manifest["input_digest"]), 64)
        self.assertIn("commit_sha", manifest)
        self.assertIn("GEMINI.md", manifest["files"])
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

    def test_adversarial_hardening_pre_stop_promote_and_ledger(self):
        from harness2.core import evidence_gate, recursion

        # 1. PreToolUse blocks RCE shell metacharacters, rm -fr /, rm -r -f /, +main push, read_file creds;
        #    allows safe git push --force-with-lease.
        for bad_cmd in (
            "ls $(whoami)",
            "ls > ~/.bashrc",
            "ls & rm -fr /",
            "ls\nid",
            "rm -r -f /",
            "rm -fr /",
            "git push origin +main",
        ):
            res = completion_gate.evaluate_pre_tool_use(
                {"toolCall": {"name": "run_command", "args": {"CommandLine": bad_cmd}}},
                auto_allow_safe_cmds=True,
            )
            self.assertNotEqual(res.get("decision"), "allow", f"must NOT auto-allow {bad_cmd!r}")

        for denied_cmd in (
            "rm -fr /",
            "echo ok;rm -rf /",
            "true&&rm -rf /",
            'bash -c "rm -rf /"',
            "git push --force;true",
            "git -C . push --force",
        ):
            self.assertEqual(
                completion_gate.evaluate_pre_tool_use(
                    {"toolCall": {"name": "run_command", "args": {"CommandLine": denied_cmd}}}
                ).get("decision"),
                "deny",
                f"expected deny for {denied_cmd!r}",
            )
        for cred_path in ("~/.config/gcloud/adc.json", r"C:\Users\alice\.ssh\id_rsa"):
            self.assertEqual(
                completion_gate.evaluate_pre_tool_use(
                    {"toolCall": {"name": "read_file", "args": {"Path": cred_path}}}
                ).get("decision"),
                "deny",
                f"expected deny for {cred_path!r}",
            )
        # Safe --force-with-lease must NOT be falsely denied.
        self.assertEqual(
            completion_gate.evaluate_pre_tool_use(
                {"toolCall": {"name": "run_command", "args": {"CommandLine": "git push --force-with-lease"}}}
            ),
            {},
        )

        # 2. Stop hook RE-VERIFIES on continuationActive=True and retains dirty queue on FAIL!
        ws = self.root / "ws_retry"
        ws.mkdir()
        broken = ws / "broken.py"
        broken.write_text("def oops(:\n", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        env["HARNESS2_GATE_STATE_DIR"] = str(self.root / "gate_retry_state")
        env["HARNESS2_MAX_CONTINUATIONS"] = "1"

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

        base_p = {"conversationId": "c-retry", "workspacePaths": [str(ws)]}
        run_gate("PostToolUse", base_p | {"toolCall": {"name": "write_file", "args": {"Path": str(broken)}}})
        # Turn 0: FAIL -> continue AND dirty_state.json stays populated!
        stop0 = run_gate("Stop", base_p | {"executionNum": 0, "continuationActive": False})
        self.assertEqual(stop0["decision"], "continue")
        # Turn 1 (agent did NOT fix): STILL re-verifies -> records truthful FAIL receipt (NOT PASS/SKIPPED_GUARD)!
        stop1 = run_gate("Stop", base_p | {"executionNum": 1, "continuationActive": True})
        self.assertEqual(stop1["decision"], "allow")  # capped at max_continuations=1
        rec_path = list((self.root / "gate_retry_state").rglob("gate_receipt.json"))[0]
        rec1 = json.loads(rec_path.read_text(encoding="utf-8"))
        self.assertEqual(rec1["status"], "FAIL")
        self.assertTrue(any(i["code"] == "PY_SYNTAX_ERROR" for i in rec1["issues"]))

        # Now fix broken.py and re-verify -> PASS + dirty queue cleared!
        broken.write_text("def ok():\n    return 1\n", encoding="utf-8")
        stop2 = run_gate("Stop", base_p | {"executionNum": 1, "continuationActive": True})
        self.assertEqual(stop2["decision"], "allow")
        rec2 = json.loads(rec_path.read_text(encoding="utf-8"))
        self.assertEqual(rec2["status"], "PASS")
        ok_v, errs_v = evidence_gate.verify_receipt(rec2)
        self.assertTrue(ok_v, errs_v)

        # 3. promote_harness removes stale target hooks/scripts, verifies merged tree with scrubbed env,
        #    and produces a promotion_manifest.json directly verifiable via verify_receipt() on Windows/POSIX!
        evolved = self.root / "ev_clean"
        (evolved / "scripts").mkdir(parents=True)
        (evolved / "GEMINI.md").write_text("# Clean rules\n", encoding="utf-8")
        (evolved / "scripts/check_pin.py").write_text(
            "import os, pathlib, sys\n"
            "assert 'OPENAI_API_KEY' not in os.environ, 'secret leaked'\n"
            "assert 'Immutable Core' in pathlib.Path('GEMINI.md').read_text(encoding='utf-8')\n",
            encoding="utf-8",
        )
        pinned = self.root / "pin_p.md"
        pinned.write_text("# Immutable Core\n", encoding="utf-8")
        prod = self.root / "prod_stale"
        (prod / ".agents").mkdir(parents=True)
        (prod / "scripts").mkdir(parents=True)
        (prod / ".agents/hooks.json").write_text('{"old": {"Stop": [{"command": "echo stale"}]}}', encoding="utf-8")
        (prod / "scripts/stale.py").write_text("print('stale')\n", encoding="utf-8")

        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-test-must-be-scrubbed"
        try:
            pres = promote.promote_harness(
                evolved,
                prod,
                surface=ANTIGRAVITY_SURFACE,
                pinned_policy=pinned,
                verify_cmd=f'"{sys.executable}" scripts/check_pin.py',
            )
        finally:
            if old_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_key
        self.assertTrue(pres.ok, pres.errors)
        self.assertFalse((prod / ".agents/hooks.json").exists())
        self.assertFalse((prod / "scripts/stale.py").exists())
        self.assertIn(".agents/hooks.json", pres.removed_stale_files)
        manifest = json.loads((prod / "promotion_manifest.json").read_text(encoding="utf-8"))
        ok_m, errs_m = evidence_gate.verify_receipt(manifest)
        self.assertTrue(ok_m, errs_m)

        # OpenCode surface with 0-byte placeholder guardrails.md must ALSO pass verify_receipt!
        from harness2.substrate import OPENCODE_SURFACE

        oc_src = self.root / "oc_src"
        oc_src.mkdir()
        (oc_src / "systemprompt.md").write_text("# OC System\n", encoding="utf-8")
        (oc_src / "guardrails.md").write_text("", encoding="utf-8")
        oc_dst = self.root / "oc_prod"
        oc_res = promote.promote_harness(oc_src, oc_dst, surface=OPENCODE_SURFACE)
        self.assertTrue(oc_res.ok, oc_res.errors)
        oc_manifest = json.loads((oc_dst / "promotion_manifest.json").read_text(encoding="utf-8"))
        ok_oc, errs_oc = evidence_gate.verify_receipt(oc_manifest)
        self.assertTrue(ok_oc, errs_oc)
        self.assertIn("guardrails.md", oc_manifest["empty_files"])

        # 4. config & ledger invariants: GOOGLE_APPLICATION_CREDENTIALS dropped unless kept; auth 401/403;
        #    contradictory ok=True + infra_failed=True rejected.
        scrubbed = config.scrub_env_allowlist(
            {"PATH": "/bin", "GOOGLE_APPLICATION_CREDENTIALS": "/sa.json", "https_proxy": "http://p:8080"}
        )
        self.assertNotIn("GOOGLE_APPLICATION_CREDENTIALS", scrubbed)
        self.assertIn("https_proxy", scrubbed)
        self.assertTrue(recursion.is_infra_fault("HTTP 401 Unauthorized"))
        self.assertTrue(recursion.is_infra_fault("403 Forbidden"))
        led = ledger.Ledger(mode="agg", task_id="t", k=1)
        with self.assertRaises(ValueError):
            led.rollout("inv1", ok=True, infra_failed=True)


if __name__ == "__main__":
    unittest.main()
