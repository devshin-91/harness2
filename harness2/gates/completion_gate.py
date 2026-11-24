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

"""Zero-token deterministic c8 Completion & Safety Gate for Antigravity lifecycle hooks (#161).

Runs as a stdlib-only command hook (`python -m harness2.gates.completion_gate <Event>`):
  - PreToolUse: blocks destructive commands (`rm -rf`/`rm -fr`/`rm -r -f`, force-push except
    `--force-with-lease`), host credential reads (via shell OR `read_file`/`view_file`), and
    (optionally) benchmark rubric hosts; optionally auto-allows strictly metacharacter-free
    read-only/test workspace commands.
  - PostToolUse: atomically records modified file paths into a per-conversation dirty queue.
  - Stop: always evaluates deterministic checks (including on `continuationActive=True` retry
    turns!), writes a replayable `gate_receipt.json` (`commit_sha`, `command`, `input_digest`,
    `artifact_hashes`, optional HMAC-SHA256), clears the dirty queue ONLY on `PASS`, and caps
    `continue` continuations at `executionNum < max_continuations` so failures are never masked
    and infinite loops cannot occur.
  - PreInvocation: emits the saved 0-token `user_markdown` receipt badge and clears it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

AGY_RULE_MAX_BYTES = 24_000
RULE_FILENAMES = frozenset({"GEMINI.md", "GEMINI.ko.md", "GEMINI.en.md", "AGENTS.md"})

_DESTRUCTIVE_GIT_PUSH_RE = re.compile(
    r"\bgit\s+push\b[^\n]*(?:(?:--force(?!-with-lease)|-f)(?:\s|$)|\s\+\S+)",
    re.IGNORECASE,
)
_CREDENTIAL_PATH_RE = re.compile(
    r"(?:~|\$HOME|%USERPROFILE%|/root|/home/[^/\s]+|/Users/[^/\s]+)"
    r"(?:[/\\]\.(?:ssh|aws|gcloud|netrc)|[/\\]\.config[/\\]gcloud)\b",
    re.IGNORECASE,
)
_BENCHMARK_LEAK_HOST_RE = re.compile(
    r"\b(?:huggingface\.co|hf\.co|raw\.githubusercontent\.com|api\.github\.com/repos)\b",
    re.IGNORECASE,
)
_SAFE_READONLY_CMD_RE = re.compile(
    r"^\s*(?:pytest|python3?\s+-m\s+(?:unittest|pytest|py_compile|ref_factcheck)|"
    r"ruff\s+check|git\s+(?:status|diff|log)|ls|dir)(?:\s|$)",
    re.IGNORECASE,
)
_SHELL_META_RE = re.compile(r"[;&|`$><\r\n]")
_DANGEROUS_RM_TARGETS = frozenset({"/", "~", "*", "..", "/*", "~/", "../"})


@dataclass(frozen=True)
class GateIssue:
    code: str
    target: str
    message: str


def _is_destructive_rm(cmd: str) -> bool:
    """Detect `rm` with recursive + force flags in any order/separation targeting broad paths."""
    if not re.search(r"\brm\b", cmd):
        return False
    try:
        tokens = shlex.split(cmd, posix=(os.name != "nt"))
    except ValueError:
        tokens = cmd.split()
    for i, tok in enumerate(tokens):
        if tok.lower() not in ("rm", "/bin/rm", "/usr/bin/rm"):
            continue
        has_r = False
        has_f = False
        targets: list[str] = []
        for arg in tokens[i + 1 :]:
            if arg in (";", "&&", "||", "|", "&"):
                break
            if arg == "--recursive":
                has_r = True
            elif arg == "--force":
                has_f = True
            elif arg.startswith("-") and not arg.startswith("--"):
                flags = arg[1:]
                if "r" in flags or "R" in flags:
                    has_r = True
                if "f" in flags:
                    has_f = True
            else:
                targets.append(arg.strip("\"'"))
        if has_r and has_f and any(
            t in _DANGEROUS_RM_TARGETS or t.startswith(("/", "~", "..", "*")) for t in targets
        ):
            return True
    return False


def _state_dir(payload: dict) -> Path:
    root = os.environ.get("HARNESS2_GATE_STATE_DIR") or os.environ.get("PLUGIN_DATA")
    base = Path(root) if root else Path.home() / ".cache" / "harness2-completion-gate"
    conv_id = str(payload.get("conversationId") or "default")
    safe = hashlib.sha1(conv_id.encode("utf-8")).hexdigest()[:12]
    d = base / "sessions" / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _workspace_root(payload: dict) -> Path:
    ws_list = payload.get("workspacePaths") or []
    if ws_list and isinstance(ws_list[0], str) and ws_list[0].strip():
        return Path(ws_list[0].strip())
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        return Path(cwd.strip())
    cur = Path.cwd()
    return cur.parent if cur.name == ".agents" else cur


def _extract_edited_path(tool_call: dict, ws_root: Path) -> Path | None:
    args = tool_call.get("args") or {}
    for key in ("Path", "TargetFile", "path", "file_path"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            p = Path(val.strip())
            return p if p.is_absolute() else (ws_root / p)
    return None


def _load_dirty(state_file: Path) -> list[str]:
    if not state_file.is_file():
        return []
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return [str(x) for x in data.get("dirty", [])]
    except Exception:
        return []


def _save_dirty(state_file: Path, paths: list[str]) -> None:
    tmp = state_file.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"dirty": sorted(set(paths))}, indent=2), encoding="utf-8")
    os.replace(tmp, state_file)


def evaluate_pre_tool_use(
    payload: dict,
    *,
    block_leak_hosts: bool = True,
    auto_allow_safe_cmds: bool = False,
) -> dict:
    """Evaluate a PreToolUse event: deny unsafe/leak commands & credential reads."""
    tc = payload.get("toolCall") or {}
    tname = str(tc.get("name") or "")
    args = tc.get("args") or {}

    # Guard file-read tools against direct host credential reads.
    if tname in ("read_file", "view_file", "read_resource"):
        target_path = str(
            args.get("Path") or args.get("AbsolutePath") or args.get("path") or args.get("file_path") or ""
        )
        if target_path and _CREDENTIAL_PATH_RE.search(target_path):
            return {
                "decision": "deny",
                "reason": f"[harness2-gate:CREDENTIAL_ACCESS] Blocked read of credential path: {target_path[:140]}",
            }
        return {}

    cmd = str(args.get("CommandLine") or args.get("command") or "")
    if tname in ("run_command", "bash", "shell") and cmd.strip():
        if _is_destructive_rm(cmd):
            return {
                "decision": "deny",
                "reason": f"[harness2-gate:DESTRUCTIVE_RM] Blocked destructive command: {cmd[:140]}",
            }
        if _DESTRUCTIVE_GIT_PUSH_RE.search(cmd):
            return {
                "decision": "deny",
                "reason": f"[harness2-gate:DESTRUCTIVE_GIT_PUSH] Blocked force-push command: {cmd[:140]}",
            }
        if _CREDENTIAL_PATH_RE.search(cmd):
            return {
                "decision": "deny",
                "reason": f"[harness2-gate:CREDENTIAL_ACCESS] Blocked credential access: {cmd[:140]}",
            }
        if block_leak_hosts and _BENCHMARK_LEAK_HOST_RE.search(cmd):
            return {
                "decision": "deny",
                "reason": (
                    "[harness2-gate:BENCHMARK_LEAK_HOST] Blocked command referencing restricted "
                    "external benchmark host."
                ),
            }
        if auto_allow_safe_cmds and _SAFE_READONLY_CMD_RE.match(cmd) and not _SHELL_META_RE.search(cmd):
            return {"decision": "allow"}
    return {}


def _check_dirty_files(dirty_paths: list[Path]) -> list[GateIssue]:
    issues: list[GateIssue] = []
    for p in dirty_paths:
        if not p.is_file():
            continue
        if p.name in RULE_FILENAMES:
            size = len(p.read_bytes())
            if size > AGY_RULE_MAX_BYTES:
                issues.append(
                    GateIssue(
                        code="RULE_BUDGET_EXCEEDED",
                        target=p.name,
                        message=f"{size} bytes exceeds Antigravity {AGY_RULE_MAX_BYTES}-byte limit",
                    )
                )
        if p.suffix.lower() == ".py":
            try:
                src = p.read_text(encoding="utf-8", errors="replace")
                compile(src, str(p), "exec")
            except (SyntaxError, ValueError) as exc:
                issues.append(
                    GateIssue(
                        code="PY_SYNTAX_ERROR",
                        target=p.name,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
    return issues


def _check_requirements(
    ws_root: Path,
    required_files: list[str],
    required_sections: list[str],
    dirty_paths: list[Path] = (),
) -> list[GateIssue]:
    issues: list[GateIssue] = []
    present_texts: list[str] = []
    for rel in required_files:
        rel_clean = rel.strip()
        if not rel_clean:
            continue
        target = Path(rel_clean)
        if not target.is_absolute():
            target = ws_root / target
        if not target.is_file():
            issues.append(
                GateIssue(
                    code="MISSING_REQUIRED_DELIVERABLE",
                    target=rel_clean,
                    message=f"required deliverable `{rel_clean}` was not written",
                )
            )
            continue
        body = target.read_text(encoding="utf-8", errors="replace")
        if not body.strip():
            issues.append(
                GateIssue(
                    code="EMPTY_DELIVERABLE",
                    target=rel_clean,
                    message=f"required deliverable `{rel_clean}` is empty",
                )
            )
        else:
            present_texts.append(body.lower())

    if required_sections and not required_files:
        for p in dirty_paths:
            if p.is_file() and p.suffix.lower() in (".md", ".txt"):
                txt = p.read_text(encoding="utf-8", errors="replace")
                if txt.strip():
                    present_texts.append(txt.lower())

    if required_sections:
        if not present_texts and not any(i.code == "MISSING_REQUIRED_DELIVERABLE" for i in issues):
            issues.append(
                GateIssue(
                    code="MISSING_REQUIRED_SECTION",
                    target=", ".join(s.strip() for s in required_sections if s.strip()),
                    message="no markdown/text deliverable available to verify required sections",
                )
            )
        elif present_texts:
            corpus = "\n".join(present_texts)
            for sec in required_sections:
                s = sec.strip()
                if s and s.lower() not in corpus:
                    issues.append(
                        GateIssue(
                            code="MISSING_REQUIRED_SECTION",
                            target=s,
                            message=f"required audit/report section `{s}` is missing from deliverables",
                        )
                    )
    return issues


def _run_verify_cmd(ws_root: Path, verify_cmd: str, timeout: int = 20) -> list[GateIssue]:
    if not verify_cmd.strip():
        return []
    try:
        proc = subprocess.run(
            verify_cmd,
            cwd=ws_root,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return [
            GateIssue(
                code="VERIFY_CMD_TIMEOUT",
                target=verify_cmd[:60],
                message=f"verification command timed out after {timeout}s",
            )
        ]
    if proc.returncode != 0:
        tail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()[-300:]
        return [
            GateIssue(
                code="VERIFY_CMD_FAILED",
                target=verify_cmd[:60],
                message=f"exit={proc.returncode}: {tail}",
            )
        ]
    return []


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_sha(ws_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ws_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def _collect_artifact_hashes(
    ws_root: Path, dirty_paths: list[Path], required_files: list[str]
) -> tuple[dict[str, str], dict[str, str], list[GateIssue]]:
    """Hash every verifiable dirty/required artifact on disk; report missing or 0-byte paths."""
    hashes: dict[str, str] = {}
    paths_map: dict[str, str] = {}
    missing_issues: list[GateIssue] = []
    seen: list[Path] = []
    for p in dirty_paths:
        if p not in seen:
            seen.append(p)
    for rel in required_files:
        rel_clean = rel.strip()
        if not rel_clean:
            continue
        p = Path(rel_clean)
        if not p.is_absolute():
            p = ws_root / p
        if p not in seen:
            seen.append(p)

    for p in seen:
        try:
            key = p.resolve().relative_to(ws_root.resolve()).as_posix()
        except ValueError:
            key = p.as_posix()
        if not p.is_file():
            missing_issues.append(
                GateIssue(
                    code="MISSING_ARTIFACT",
                    target=key,
                    message=f"tracked artifact `{key}` is missing on disk (fail-closed)",
                )
            )
            continue
        if p.stat().st_size == 0:
            missing_issues.append(
                GateIssue(
                    code="EMPTY_ARTIFACT",
                    target=key,
                    message=f"tracked artifact `{key}` is 0 bytes on disk (fail-closed)",
                )
            )
            continue
        hashes[key] = _sha256_file(p)
        paths_map[key] = str(p.resolve())
    return hashes, paths_map, missing_issues


def evaluate_stop(
    payload: dict,
    *,
    dirty_paths: list[Path],
    ws_root: Path,
    required_files: list[str] = (),
    required_sections: list[str] = (),
    verify_cmd: str = "",
    max_continuations: int = 2,
    require_artifacts: bool = False,
    receipt_out: dict | None = None,
) -> tuple[dict, str]:
    """Evaluate Stop hook; returns `(hook_output_dict, user_badge_markdown)`.

    Always runs deterministic verification (including on `continuationActive=True` retry turns)
    so `receipt_out` reflects the true `PASS`/`FAIL` state. Loop-cap guards only bound whether
    another `{"decision": "continue"}` is emitted, preventing both infinite loops and unverified
    passes.
    """
    term_reason = str(payload.get("terminationReason") or "")
    cont_active = bool(payload.get("continuationActive", False))
    exec_num = int(payload.get("executionNum") or 0)

    commit_sha = _git_commit_sha(ws_root)
    input_digest = hashlib.sha256(
        json.dumps(
            {
                "required_files": sorted(str(f).strip() for f in required_files if str(f).strip()),
                "required_sections": sorted(
                    str(s).strip() for s in required_sections if str(s).strip()
                ),
                "verify_cmd": verify_cmd.strip(),
                "last_user_input": str(payload.get("lastUserInput") or ""),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    if term_reason in ("USER_CANCELED", "ERROR", "MAX_INVOCATIONS", "MAX_FORCED_INVOCATIONS"):
        if receipt_out is not None:
            receipt_out.clear()
            receipt_out.update(
                {
                    "schema_version": "1",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "ABORTED",
                    "termination_reason": term_reason,
                    "commit_sha": commit_sha,
                    "command": verify_cmd.strip(),
                    "input_digest": input_digest,
                    "artifact_hashes": {},
                    "artifact_paths": {},
                    "issues": [],
                }
            )
        return {"decision": "allow"}, ""

    issues: list[GateIssue] = []
    issues.extend(_check_dirty_files(dirty_paths))
    issues.extend(
        _check_requirements(ws_root, list(required_files), list(required_sections), list(dirty_paths))
    )
    artifact_hashes, artifact_paths, missing_art_issues = _collect_artifact_hashes(
        ws_root, list(dirty_paths), list(required_files)
    )
    if require_artifacts:
        already_missing = {
            i.target
            for i in issues
            if i.code in ("MISSING_REQUIRED_DELIVERABLE", "EMPTY_DELIVERABLE")
        }
        for miss in missing_art_issues:
            if miss.target not in already_missing:
                issues.append(miss)
        if not artifact_hashes and not issues:
            issues.append(
                GateIssue(
                    code="MISSING_ARTIFACT",
                    target="workspace",
                    message="no verifiable artifact found on disk (fail-closed)",
                )
            )
    if dirty_paths or required_files or verify_cmd.strip():
        issues.extend(_run_verify_cmd(ws_root, verify_cmd))

    receipt_data: dict = {
        "schema_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not issues else "FAIL",
        "commit_sha": commit_sha,
        "command": verify_cmd.strip(),
        "input_digest": input_digest,
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "artifact_paths": dict(sorted(artifact_paths.items())),
        "issues": [
            {"code": i.code, "target": i.target, "message": i.message} for i in issues
        ],
    }
    hmac_key = os.environ.get("HARNESS2_RECEIPT_HMAC_KEY", "")
    if hmac_key:
        canonical = json.dumps(receipt_data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        receipt_data["signature_hmac_sha256"] = hmac.new(
            hmac_key.encode("utf-8"), canonical, hashlib.sha256
        ).hexdigest()

    if receipt_out is not None:
        receipt_out.clear()
        receipt_out.update(receipt_data)

    if not issues:
        checked = len(artifact_hashes)
        commit_tag = f"commit `{commit_sha[:8]}`, " if commit_sha else ""
        badge = (
            f"**[harness2 c8 gate: PASS]** {checked} artifact(s) hashed & verified "
            f"({commit_tag}input `{input_digest[:12]}`, receipt `gate_receipt.json`, 0 LLM tokens)."
            if checked
            else ""
        )
        return {"decision": "allow"}, badge

    # Loop-cap guard: once continuation budget is exhausted, stop looping WITHOUT falsifying PASS.
    if exec_num >= max_continuations or (cont_active and exec_num >= max(1, max_continuations)):
        return {"decision": "allow"}, ""

    compact_lines = [
        f"[harness2-completion-gate FAIL ({len(issues)} issue(s)); fix before finishing]:"
    ]
    for iss in issues[:8]:
        compact_lines.append(f"- {iss.code} ({iss.target}): {iss.message}")
    if len(issues) > 8:
        compact_lines.append(f"- ... (+{len(issues) - 8} more)")
    return {"decision": "continue", "reason": "\n".join(compact_lines)}, ""


def build_hooks_json(python_exe: str = "python") -> dict:
    """Return a valid Antigravity `.agents/hooks.json` wiring the deterministic completion gate."""
    base_cmd = f"{python_exe} -m harness2.gates.completion_gate"
    return {
        "harness2-completion-gate": {
            "PreToolUse": [
                {
                    "matcher": "run_command|read_file|view_file",
                    "hooks": [{"type": "command", "command": f"{base_cmd} PreToolUse", "timeout": 10}],
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "write_file|edit_file|write_to_file|replace_file_content|multi_replace_file_content",
                    "hooks": [{"type": "command", "command": f"{base_cmd} PostToolUse", "timeout": 10}],
                }
            ],
            "PreInvocation": [
                {"type": "command", "command": f"{base_cmd} PreInvocation", "timeout": 10}
            ],
            "Stop": [
                {"type": "command", "command": f"{base_cmd} Stop", "timeout": 25}
            ],
        }
    }


def _csv_env(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    event = args[0] if args else "unknown"
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {}

    sdir = _state_dir(payload)
    dirty_file = sdir / "dirty_state.json"
    badge_file = sdir / "last_badge.json"
    ws_root = _workspace_root(payload)

    if event == "PreToolUse":
        out = evaluate_pre_tool_use(
            payload,
            block_leak_hosts=os.environ.get("HARNESS2_BLOCK_LEAK_HOSTS", "1") != "0",
            auto_allow_safe_cmds=os.environ.get("HARNESS2_AUTO_ALLOW_SAFE", "0") == "1",
        )
        print(json.dumps(out, ensure_ascii=False))
        return 0

    if event == "PostToolUse":
        if not payload.get("error"):
            target = _extract_edited_path(payload.get("toolCall") or {}, ws_root)
            if target is not None and target.exists():
                cur = _load_dirty(dirty_file)
                cur.append(str(target))
                _save_dirty(dirty_file, cur)
        print("{}")
        return 0

    if event == "PreInvocation":
        if badge_file.is_file():
            try:
                out = json.loads(badge_file.read_text(encoding="utf-8"))
                badge_file.unlink(missing_ok=True)
                print(json.dumps(out, ensure_ascii=False))
                return 0
            except Exception:
                pass
        print("{}")
        return 0

    if event == "Stop":
        dirty_paths = [Path(p) for p in _load_dirty(dirty_file)]
        max_cont = int(os.environ.get("HARNESS2_MAX_CONTINUATIONS", "2"))
        req_artifacts = os.environ.get("HARNESS2_REQUIRE_ARTIFACTS", "0") == "1"
        receipt_data: dict = {}
        out, badge = evaluate_stop(
            payload,
            dirty_paths=dirty_paths,
            ws_root=ws_root,
            required_files=_csv_env("HARNESS2_REQUIRED_FILES"),
            required_sections=_csv_env("HARNESS2_REQUIRED_SECTIONS"),
            verify_cmd=os.environ.get("HARNESS2_VERIFY_CMD", ""),
            max_continuations=max_cont,
            require_artifacts=req_artifacts,
            receipt_out=receipt_data,
        )
        # Clear dirty queue ONLY on PASS (or hard abort), never on FAIL!
        if receipt_data.get("status") in ("PASS", "ABORTED"):
            _save_dirty(dirty_file, [])
        if receipt_data:
            receipt_path = Path(
                os.environ.get("HARNESS2_RECEIPT_PATH") or (sdir / "gate_receipt.json")
            )
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = receipt_path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(
                json.dumps(receipt_data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, receipt_path)
        if badge:
            badge_payload = {
                "injectSteps": [
                    {
                        "systemMessage": {
                            "systemMessage": "",
                            "metadata": {"user_markdown": badge},
                        }
                    }
                ]
            }
            badge_file.write_text(json.dumps(badge_payload, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(out, ensure_ascii=False))
        return 0

    print("{}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
