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

"""Zero-token Tier 0 evidence-grounding gate, replayable receipts, and static security scanner.

Two deterministic, stdlib-only guards:
1. `verify_experience_evidence` + `EvidenceReceipt` + `verify_receipt`: checks that a
   reflection's `<experience>` cites real live rollout labels or changed harness components,
   fails closed on missing or 0-byte delivered artifacts, emits a content-bound receipt
   (`commit_sha`, `command`, `input_digest`, `artifact_hashes`, plus optional HMAC-SHA256
   authentication when `HARNESS2_RECEIPT_HMAC_KEY` is configured), and ships a runnable CLI
   (`python -m harness2.core.evidence_gate verify <receipt.json>`).
2. `scan_harness_security`: checks ALL evolved harness files (including Markdown prompt/skill
   surfaces) for hidden Unicode payloads and credential/benchmark exfiltration patterns (#96).
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

# Explicit rollout prefixes (`base1`, `inv2`, `dom1`, `agg1`); excludes radix/encoding terms.
_EXPLICIT_ROLLOUT_RE = re.compile(r"\b(?:base|inv|dom|agg)\d+\b", re.IGNORECASE)
_EXCLUDED_BASE_TOKENS = frozenset(
    {"base2", "base8", "base10", "base16", "base32", "base58", "base64", "base85"}
)
# Sequential round labels (`c1`, `c2`, ...) only count when in rollout context or `valid_labels`
# so Harness²'s own component taxonomy (`c1..c8`) and spreadsheet cells (`C2`) never false-positive.
_CONTEXTUAL_C_ROLLOUT_RE = re.compile(
    r"\b(?:rollout|candidate|compose|round|final|selected|adopted)\s+(c\d+)\b",
    re.IGNORECASE,
)
_GENERIC_BASENAMES = frozenset(
    {
        "skill.md",
        "agent.md",
        "readme.md",
        "gemini.md",
        "agents.md",
        "hooks.json",
        "systemprompt.md",
        "guardrails.md",
    }
)

_HIDDEN_UNICODE_RANGES = (
    (0x200B, 0x200F, "ZERO_WIDTH_OR_DIRECTIONAL_MARK"),
    (0x202A, 0x202E, "BIDI_OVERRIDE"),
    (0x2060, 0x2060, "WORD_JOINER"),
    (0x2061, 0x2064, "INVISIBLE_OPERATOR"),
    (0x2066, 0x2069, "BIDI_ISOLATE"),
    (0xFE00, 0xFE0F, "VARIATION_SELECTOR"),
    (0xFEFF, 0xFEFF, "ZERO_WIDTH_BOM"),
    (0xE0000, 0xE007F, "UNICODE_TAGS_BLOCK"),
    (0xE0100, 0xE01EF, "VARIATION_SELECTOR_SUPPLEMENT"),
)

_EXFIL_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "BENCHMARK_LEAK_HOST",
        re.compile(
            r"\b(?:huggingface\.co|hf\.co|raw\.githubusercontent\.com|api\.github\.com/repos)\b",
            re.IGNORECASE,
        ),
        "references a benchmark-hosting domain that may leak grading rubrics",
    ),
    (
        "CREDENTIAL_PATH_ACCESS",
        re.compile(
            r"(?:~|\$\{?HOME\}?|%USERPROFILE%|/root|/home/[^/\s]+|/Users/[^/\s]+|"
            r"[A-Za-z]:[/\\]Users[/\\][^/\\\s]+)"
            r"(?:[/\\]\.(?:ssh|aws|gcloud|netrc)|[/\\]\.config[/\\]gcloud|"
            r"[/\\]\.gemini[/\\]antigravity-cli[/\\]settings\.json)\b",
            re.IGNORECASE,
        ),
        "references a host credential directory inside a harness file",
    ),
    (
        "SHELL_EXFIL_COMMAND",
        re.compile(
            r"\b(?:curl|wget)\b[^\n]{0,160}"
            r"(?:\$\{?[A-Z_]*(?:KEY|TOKEN|SECRET|CREDENTIAL)\}?|os\.environ|printenv)",
            re.IGNORECASE,
        ),
        "potential credential exfiltration via curl/wget",
    ),
)

_NET_CLIENT_RE = re.compile(
    r"(?:\b(?:urllib\.request|from\s+urllib\s+import\s+request|requests\.(?:post|put|get)|"
    r"axios\.(?:post|put|get|request)|http\.client|socket\.create_connection|"
    r"socket\.socket|curl|wget)\b|\b(?:fetch|urlopen)\s*\()",
    re.IGNORECASE,
)
_SECRET_READ_RE = re.compile(
    r"(?:\b(?:os\.environ|os\.getenv|printenv|process\.env|Bun\.env|Deno\.env)\b|"
    r"\$\{?[A-Z_]*(?:KEY|TOKEN|SECRET|CREDENTIAL)\}?)",
    re.IGNORECASE,
)

_TOP_LEVEL_JUDGE_FILE_RE = re.compile(
    r"^(?:judge|grade|score|scores|eval)[a-z0-9_.-]*\.(?:json|jsonl|log|txt)$",
    re.IGNORECASE,
)
_VERIFIER_MUTABLE_PATHS = frozenset({"verifier/score.json", "verifier/eval_result.json"})


@dataclass(frozen=True)
class EvidenceReceipt:
    """Replayable content-bound receipt (optionally HMAC-SHA256 authenticated)."""

    status: str
    commit_sha: str
    command: str
    input_digest: str
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    verified_refs: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "1"
    signature_hmac_sha256: str = ""

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "status": self.status,
            "commit_sha": self.commit_sha,
            "command": self.command,
            "input_digest": self.input_digest,
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "artifact_paths": dict(sorted(self.artifact_paths.items())),
            "verified_refs": list(self.verified_refs),
            "issues": list(self.issues),
        }
        if self.signature_hmac_sha256:
            d["signature_hmac_sha256"] = self.signature_hmac_sha256
        return d


@dataclass(frozen=True)
class EvidenceCheckResult:
    """Outcome of deterministic Tier 0 citation verification on a reflection experience."""

    ok: bool
    verified_refs: tuple[str, ...]
    issues: tuple[str, ...]
    receipt: EvidenceReceipt | None = None


@dataclass(frozen=True)
class SecurityFinding:
    """One static security finding on an evolved harness tree."""

    file_rel: str
    line: int
    code: str
    message: str


def sha256_file(path: Path | str) -> str:
    """Return hex SHA-256 digest of a file's raw bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_immutable_rollout_file(rel_posix: str) -> bool:
    """Exclude only post-hoc Harness² judge/verifier metadata; never skip solver deliverables."""
    norm = rel_posix.strip("/")
    if norm.lower() in _VERIFIER_MUTABLE_PATHS:
        return False
    if "/" not in norm and _TOP_LEVEL_JUDGE_FILE_RE.match(norm):
        return False
    return True


def sha256_path(path: Path | str, *, exclude_judge_files: bool = False) -> str:
    """Return hex SHA-256 digest of a non-empty file or canonical sorted directory manifest.

    Returns empty string `""` for missing paths, 0-byte empty files, or empty directories so
    callers enforcing fail-closed semantics never pass on empty artifacts.
    """
    p = Path(path)
    if p.is_file():
        if p.stat().st_size == 0:
            return ""
        return sha256_file(p)
    if p.is_dir():
        h = hashlib.sha256()
        files = sorted(
            f
            for f in p.rglob("*")
            if f.is_file()
            and "__pycache__" not in f.parts
            and f.stat().st_size > 0
            and (
                not exclude_judge_files
                or _is_immutable_rollout_file(f.relative_to(p).as_posix())
            )
        )
        if not files:
            return ""
        for f in files:
            rel = f.relative_to(p).as_posix()
            h.update(f"{rel}:{sha256_file(f)}\n".encode("utf-8"))
        return h.hexdigest()
    return ""


def sha256_mapping(mapping: Mapping[str, object]) -> str:
    """Deterministic SHA-256 digest of a JSON-serializable input mapping."""
    payload = json.dumps(mapping, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_receipt_hmac(receipt: Mapping[str, object], secret_key: str) -> str:
    """Compute HMAC-SHA256 over the canonical receipt fields (excluding any existing signature)."""
    body = {k: v for k, v in receipt.items() if k != "signature_hmac_sha256"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hmac.new(secret_key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()


def git_commit_sha(cwd: Path | str | None = None) -> str:
    """Return current git HEAD commit SHA if inside a git repository, else empty string."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
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


def verify_receipt(
    receipt: Mapping[str, object],
    *,
    artifact_paths: Mapping[str, Path | str] | None = None,
    expected_input_digest: str = "",
    expected_commit_sha: str = "",
    hmac_key: str = "",
    exclude_judge_files: bool = False,
) -> tuple[bool, list[str]]:
    """Deterministically verify a saved receipt against on-disk artifacts (fail-closed)."""
    errs: list[str] = []
    if str(receipt.get("status") or "").upper() != "PASS":
        errs.append(f"receipt status is {receipt.get('status')!r}, not 'PASS' (fail-closed)")
    recorded_hashes = receipt.get("artifact_hashes")
    if not isinstance(recorded_hashes, Mapping) or not recorded_hashes:
        errs.append("receipt has no recorded artifact_hashes (fail-closed)")
        return False, errs
    if expected_input_digest and receipt.get("input_digest") != expected_input_digest:
        errs.append(
            f"input_digest mismatch: receipt={receipt.get('input_digest')!r} "
            f"!= expected={expected_input_digest!r}"
        )
    if expected_commit_sha and receipt.get("commit_sha") != expected_commit_sha:
        errs.append(
            f"commit_sha mismatch: receipt={receipt.get('commit_sha')!r} "
            f"!= expected={expected_commit_sha!r}"
        )
    key = hmac_key or os.environ.get("HARNESS2_RECEIPT_HMAC_KEY", "")
    sig = str(receipt.get("signature_hmac_sha256") or "")
    if key:
        expected_sig = compute_receipt_hmac(receipt, key)
        if not sig or not hmac.compare_digest(sig, expected_sig):
            errs.append("receipt HMAC-SHA256 signature missing or invalid (fail-closed)")
    elif sig:
        errs.append(
            "receipt carries HMAC-SHA256 signature but no HMAC key was supplied to verify it (fail-closed)"
        )

    resolved_paths: dict[str, Path] = {}
    stored_paths = receipt.get("artifact_paths")
    if isinstance(stored_paths, Mapping):
        for k, v in stored_paths.items():
            resolved_paths[str(k)] = Path(str(v))
    if artifact_paths is not None:
        for k, v in artifact_paths.items():
            resolved_paths[str(k)] = Path(v)

    for key_name, expected_hash in sorted(recorded_hashes.items()):
        path = resolved_paths.get(str(key_name))
        if path is None:
            errs.append(f"artifact path for `{key_name}` not supplied (fail-closed)")
            continue
        if not path.exists():
            errs.append(f"artifact `{key_name}` missing on disk at {path} (fail-closed)")
            continue
        actual_hash = sha256_path(path, exclude_judge_files=exclude_judge_files)
        if not actual_hash or actual_hash != str(expected_hash):
            errs.append(
                f"artifact `{key_name}` hash mismatch: actual={actual_hash!r} != {expected_hash!r}"
            )

    # Also enforce fail-closed 0-byte integrity for promoted placeholder files (`empty_files`).
    empty_files = receipt.get("empty_files")
    if isinstance(empty_files, Sequence) and not isinstance(empty_files, (str, bytes)):
        empty_paths_map = receipt.get("empty_file_paths")
        resolved_empty: dict[str, Path] = {}
        if isinstance(empty_paths_map, Mapping):
            for k, v in empty_paths_map.items():
                resolved_empty[str(k)] = Path(str(v))
        for k, v in resolved_paths.items():
            resolved_empty.setdefault(k, v)
        for empty_name in sorted(str(x) for x in empty_files):
            epath = resolved_empty.get(empty_name)
            if epath is None:
                errs.append(f"empty artifact path for `{empty_name}` not supplied (fail-closed)")
                continue
            if not epath.is_file():
                errs.append(f"empty artifact `{empty_name}` missing on disk at {epath} (fail-closed)")
                continue
            sz = epath.stat().st_size
            if sz != 0:
                errs.append(
                    f"empty artifact `{empty_name}` was modified on disk (size={sz} != 0, fail-closed)"
                )
    return (not errs), errs


def _extract_cited_rollout_labels(text: str, valid_set: Mapping[str, str]) -> list[str]:
    found: set[str] = set()
    for m in _EXPLICIT_ROLLOUT_RE.finditer(text):
        tok = m.group(0).lower()
        if tok not in _EXCLUDED_BASE_TOKENS:
            found.add(tok)
    for m in _CONTEXTUAL_C_ROLLOUT_RE.finditer(text):
        found.add(m.group(1).lower())
    # Also recognize any valid sequential round label `c1..ck` explicitly present in `valid_set`.
    for lbl_low in valid_set:
        if re.fullmatch(r"c\d+", lbl_low) and re.search(rf"\b{re.escape(lbl_low)}\b", text, re.I):
            found.add(lbl_low)
    return sorted(found)


def verify_experience_evidence(
    experience: Mapping[str, str],
    *,
    valid_labels: Sequence[str],
    changed_components: Sequence[str] = (),
    live_labels: Sequence[str] | None = None,
    delivered_labels: Sequence[str] | None = None,
    artifact_paths: Mapping[str, Path | str] | None = None,
    require_artifacts: bool = False,
    input_digest: str = "",
    commit_sha: str = "",
    command: str = "",
    hmac_key: str = "",
    exclude_judge_files: bool = False,
) -> EvidenceCheckResult:
    """Verify `<experience>` against real rollouts, changed components, and on-disk artifact hashes.

    When `require_artifacts=True` (or `artifact_paths` is supplied), fails closed if any cited
    delivered rollout's artifact is missing or 0-byte empty on disk, and binds the verdict into a
    replayable `EvidenceReceipt`.
    """
    evidence_text = str(experience.get("evidence") or "").strip()
    edits_text = str(experience.get("edits") or "").strip()
    combined = f"{evidence_text}\n{edits_text}".strip()
    computed_input = input_digest or sha256_mapping(
        {
            "valid_labels": sorted(valid_labels),
            "changed_components": sorted(changed_components),
            "experience": dict(experience),
        }
    )

    def _build_receipt(
        ok_flag: bool,
        verified_list: list[str],
        issues_list: list[str],
        hashes: dict[str, str],
        paths_map: dict[str, str],
    ) -> EvidenceReceipt:
        rec = EvidenceReceipt(
            status="PASS" if ok_flag else "FAIL",
            commit_sha=commit_sha,
            command=command,
            input_digest=computed_input,
            artifact_hashes=hashes,
            artifact_paths=paths_map,
            verified_refs=tuple(sorted(set(verified_list))),
            issues=tuple(issues_list),
        )
        key = hmac_key or os.environ.get("HARNESS2_RECEIPT_HMAC_KEY", "")
        if key:
            sig = compute_receipt_hmac(rec.to_dict(), key)
            rec = EvidenceReceipt(**{**rec.__dict__, "signature_hmac_sha256": sig})
        return rec

    if not evidence_text:
        issues_empty = ["empty evidence field"]
        rec = _build_receipt(False, [], issues_empty, {}, {})
        return EvidenceCheckResult(ok=False, verified_refs=(), issues=tuple(issues_empty), receipt=rec)

    valid_set = {lbl.lower(): lbl for lbl in valid_labels}
    live_set = (
        {lbl.lower() for lbl in live_labels}
        if live_labels is not None
        else set(valid_set.keys())
    )
    delivered_set = (
        {lbl.lower() for lbl in delivered_labels}
        if delivered_labels is not None
        else set(live_set)
    )
    changed_norm = [c.replace("\\", "/") for c in changed_components if c]

    cited_labels = _extract_cited_rollout_labels(combined, valid_set)
    issues_list: list[str] = []
    verified: list[str] = []
    artifact_hashes: dict[str, str] = {}
    recorded_paths: dict[str, str] = {}

    for lbl in cited_labels:
        if lbl not in valid_set:
            issues_list.append(f"fabricated rollout reference `{lbl}` not present in task record")
        elif lbl not in live_set:
            issues_list.append(f"cited rollout `{valid_set[lbl]}` did not complete with live evidence")
        else:
            verified.append(f"rollout:{valid_set[lbl]}")

    low_combined = combined.lower()
    for comp in changed_norm:
        comp_low = comp.lower()
        basename = comp_low.rsplit("/", 1)[-1]
        if comp_low in low_combined or (
            len(basename) >= 5
            and basename not in _GENERIC_BASENAMES
            and basename in low_combined
        ):
            verified.append(f"component:{comp}")

    if not verified:
        issues_list.append(
            "evidence cites neither a real rollout label nor a modified harness component "
            f"(valid rollouts: {sorted(valid_labels)[:8]}; changed: {changed_norm[:6]})"
        )

    # Fail-closed on-disk artifact hash binding for delivered rollouts.
    if require_artifacts or artifact_paths is not None:
        norm_artifacts = {str(k).lower(): (str(k), Path(v)) for k, v in (artifact_paths or {}).items()}
        if require_artifacts and not norm_artifacts:
            issues_list.append("no on-disk artifact paths supplied (fail-closed)")
        else:
            cited_delivered = [
                lbl for lbl in cited_labels if lbl in valid_set and lbl in delivered_set
            ]
            targets_to_hash: list[tuple[str, Path | None]] = []
            if cited_delivered:
                for lbl in cited_delivered:
                    entry = norm_artifacts.get(lbl)
                    targets_to_hash.append((valid_set[lbl], entry[1] if entry else None))
            else:
                # Contrast lessons citing only non-delivered base + component: hash delivered artifacts.
                for low_k, (orig_k, path) in sorted(norm_artifacts.items()):
                    if low_k in delivered_set or not valid_set:
                        targets_to_hash.append((orig_k, path))

            if require_artifacts and not targets_to_hash:
                issues_list.append("no delivered artifact available to bind receipt (fail-closed)")

            for key_name, path in targets_to_hash:
                if path is None or not path.exists():
                    issues_list.append(f"artifact for `{key_name}` is missing on disk (fail-closed)")
                    continue
                digest = sha256_path(path, exclude_judge_files=exclude_judge_files)
                if not digest:
                    issues_list.append(
                        f"artifact for `{key_name}` at {path} is empty (fail-closed)"
                    )
                    continue
                artifact_hashes[key_name] = digest
                recorded_paths[key_name] = str(path)

    ok = not issues_list
    receipt = _build_receipt(ok, verified, issues_list, artifact_hashes, recorded_paths)
    return EvidenceCheckResult(
        ok=ok,
        verified_refs=receipt.verified_refs,
        issues=receipt.issues,
        receipt=receipt,
    )


def _hidden_unicode_code(ch: str) -> str | None:
    cp = ord(ch)
    for lo, hi, label in _HIDDEN_UNICODE_RANGES:
        if lo <= cp <= hi:
            return label
    return None


def scan_harness_security(tree: Mapping[str, str]) -> list[SecurityFinding]:
    """Static pre-adoption scan across ALL harness files (Markdown + executable surfaces) (#96)."""
    findings: list[SecurityFinding] = []
    for rel in sorted(tree):
        norm = rel.replace("\\", "/")
        body = str(tree.get(rel) or "")
        for line_no, line in enumerate(body.splitlines(), 1):
            seen_ucodes: set[str] = set()
            for ch in line:
                ucode = _hidden_unicode_code(ch)
                if ucode is not None and ucode not in seen_ucodes:
                    seen_ucodes.add(ucode)
                    findings.append(
                        SecurityFinding(
                            file_rel=norm,
                            line=line_no,
                            code=ucode,
                            message=f"hidden Unicode code point U+{ord(ch):04X} ({ucode})",
                        )
                    )
            norm_line = re.sub(r"\\(?:\.\\)+", r"\\", re.sub(r"/(?:\./)+", "/", re.sub(r"/{2,}", "/", line)))
            for code, rx, desc in _EXFIL_PATTERNS:
                if rx.search(line) or rx.search(norm_line):
                    findings.append(
                        SecurityFinding(
                            file_rel=norm,
                            line=line_no,
                            code=code,
                            message=desc,
                        )
                    )
        # File-level multi-line exfiltration check strictly on code/executable surfaces
        # (avoids false positives on Markdown prose mentioning curl/os.environ in rules).
        is_code_file = (
            norm.endswith((".py", ".sh", ".ts", ".js", ".ps1", ".bat", ".cmd"))
            or norm in (".agents/hooks.json", ".codex/hooks.json")
            or norm.startswith(("scripts/", "plugin/"))
            or "/scripts/" in norm
        )
        if is_code_file and _NET_CLIENT_RE.search(body) and _SECRET_READ_RE.search(body):
            findings.append(
                SecurityFinding(
                    file_rel=norm,
                    line=1,
                    code="MULTILINE_CREDENTIAL_EXFIL",
                    message="file combines host environment/secret reads with outbound network client",
                )
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for replaying/verifying a saved evidence receipt (`verify <receipt.json>`)."""
    p = argparse.ArgumentParser(prog="python -m harness2.core.evidence_gate")
    sub = p.add_subparsers(dest="cmd", required=True)
    vp = sub.add_parser("verify", help="Verify a saved evidence_receipt.json against disk")
    vp.add_argument("receipt", type=Path, help="Path to evidence_receipt.json or promotion_manifest.json")
    vp.add_argument("--exclude-judge-files", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "verify":
        data = json.loads(args.receipt.read_text(encoding="utf-8"))
        ok, errs = verify_receipt(data, exclude_judge_files=args.exclude_judge_files)
        if ok:
            print(f"PASS: verified {len(data.get('artifact_hashes') or {})} artifact(s)")
            return 0
        for e in errs:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
