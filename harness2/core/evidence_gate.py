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

"""Zero-token Tier 0 evidence-grounding gate and static security scanner for evolved harnesses.

Two deterministic, zero-dependency guards:
1. `verify_experience_evidence`: checks that a reflection's `<experience>` cites actual
   rollout labels or changed harness components from the task record instead of fabricated
   attributions or dead rollouts (#71, #143).
2. `scan_harness_security`: checks an evolved harness tree for hidden Unicode payloads
   (zero-width chars, BiDi overrides, Unicode Tags block) and risky network/credential
   exfiltration patterns before adoption or promotion (#96).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

_ROLLOUT_LABEL_RE = re.compile(r"\b(?:base|inv|dom|agg|c)\d+\b", re.IGNORECASE)

_HIDDEN_UNICODE_RANGES = (
    (0x200B, 0x200F, "ZERO_WIDTH_OR_DIRECTIONAL_MARK"),
    (0x202A, 0x202E, "BIDI_OVERRIDE"),
    (0x2066, 0x2069, "BIDI_ISOLATE"),
    (0xE0000, 0xE007F, "UNICODE_TAGS_BLOCK"),
)

_EXFIL_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "BENCHMARK_LEAK_HOST",
        re.compile(r"\b(?:huggingface\.co|hf\.co|raw\.githubusercontent\.com)\b", re.IGNORECASE),
        "references a benchmark-hosting domain that may leak grading rubrics",
    ),
    (
        "CREDENTIAL_PATH_ACCESS",
        re.compile(r"(?:~|/root|/home/[^/\s]+)/\.(?:ssh|aws|gcloud|netrc)\b", re.IGNORECASE),
        "references a host credential directory inside a harness file",
    ),
    (
        "SHELL_EXFIL_COMMAND",
        re.compile(
            r"\b(?:curl|wget)\b[^\n]{0,160}(?:\$[A-Z_]*KEY|\$[A-Z_]*TOKEN|os\.environ|printenv)",
            re.IGNORECASE,
        ),
        "potential credential exfiltration via curl/wget",
    ),
)

_EXEC_SURFACE_PREFIXES = ("scripts/", "plugin/", ".codex/", ".agents/hooks.json")


@dataclass(frozen=True)
class EvidenceReceipt:
    """Replayable cryptographic receipt binding an evidence gate verdict to exact inputs & artifacts."""

    status: str
    commit_sha: str
    command: str
    input_digest: str
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    verified_refs: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "1"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "status": self.status,
            "commit_sha": self.commit_sha,
            "command": self.command,
            "input_digest": self.input_digest,
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "verified_refs": list(self.verified_refs),
            "issues": list(self.issues),
        }


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


def sha256_path(path: Path | str) -> str:
    """Return hex SHA-256 digest of a file, or canonical sorted manifest digest of a directory."""
    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    if p.is_dir():
        h = hashlib.sha256()
        files = sorted(f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
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
    artifact_paths: Mapping[str, Path | str],
    expected_input_digest: str = "",
    expected_commit_sha: str = "",
) -> tuple[bool, list[str]]:
    """Deterministically verify a saved receipt against on-disk artifacts (fail-closed)."""
    errs: list[str] = []
    recorded_hashes = receipt.get("artifact_hashes")
    if not isinstance(recorded_hashes, Mapping) or not recorded_hashes:
        return False, ["receipt has no recorded artifact_hashes (fail-closed)"]
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
    for key, expected_hash in sorted(recorded_hashes.items()):
        path = artifact_paths.get(str(key))
        if path is None:
            errs.append(f"artifact path for `{key}` not supplied (fail-closed)")
            continue
        p = Path(path)
        if not p.exists():
            errs.append(f"artifact `{key}` missing on disk at {p} (fail-closed)")
            continue
        actual_hash = sha256_path(p)
        if not actual_hash or actual_hash != str(expected_hash):
            errs.append(f"artifact `{key}` hash mismatch: actual={actual_hash!r} != {expected_hash!r}")
    return (not errs), errs


def verify_experience_evidence(
    experience: Mapping[str, str],
    *,
    valid_labels: Sequence[str],
    changed_components: Sequence[str] = (),
    live_labels: Sequence[str] | None = None,
    artifact_paths: Mapping[str, Path | str] | None = None,
    require_artifacts: bool = False,
    input_digest: str = "",
    commit_sha: str = "",
    command: str = "",
) -> EvidenceCheckResult:
    """Verify `<experience>` against real rollouts, changed components, and on-disk artifact hashes.

    When `require_artifacts=True` (or `artifact_paths` is supplied), fails closed if any cited
    rollout's artifact is missing or empty on disk, and binds the verdict into a replayable
    `EvidenceReceipt` (`commit_sha`, `command`, `input_digest`, `artifact_hashes`).
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

    if not evidence_text:
        issues = ("empty evidence field",)
        return EvidenceCheckResult(
            ok=False,
            verified_refs=(),
            issues=issues,
            receipt=EvidenceReceipt(
                status="FAIL",
                commit_sha=commit_sha,
                command=command,
                input_digest=computed_input,
                issues=issues,
            ),
        )

    valid_set = {lbl.lower(): lbl for lbl in valid_labels}
    live_set = (
        {lbl.lower() for lbl in live_labels}
        if live_labels is not None
        else set(valid_set.keys())
    )
    changed_norm = [c.replace("\\", "/") for c in changed_components if c]

    cited_labels = sorted({m.group(0).lower() for m in _ROLLOUT_LABEL_RE.finditer(combined)})
    issues_list: list[str] = []
    verified: list[str] = []
    artifact_hashes: dict[str, str] = {}

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
        if comp_low in low_combined or (len(basename) >= 5 and basename in low_combined):
            verified.append(f"component:{comp}")

    if not verified:
        issues_list.append(
            "evidence cites neither a real rollout label nor a modified harness component "
            f"(valid rollouts: {sorted(valid_labels)[:8]}; changed: {changed_norm[:6]})"
        )

    # Fail-closed on-disk artifact hash binding when artifacts are required or supplied.
    if require_artifacts or artifact_paths is not None:
        norm_artifacts = {str(k).lower(): (str(k), Path(v)) for k, v in (artifact_paths or {}).items()}
        if require_artifacts and not norm_artifacts:
            issues_list.append("no on-disk artifact paths supplied (fail-closed)")
        else:
            # Check every cited live rollout; if none cited, check all supplied artifacts.
            targets_to_hash: list[tuple[str, Path | None]] = []
            cited_live = [lbl for lbl in cited_labels if lbl in valid_set and lbl in live_set]
            if cited_live:
                for lbl in cited_live:
                    entry = norm_artifacts.get(lbl)
                    targets_to_hash.append((valid_set[lbl], entry[1] if entry else None))
            else:
                for orig_k, path in sorted(norm_artifacts.values()):
                    targets_to_hash.append((orig_k, path))

            for key, path in targets_to_hash:
                if path is None or not path.exists():
                    issues_list.append(f"artifact for `{key}` is missing on disk (fail-closed)")
                    continue
                digest = sha256_path(path)
                if not digest:
                    issues_list.append(f"artifact for `{key}` at {path} is empty (fail-closed)")
                    continue
                artifact_hashes[key] = digest

    ok = not issues_list
    receipt = EvidenceReceipt(
        status="PASS" if ok else "FAIL",
        commit_sha=commit_sha,
        command=command,
        input_digest=computed_input,
        artifact_hashes=artifact_hashes,
        verified_refs=tuple(sorted(set(verified))),
        issues=tuple(issues_list),
    )
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
    """Static pre-adoption scan for hidden Unicode instructions and exfiltration patterns (#96)."""
    findings: list[SecurityFinding] = []
    for rel in sorted(tree):
        norm = rel.replace("\\", "/")
        body = str(tree.get(rel) or "")
        is_exec_file = (
            norm == ".agents/hooks.json"
            or norm == ".codex/hooks.json"
            or norm.startswith(_EXEC_SURFACE_PREFIXES)
            or "/scripts/" in norm
        )
        for line_no, line in enumerate(body.splitlines(), 1):
            for ch in line:
                ucode = _hidden_unicode_code(ch)
                if ucode is not None:
                    findings.append(
                        SecurityFinding(
                            file_rel=norm,
                            line=line_no,
                            code=ucode,
                            message=f"hidden Unicode code point U+{ord(ch):04X} ({ucode})",
                        )
                    )
                    break
            if is_exec_file:
                for code, rx, desc in _EXFIL_PATTERNS:
                    if rx.search(line):
                        findings.append(
                            SecurityFinding(
                                file_rel=norm,
                                line=line_no,
                                code=code,
                                message=desc,
                            )
                        )
    return findings
