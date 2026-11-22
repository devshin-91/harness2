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

import re
from dataclasses import dataclass
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
class EvidenceCheckResult:
    """Outcome of deterministic Tier 0 citation verification on a reflection experience."""

    ok: bool
    verified_refs: tuple[str, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class SecurityFinding:
    """One static security finding on an evolved harness tree."""

    file_rel: str
    line: int
    code: str
    message: str


def verify_experience_evidence(
    experience: Mapping[str, str],
    *,
    valid_labels: Sequence[str],
    changed_components: Sequence[str] = (),
    live_labels: Sequence[str] | None = None,
) -> EvidenceCheckResult:
    """Verify that `<experience>` cites real rollouts or changed components, not fabrications."""
    evidence_text = str(experience.get("evidence") or "").strip()
    edits_text = str(experience.get("edits") or "").strip()
    combined = f"{evidence_text}\n{edits_text}".strip()
    if not evidence_text:
        return EvidenceCheckResult(ok=False, verified_refs=(), issues=("empty evidence field",))

    valid_set = {lbl.lower(): lbl for lbl in valid_labels}
    live_set = (
        {lbl.lower() for lbl in live_labels}
        if live_labels is not None
        else set(valid_set.keys())
    )
    changed_norm = [c.replace("\\", "/") for c in changed_components if c]

    cited_labels = sorted({m.group(0).lower() for m in _ROLLOUT_LABEL_RE.finditer(combined)})
    issues: list[str] = []
    verified: list[str] = []

    for lbl in cited_labels:
        if lbl not in valid_set:
            issues.append(f"fabricated rollout reference `{lbl}` not present in task record")
        elif lbl not in live_set:
            issues.append(f"cited rollout `{valid_set[lbl]}` did not complete with live evidence")
        else:
            verified.append(f"rollout:{valid_set[lbl]}")

    low_combined = combined.lower()
    for comp in changed_norm:
        comp_low = comp.lower()
        basename = comp_low.rsplit("/", 1)[-1]
        if comp_low in low_combined or (len(basename) >= 5 and basename in low_combined):
            verified.append(f"component:{comp}")

    if not verified:
        issues.append(
            "evidence cites neither a real rollout label nor a modified harness component "
            f"(valid rollouts: {sorted(valid_labels)[:8]}; changed: {changed_norm[:6]})"
        )

    return EvidenceCheckResult(
        ok=not issues,
        verified_refs=tuple(sorted(set(verified))),
        issues=tuple(issues),
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
