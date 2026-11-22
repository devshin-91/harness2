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

"""Offline-to-production harness promotion pipeline (#149).

Validates an evolved harness tree (schema validation, static security scan, pinned immutable
policy preservation, optional regression test command) before promoting it into a target
production base harness directory with a SHA-256 provenance manifest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..substrate.antigravity import (
    ANTIGRAVITY_SURFACE,
    PINNED_POLICY_BANNER,
    PINNED_POLICY_END,
)
from . import evidence_gate
from . import tree as harness_tree
from .tree import SurfaceSpec


@dataclass(frozen=True)
class PromotionResult:
    ok: bool
    source_dir: Path
    target_dir: Path
    promoted_files: tuple[str, ...] = ()
    pinned_policy_sha256: str = ""
    manifest_path: Path | None = None
    errors: tuple[str, ...] = ()
    security_findings: tuple[evidence_gate.SecurityFinding, ...] = field(default_factory=tuple)


def _read_tree_posix(surface: SurfaceSpec, root: Path) -> harness_tree.Tree:
    """Cross-platform tree reader ensuring POSIX keys on Windows and POSIX alike."""
    root = Path(root)
    out: harness_tree.Tree = {}
    if not root.is_dir():
        return out
    stubs = {rel for rel, _ in surface.stubs}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        if rel in stubs:
            continue
        if harness_tree.is_allowed(surface, rel):
            out[rel] = p.read_text(encoding="utf-8", errors="replace")
    return out


def promote_harness(
    source_harness_dir: Path | str,
    target_dir: Path | str,
    *,
    surface: SurfaceSpec = ANTIGRAVITY_SURFACE,
    pinned_policy: Path | str | None = None,
    verify_cmd: str = "",
    allow_subagents: bool = True,
    dry_run: bool = False,
) -> PromotionResult:
    """Validate, security-scan, regression-check, and promote an evolved harness to production."""
    src = Path(source_harness_dir).resolve()
    dst = Path(target_dir).resolve()
    errors: list[str] = []

    if not src.is_dir():
        return PromotionResult(
            ok=False, source_dir=src, target_dir=dst, errors=(f"source dir not found: {src}",)
        )

    evolved = _read_tree_posix(surface, src)
    if not evolved:
        return PromotionResult(
            ok=False,
            source_dir=src,
            target_dir=dst,
            errors=(f"no allowed harness components found under {src} for surface {surface.name}",),
        )

    # 1. Surface schema & compilation check.
    try:
        harness_tree.validate(surface, evolved, allow_subagents=allow_subagents)
    except ValueError as exc:
        errors.append(f"tree validation failed: {exc}")

    # 2. Static security scan (hidden Unicode + credential/rubric exfiltration).
    sec_findings = tuple(evidence_gate.scan_harness_security(evolved))
    if sec_findings:
        errors.append(
            f"security scan blocked promotion ({len(sec_findings)} finding(s)): "
            + "; ".join(f"{f.file_rel}:{f.line} [{f.code}]" for f in sec_findings[:5])
        )

    # 3. Optional pinned immutable policy composition & byte-budget check.
    pinned_sha = ""
    promoted_tree = dict(evolved)
    if pinned_policy is not None:
        pp = Path(pinned_policy)
        if not pp.is_file():
            errors.append(f"pinned policy file not found: {pp}")
        else:
            policy_text = pp.read_text(encoding="utf-8").strip()
            pinned_sha = hashlib.sha256(policy_text.encode("utf-8")).hexdigest()
            primary = surface.always_on[0]
            evolved_primary = promoted_tree.get(primary, "")
            # Strip any previous pinned banner to keep promotion idempotent.
            if PINNED_POLICY_BANNER in evolved_primary and PINNED_POLICY_END in evolved_primary:
                tail_start = evolved_primary.index(PINNED_POLICY_END) + len(PINNED_POLICY_END)
                evolved_primary = evolved_primary[tail_start:]
            combined = PINNED_POLICY_BANNER + policy_text + PINNED_POLICY_END + evolved_primary
            if len(combined.encode("utf-8")) > surface.always_on_max:
                errors.append(
                    f"combined pinned policy + {primary} is {len(combined.encode('utf-8'))} "
                    f"bytes, exceeding {surface.always_on_max}-byte cap"
                )
            else:
                promoted_tree[primary] = combined

    # 4. Optional regression verification command before touching target_dir.
    if verify_cmd.strip() and not errors:
        proc = subprocess.run(
            verify_cmd,
            cwd=src,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        if proc.returncode != 0:
            tail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()[-300:]
            errors.append(f"regression check failed (exit={proc.returncode}): {tail}")

    if errors:
        return PromotionResult(
            ok=False,
            source_dir=src,
            target_dir=dst,
            pinned_policy_sha256=pinned_sha,
            errors=tuple(errors),
            security_findings=sec_findings,
        )

    file_hashes: dict[str, str] = {}
    for rel in sorted(promoted_tree):
        content = promoted_tree[rel]
        file_hashes[rel] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if not dry_run:
            dest_file = dst / harness_tree.normalize_rel(rel)
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            dest_file.write_text(content, encoding="utf-8")

    manifest_path: Path | None = None
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        manifest_path = dst / "promotion_manifest.json"
        manifest = {
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "surface": surface.name,
            "source_dir": str(src),
            "pinned_policy_sha256": pinned_sha,
            "files": file_hashes,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return PromotionResult(
        ok=True,
        source_dir=src,
        target_dir=dst,
        promoted_files=tuple(sorted(promoted_tree)),
        pinned_policy_sha256=pinned_sha,
        manifest_path=manifest_path,
    )
