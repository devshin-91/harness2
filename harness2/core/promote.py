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

Validates the final merged harness tree (pinned policy merge, surface schema validation,
static security scan, scrubbed-env regression verification inside a temporary staged copy of
the merged tree), removes stale on-surface files from the destination, writes LF-normalized
files, and emits a replayable SHA-256 `promotion_manifest.json` compatible with
`evidence_gate.verify_receipt()`.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from . import evidence_gate
from . import tree as harness_tree
from .tree import SurfaceSpec

PINNED_POLICY_BANNER = (
    "<!-- HARNESS2 PINNED IMMUTABLE POLICY START (DO NOT EDIT OR REMOVE) -->\n"
)
PINNED_POLICY_END = (
    "\n<!-- HARNESS2 PINNED IMMUTABLE POLICY END -->\n\n"
)


@dataclass(frozen=True)
class PromotionResult:
    ok: bool
    source_dir: Path
    target_dir: Path
    promoted_files: tuple[str, ...] = ()
    removed_stale_files: tuple[str, ...] = ()
    pinned_policy_sha256: str = ""
    commit_sha: str = ""
    input_digest: str = ""
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


def _write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def promote_harness(
    source_harness_dir: Path | str,
    target_dir: Path | str,
    *,
    surface: SurfaceSpec,
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

    # 1. Merge immutable pinned policy FIRST so validation, security scan, and verify_cmd inspect
    #    the exact tree that will be deployed to production.
    pinned_sha = ""
    promoted_tree = dict(evolved)
    if pinned_policy is not None:
        pp = Path(pinned_policy)
        if not pp.is_file():
            errors.append(f"pinned policy file not found: {pp}")
        else:
            policy_text = pp.read_text(encoding="utf-8").strip()
            if not policy_text:
                errors.append(f"pinned policy file is empty: {pp}")
            else:
                pinned_sha = hashlib.sha256(policy_text.encode("utf-8")).hexdigest()
                primary = surface.always_on[0]
                evolved_primary = promoted_tree.get(primary, "")
                # Strip only a genuine leading pinned block to avoid double insertion or truncation.
                if (
                    evolved_primary.startswith(PINNED_POLICY_BANNER)
                    and PINNED_POLICY_END in evolved_primary
                ):
                    tail_start = evolved_primary.index(PINNED_POLICY_END) + len(PINNED_POLICY_END)
                    evolved_primary = evolved_primary[tail_start:]
                if PINNED_POLICY_BANNER in evolved_primary or PINNED_POLICY_END in evolved_primary:
                    errors.append(
                        f"evolved {primary} contains duplicate or stray pinned policy markers"
                    )
                else:
                    combined = PINNED_POLICY_BANNER + policy_text + PINNED_POLICY_END + evolved_primary
                    promoted_tree[primary] = combined

    # 2. Surface schema & compilation check on the merged tree (UTF-8 bytes enforced on byte surfaces).
    if not errors:
        try:
            harness_tree.validate(
                surface,
                promoted_tree,
                allow_subagents=allow_subagents,
            )
        except ValueError as exc:
            errors.append(f"surface validation failed: {exc}")

    # 3. Static security scan across all files of the merged tree.
    sec_findings = tuple(evidence_gate.scan_harness_security(promoted_tree))
    if sec_findings:
        for f in sec_findings:
            errors.append(f"security scan blocked promotion: {f.file_rel}:{f.line} [{f.code}] {f.message}")

    # 4. Regression verification command executed on a temporary staged copy of `promoted_tree`
    #    with scrubbed environment (no host secret leakage, timeout handled cleanly).
    if verify_cmd.strip() and not errors:
        with tempfile.TemporaryDirectory(prefix="harness2-promote-verify-") as stage_td:
            stage_root = Path(stage_td)
            for rel, content in promoted_tree.items():
                _write_lf(stage_root / harness_tree.normalize_rel(rel), content)
            try:
                proc = subprocess.run(
                    verify_cmd,
                    cwd=stage_root,
                    env=config.scrub_env(),
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                errors.append("regression check timed out after 120s")
            else:
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

    commit_sha = evidence_gate.git_commit_sha(src)
    target_commit_sha = evidence_gate.git_commit_sha(dst)
    input_digest = evidence_gate.sha256_mapping(
        {
            "surface": surface.name,
            "evolved_tree": evolved,
            "pinned_policy_sha256": pinned_sha,
            "command": verify_cmd.strip(),
        }
    )

    removed_stale: list[str] = []
    file_hashes: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    artifact_paths: dict[str, str] = {}
    empty_files: list[str] = []
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        dst_resolved = dst.resolve()
        new_rel_set = {harness_tree.normalize_rel(r) for r in promoted_tree}
        # Remove stale on-surface files (e.g., deleted hooks/scripts/skills) before writing new ones.
        for existing in sorted(dst.rglob("*"), reverse=True):
            if not existing.is_file() or "__pycache__" in existing.parts:
                continue
            rel = existing.relative_to(dst).as_posix()
            if harness_tree.is_allowed(surface, rel) and rel not in new_rel_set:
                existing.unlink()
                removed_stale.append(rel)

        for rel in sorted(promoted_tree):
            norm_rel = harness_tree.normalize_rel(rel)
            dest_file = (dst / norm_rel).resolve()
            if not dest_file.is_relative_to(dst_resolved):
                raise ValueError(f"promoted path escapes target_dir: {rel!r}")
            _write_lf(dest_file, promoted_tree[rel])
            # Hash the ACTUAL bytes written on disk so Windows & POSIX receipts match 100%.
            digest = evidence_gate.sha256_file(dest_file)
            file_hashes[norm_rel] = digest
            if dest_file.stat().st_size > 0:
                artifact_hashes[norm_rel] = digest
                artifact_paths[norm_rel] = str(dest_file)
            else:
                empty_files.append(norm_rel)
    else:
        for rel in sorted(promoted_tree):
            norm_rel = harness_tree.normalize_rel(rel)
            lf_bytes = promoted_tree[rel].replace("\r\n", "\n").encode("utf-8")
            file_hashes[norm_rel] = hashlib.sha256(lf_bytes).hexdigest()

    manifest_path: Path | None = None
    if not dry_run:
        manifest_path = dst / "promotion_manifest.json"
        manifest: dict = {
            "schema_version": "1",
            "status": "PASS",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "surface": surface.name,
            "source_dir": str(src),
            "commit_sha": commit_sha,
            "target_commit_sha": target_commit_sha,
            "command": verify_cmd.strip(),
            "input_digest": input_digest,
            "pinned_policy_sha256": pinned_sha,
            "artifact_hashes": artifact_hashes,
            "artifact_paths": artifact_paths,
            "empty_files": sorted(empty_files),
            "files": file_hashes,
            "removed_stale_files": sorted(removed_stale),
        }
        hmac_key = os.environ.get("HARNESS2_RECEIPT_HMAC_KEY", "")
        if hmac_key:
            manifest["signature_hmac_sha256"] = evidence_gate.compute_receipt_hmac(
                manifest, hmac_key
            )
        _write_lf(manifest_path, json.dumps(manifest, indent=2) + "\n")

    return PromotionResult(
        ok=True,
        source_dir=src,
        target_dir=dst,
        promoted_files=tuple(sorted(file_hashes)),
        removed_stale_files=tuple(sorted(removed_stale)),
        pinned_policy_sha256=pinned_sha,
        commit_sha=commit_sha,
        input_digest=input_digest,
        manifest_path=manifest_path,
    )
