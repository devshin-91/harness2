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

"""CLI for promoting an offline-evolved harness into a production base workspace (#149)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..core.promote import promote_harness
from ..substrate import ANTIGRAVITY_SURFACE, CODEX_SURFACE, OPENCODE_SURFACE

_SURFACES = {
    "agy": ANTIGRAVITY_SURFACE,
    "antigravity": ANTIGRAVITY_SURFACE,
    "oc": OPENCODE_SURFACE,
    "opencode": OPENCODE_SURFACE,
    "cx": CODEX_SURFACE,
    "codex": CODEX_SURFACE,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="harness2-promote",
        description="Validate, security-scan, and promote an evolved harness into production.",
    )
    p.add_argument("source", type=Path, help="Path to the adopted offline harness directory")
    p.add_argument("target", type=Path, help="Destination production harness / workspace root")
    p.add_argument(
        "--substrate",
        default="agy",
        choices=sorted(_SURFACES),
        help="Target harness surface schema (default: agy)",
    )
    p.add_argument(
        "--pinned-policy",
        type=Path,
        default=None,
        help="Optional read-only immutable policy file prepended to the always-on rule file",
    )
    p.add_argument(
        "--verify-cmd",
        default="",
        help="Optional shell regression command that must exit 0 before promotion succeeds",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate without writing files")
    args = p.parse_args(argv)

    res = promote_harness(
        args.source,
        args.target,
        surface=_SURFACES[args.substrate],
        pinned_policy=args.pinned_policy,
        verify_cmd=args.verify_cmd,
        dry_run=args.dry_run,
    )
    if not res.ok:
        for err in res.errors:
            print(f"error: {err}", file=sys.stderr)
        return 1
    mode = "dry-run passed" if args.dry_run else "promoted"
    print(f"{mode}: {len(res.promoted_files)} file(s) -> {res.target_dir}")
    if res.pinned_policy_sha256:
        print(f"pinned policy sha256: {res.pinned_policy_sha256[:16]}")
    if res.manifest_path:
        print(f"manifest: {res.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
