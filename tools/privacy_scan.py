"""Report personal path locations without printing their values; compare to a Git base."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

PATTERN = re.compile(rb"(?:[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"<>]+|/(?:Users|home)/[^/\s\"<>]+/|/(?:root)/)")


def scan(base: str = "main") -> dict:
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"])
    locations, introduced = [], []
    for encoded in sorted(set(names.split(b"\0")) - {b""}):
        name = encoded.decode("utf-8")
        path = Path(name)
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        matches = list(PATTERN.finditer(data))
        if not matches:
            continue
        previous = subprocess.run(["git", "show", f"{base}:{name}"], capture_output=True, check=False)
        old = set(PATTERN.findall(previous.stdout)) if previous.returncode == 0 else set()
        row = {"file": name, "lines": sorted({data[:m.start()].count(b"\n") + 1 for m in matches}),
               "file_sha256": hashlib.sha256(data).hexdigest(), "matches": len(matches),
               "new_values": sum(m.group() not in old for m in matches)}
        locations.append(row)
        if row["new_values"]:
            introduced.append(name)
    return {"base": base, "files_with_personal_paths": len(locations),
            "new_personal_path_files": introduced, "locations": locations,
            "scope": "tracked plus nonignored untracked files; ignored datasets/logs/environments excluded",
            "historical_policy": "Report existing evidence paths without altering historical records or their seals."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="main")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = scan(args.base)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "locations"}))
    return 1 if result["new_personal_path_files"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
