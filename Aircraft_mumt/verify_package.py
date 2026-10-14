"""Verify the packaged files and pinned third-party repositories (stdlib only)."""
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            failures.append(f"Missing: {entry['path']}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            failures.append(f"Changed: {entry['path']}")
    for name, entry in manifest["vendor_sources"].items():
        source = root / "external_methods" / "inbox" / name / "SOURCE"
        try:
            result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            if result.stdout.strip() != entry["commit"]:
                failures.append(f"Wrong vendor commit: {name}")
        except (OSError, subprocess.CalledProcessError) as error:
            failures.append(f"Git verification failed for {name}: {error}")
    if failures:
        for error in failures:
            print(error)
        raise SystemExit(1)
    print(json.dumps({"verified_files": len(manifest["files"]), "vendor_commits_verified": len(manifest["vendor_sources"]), "source_commit": manifest["source_commit"], "status": "passed"}))


if __name__ == "__main__":
    main()
