#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request


CHUNK_SIZE = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path, expected_size: int, expected_sha256: str | None) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    if not expected_sha256:
        return True
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )
    return True


def download(artifact: dict, output_dir: Path) -> None:
    target = output_dir / artifact["file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    expected_size = int(artifact["size"])
    expected_sha256 = artifact.get("sha256")

    if validate(target, expected_size, expected_sha256):
        print(f"[ready] {target.name}", flush=True)
        return

    if partial.is_file() and partial.stat().st_size == expected_size:
        if validate(partial, expected_size, expected_sha256):
            partial.replace(target)
            print(f"[complete] {target.name} recovered from complete partial", flush=True)
            return

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0

    request = urllib.request.Request(
        artifact["url"],
        headers={"User-Agent": "VicenTrent-CosyVoice3-MNN/1.0"},
    )
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    started = time.monotonic()
    last_report = started
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            offset = 0
            mode = "wb"
        else:
            mode = "ab" if offset else "wb"
        with partial.open(mode) as output:
            downloaded = offset
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5:
                    elapsed = max(now - started, 0.001)
                    session_bytes = downloaded - offset
                    speed = session_bytes / elapsed / (1024 * 1024)
                    percent = downloaded * 100.0 / expected_size
                    print(
                        f"[download] {target.name} {percent:6.2f}% "
                        f"{downloaded / (1024 ** 2):.1f}/{expected_size / (1024 ** 2):.1f} MiB "
                        f"{speed:.2f} MiB/s",
                        flush=True,
                    )
                    last_report = now

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {target.name}: expected {expected_size}, got {actual_size}"
        )
    actual_sha256 = sha256_file(partial)
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {target.name}: expected {expected_sha256}, got {actual_sha256}"
        )
    partial.replace(target)
    print(f"[complete] {target.name} sha256={actual_sha256}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = set(args.only)
    artifacts = [
        item for item in manifest["artifacts"]
        if not selected or item["name"] in selected
    ]
    if selected and len(artifacts) != len(selected):
        known = {item["name"] for item in manifest["artifacts"]}
        raise RuntimeError(f"Unknown artifact name: {sorted(selected - known)}")

    args.output.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        download(artifact, args.output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; partial files were kept for resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
