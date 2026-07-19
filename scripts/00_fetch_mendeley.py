"""Download a published Mendeley Data dataset.

Solves the input half of running on a runtime with no persistent storage: rather
than re-uploading images or mounting Drive every session, the dataset is pulled
straight from its DOI in one command.

    python scripts/00_fetch_mendeley.py --doi 10.17632/abcd1234ef.1 --out data/raw
    python scripts/00_fetch_mendeley.py --id abcd1234ef --version 2 --out data/raw

Only works for **published** datasets. Drafts and datasets under embargo are not
served by the public API and need either an OAuth token or a manual download.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PUBLIC_API = "https://data.mendeley.com/public-api/datasets"
USER_AGENT = "aksara-ocr-benchmark/1.0 (research dataset fetcher)"


def parse_dataset_ref(value: str) -> tuple[str, str | None]:
    """Accept a DOI, a dataset URL, or a bare id; return (dataset_id, version)."""
    value = value.strip()

    # DOI form: 10.17632/<id>.<version>
    doi = re.match(r"(?:doi:)?10\.17632/([a-z0-9]+)(?:\.(\d+))?$", value, re.IGNORECASE)
    if doi:
        return doi.group(1), doi.group(2)

    # URL form: https://data.mendeley.com/datasets/<id>/<version>
    url = re.search(r"data\.mendeley\.com/datasets/([a-z0-9]+)(?:/(\d+))?", value, re.IGNORECASE)
    if url:
        return url.group(1), url.group(2)

    if re.fullmatch(r"[a-z0-9]+", value, re.IGNORECASE):
        return value, None

    raise ValueError(
        f"Could not parse {value!r} as a Mendeley dataset reference.\n"
        "Expected a DOI (10.17632/abcd1234ef.1), a dataset URL, or a bare id."
    )


NOT_FOUND_HELP = (
    "Check the dataset id and version.\n"
    "Unpublished drafts and embargoed datasets are not served by the public API. "
    "If yours\nisn't published yet, download it from the web page and pass the "
    "folder to\nscripts/01_prepare_data.py --data-root directly."
)


def _get_json(url: str):
    """GET and parse, treating Mendeley's in-body error codes as errors.

    The public API answers unknown datasets with HTTP 200 and a body of
    ``{"error": 404}`` rather than a 404 status, so status-only checking passes
    the error straight through and it surfaces later as a confusing type error.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise SystemExit(f"Dataset not found (HTTP 404): {url}\n{NOT_FOUND_HELP}") from exc
        raise SystemExit(f"Mendeley API error (HTTP {exc.code}): {url}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach Mendeley: {exc.reason}") from exc

    if isinstance(payload, dict) and "error" in payload:
        if str(payload["error"]) == "404":
            raise SystemExit(f"Dataset not found: {url}\n{NOT_FOUND_HELP}")
        raise SystemExit(f"Mendeley API returned an error: {payload['error']}\n{url}")

    return payload


def _expect_list(payload, what: str, url: str) -> list:
    if not isinstance(payload, list):
        raise SystemExit(
            f"Unexpected response shape for {what} (got {type(payload).__name__}, "
            f"expected a list).\n{url}\nThe public API may have changed."
        )
    return payload


def list_files(dataset_id: str, version: str | None) -> tuple[list[dict], str]:
    """Return (files, version). Resolves the latest version when none is given."""
    if version is None:
        url = f"{PUBLIC_API}/{dataset_id}/versions"
        versions = _expect_list(_get_json(url), "version list", url)
        if not versions:
            raise SystemExit(
                f"Dataset {dataset_id} has no published versions.\n{NOT_FOUND_HELP}"
            )
        version = str(max(int(v["version"]) for v in versions))
        print(f"No version given; using latest: v{version}")

    url = f"{PUBLIC_API}/{dataset_id}/files?folder_id=root&version={version}"
    files = _expect_list(_get_json(url), "file list", url)
    if not files:
        raise SystemExit(
            f"Dataset {dataset_id} v{version} lists no files.\n"
            "Check the version number — it may not exist."
        )
    return files, version


def download(url: str, destination: Path, expected_size: int | None = None) -> None:
    """Stream to a .part file, then rename.

    The rename is what makes a re-run safe: a download interrupted by a dying
    session leaves a .part file, never a truncated file that a later run would
    mistake for complete.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or (expected_size or 0))
        done = 0
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB  ({pct:5.1f}%)", end="")
        print()

    if expected_size and partial.stat().st_size != expected_size:
        partial.unlink()
        raise SystemExit(
            f"Size mismatch for {destination.name}: expected {expected_size}, "
            f"got {partial.stat().st_size}. Re-run to retry."
        )
    partial.replace(destination)


def extract(archive: Path, out_dir: Path) -> bool:
    """Unpack a zip/tar in place. Returns False for non-archives."""
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        return True
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            # filter='data' refuses absolute paths and ../ escapes. Required on
            # Python 3.14, where the unfiltered default was removed.
            tf.extractall(out_dir, filter="data")
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--doi", help="e.g. 10.17632/abcd1234ef.1")
    source.add_argument("--id", dest="dataset_id", help="bare dataset id")
    source.add_argument("--url", help="https://data.mendeley.com/datasets/<id>/<version>")
    parser.add_argument("--version", default=None)
    parser.add_argument("--out", type=Path, default=Path("data/mendeley"))
    parser.add_argument("--list-only", action="store_true", help="Show files and exit.")
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        metavar="SUBSTRING",
        help="Only fetch files whose name contains one of these. Useful for "
             "inspecting one archive before committing to a large download.",
    )
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    dataset_id, ref_version = parse_dataset_ref(args.doi or args.url or args.dataset_id)
    version = args.version or ref_version

    print(f"Dataset {dataset_id}" + (f" v{version}" if version else " (latest)"))
    files, version = list_files(dataset_id, version)

    total_bytes = sum(f.get("size", 0) for f in files)
    print(f"\n{len(files)} file(s), {total_bytes / 1e6:.1f} MB total:\n")
    for f in files:
        print(f"  {f['size'] / 1e6:9.1f} MB  {f['filename']}")

    # Only the root folder is listed — the public API exposes no documented
    # endpoint for walking subfolders. Datasets uploaded as a single archive are
    # unaffected (the archive carries its own tree), but one uploaded as loose
    # files inside folders would silently list only the top level.
    undownloadable = [f for f in files if not (f.get("content_details") or {}).get("download_url")]
    if undownloadable:
        print(
            f"\n  NOTE: {len(undownloadable)} entr(ies) have no download_url and may be "
            "subfolders.\n  Only the dataset's root folder is listed. If files are missing, "
            "download\n  the dataset manually from its web page and use --data-root instead."
        )

    if args.list_only:
        return 0

    if args.include:
        keep = [f for f in files if any(s.lower() in f["filename"].lower() for s in args.include)]
        if not keep:
            raise SystemExit(f"--include {args.include} matched none of the {len(files)} files.")
        print(f"\n--include matched {len(keep)} of {len(files)} file(s).")
        files = keep

    args.out.mkdir(parents=True, exist_ok=True)
    archives: list[Path] = []

    for f in files:
        destination = args.out / f["filename"]
        if destination.exists() and destination.stat().st_size == f.get("size"):
            print(f"\n  have: {f['filename']}")
        else:
            print(f"\n  get : {f['filename']}")
            details = f.get("content_details") or {}
            url = details.get("download_url")
            if not url:
                print(f"    no download_url — skipped")
                continue
            download(url, destination, f.get("size"))

        if not args.no_extract and extract(destination, args.out):
            print(f"    extracted -> {args.out}")
            archives.append(destination)

    if archives and not args.keep_archives:
        for archive in archives:
            archive.unlink()
        print(f"\nRemoved {len(archives)} archive(s) after extraction (--keep-archives to keep).")

    print(f"\nDone -> {args.out}")
    print("\nInspect the layout, then point the prepare script at the directory that\n"
          "contains the per-script folders:")
    print(f"  python scripts/01_prepare_data.py --data-root {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
