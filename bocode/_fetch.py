"""Download-on-demand for large data files that are not shipped in the wheel.

Large problem data (the 78 MB SVM dataset, the MOPTA08 and Mazda binaries) is
hosted as release assets rather than bundled, to keep the installed package
small. The first time a problem needs such a file it is downloaded to a local
cache, verified against a sha256 checksum, and reused thereafter.

Hosting: files are resolved from a Hugging Face dataset repo by default
(``BOCODE_DATA_BASE_URL`` overrides the base URL; a GitHub release works too,
since both serve plain HTTPS GETs). For development, a local in-repo copy is
used directly when present via the ``local_fallback`` argument, so contributors
who already have the files do not need network access.

Cache location: ``$BOCODE_CACHE_DIR`` if set, else ``~/.cache/bocode``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

# Hugging Face dataset that hosts the large assets. The resolve/main path serves
# raw files over HTTPS. Override with BOCODE_DATA_BASE_URL (must end with '/').
DEFAULT_BASE_URL = "https://huggingface.co/datasets/rosenyu/bocode-data/resolve/main/"

# filename -> sha256 of the expected file. Fill in once assets are uploaded;
# an entry of None skips verification (logged) until the checksum is known.
_CHECKSUMS = {
    "slice_localization_data.csv": None,
    "mopta08_armhf.bin": None,
    "mopta08_elf32.bin": None,
    "mopta08_elf64.bin": None,
    "mazda_mop_sca": None,
    "mazda_mop": None,
}


def _cache_dir() -> Path:
    base = os.environ.get("BOCODE_CACHE_DIR")
    path = Path(base) if base else Path.home() / ".cache" / "bocode"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, filename: str) -> None:
    expected = _CHECKSUMS.get(filename)
    if expected is None:
        return
    actual = _sha256(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {filename}: expected {expected}, got {actual}. "
            "The cached file was removed; re-run to download again."
        )


def fetch_data_file(filename: str, local_fallback: str | None = None) -> str:
    """Return a local path to ``filename``, downloading and caching it if needed.

    Resolution order:
      1. in-repo ``local_fallback`` if it exists (developer convenience);
      2. previously cached copy in the bocode cache directory;
      3. download from ``BOCODE_DATA_BASE_URL`` (default: the bocode HF dataset).

    Args:
        filename: the asset's file name (also its key in the checksum table).
        local_fallback: optional path to an in-repo copy to use directly.

    Returns:
        Filesystem path (as ``str``) to a readable copy of the file.
    """
    if local_fallback and Path(local_fallback).exists():
        return str(local_fallback)

    cached = _cache_dir() / filename
    if cached.exists():
        return str(cached)

    base = os.environ.get("BOCODE_DATA_BASE_URL", DEFAULT_BASE_URL)
    url = base + filename
    tmp = cached.with_suffix(cached.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as exc:  # noqa: BLE001 - re-raised with actionable guidance
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download '{filename}' from {url}: {exc}. "
            "Set BOCODE_DATA_BASE_URL to a reachable host, or place the file at "
            f"{cached}."
        ) from exc

    tmp.rename(cached)
    _verify(cached, filename)
    return str(cached)
