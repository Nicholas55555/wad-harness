"""Download DAWN (Kenk & Hassaballah 2020) from its Mendeley Data record into data/DAWN/.

Four zips, about 139 MB in total, each checked against the SHA-256 that Mendeley publishes for
it, then unpacked. Licence: CC BY-NC 3.0 (non-commercial use with attribution).

    python scripts/get_dawn.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "766ygrbt8y"  # doi:10.17632/766ygrbt8y.3
API = f"https://data.mendeley.com/public-api/datasets/{DATASET}"
EXPECTED = {  # from the v3 record, checked 2026-10-01
    "Fog.zip": "09f4f8987e030b8da9e76eb73ab00b64fb628cd43df42e460bc26b9a0912c7c2",
    "Rain.zip": "005bff2cc6c4dad1d4da76d58a50d2a6e071194b3583c28a4d1c8f90a972b76c",
    "Sand.zip": "19dff0e43f7d7e00b9c3768c39245f3164b9a8a8c5a92274252c9449d92ab81c",
    "Snow.zip": "9cf3ccbef87e2ae4055d87a9b4a9e24010f6ef2951a84ced6bd30b04820b3f6f",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = ROOT / "data" / "DAWN"
    zips = out / "_zips"
    zips.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as r:
        record = json.load(r)
    if record.get("version") != 3:
        print(f"note: Mendeley record is now version {record.get('version')}; expected 3", file=sys.stderr)
    for f in record["files"]:
        name, url = f["filename"], f["content_details"]["download_url"]
        want = EXPECTED.get(name, f["content_details"]["sha256_hash"])
        dst = zips / name
        if not dst.exists() or sha256(dst) != want:
            print(f"downloading {name} ({f['size'] / 1e6:.1f} MB)")
            urllib.request.urlretrieve(url, dst)
        got = sha256(dst)
        if got != want:
            raise SystemExit(f"{name}: SHA-256 {got} does not match {want}")
        with zipfile.ZipFile(dst) as z:
            z.extractall(out)
        print(f"{name}: verified and unpacked")
    print(f"DAWN is in {out}")


if __name__ == "__main__":
    main()
