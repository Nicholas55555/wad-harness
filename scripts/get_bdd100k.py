"""Fetch the parts of BDD100K this project uses, from the dataset's official host.

    python scripts/get_bdd100k.py                     # labels + the 10,000 val images (~0.8 GB)
    python scripts/get_bdd100k.py --train-list FILE   # also the train images named in FILE

The image archive (bdd100k_images_100k.zip) is 5.7 GB; the server supports HTTP range requests,
so only the requested members are read out of it (remotezip), each checked against the CRC-32 in
the zip directory. Labels: bdd100k_labels.zip (190 MB, one JSON per image with weather, scene and
time-of-day tags and 2D boxes). The mirror's bdd100k_det_20_labels.zip is mispackaged (it holds
images, not labels) and is not used.

Licence (BDD100K, The Regents of the University of California): use, copy and modify for
educational, research and not-for-profit purposes; cite Yu et al., CVPR 2020.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "http://128.32.162.150/bdd100k"
LABELS = ("bdd100k_labels.zip", "7f1f9043c70a6ff0788a323cfb914aeced109a37b7150740d670a347c881394a")
IMAGES = f"{HOST}/bdd100k_images_100k.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_labels(out: Path) -> None:
    name, want = LABELS
    dst = out / "_zips" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or sha256(dst) != want:
        print(f"downloading {name}")
        urllib.request.urlretrieve(f"{HOST}/{name}", dst)
    if sha256(dst) != want:
        raise SystemExit(f"{name}: SHA-256 mismatch")
    with zipfile.ZipFile(dst) as z:
        info = {i.filename: i for i in z.infolist()}
        members = [m for m in info if m.startswith(("100k/train/", "100k/val/")) and m.endswith(".json")]

        def intact(m):  # an interrupted run can leave a truncated file: check size and CRC, not existence
            p = out / "labels" / m
            return p.exists() and p.stat().st_size == info[m].file_size and                 (zlib.crc32(p.read_bytes()) & 0xFFFFFFFF) == info[m].CRC

        todo = [m for m in members if not intact(m)]
        z.extractall(out / "labels", members=todo)
    print(f"labels: {len(members)} files in {out / 'labels'}")


def fetch(members: list[str], out: Path, workers: int = 16) -> None:
    from remotezip import RemoteZip

    todo = [m for m in members if not (out / "images" / m).exists()]
    print(f"{len(members) - len(todo)} images already present, {len(todo)} to fetch")
    if not todo:
        return
    chunks = [todo[i::workers] for i in range(workers)]

    def work(chunk):
        import time

        n, z = 0, None
        for m in chunk:
            for attempt in range(8):  # the host drops connections now and then; reopen and retry
                try:
                    if z is None:
                        z = RemoteZip(IMAGES, timeout=60)
                    data = z.read(m)  # zipfile verifies the member's CRC-32 on read
                    break
                except Exception as e:  # noqa: BLE001
                    z = None
                    if attempt == 7:
                        raise RuntimeError(f"{m}: {e}") from e
                    time.sleep(2 * (attempt + 1))
            dst = out / "images" / m
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dst)
            n += 1
        return n

    with cf.ThreadPoolExecutor(workers) as ex:
        done = sum(ex.map(work, chunks))
    print(f"fetched {done} images")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--train-list", help="text file of train image names (one per line, without extension)")
    args = ap.parse_args(argv)
    out = ROOT / "data" / "BDD100K"
    get_labels(out)
    members = []
    if not args.no_val:
        members += [f"100k/val/{p.stem}.jpg" for p in sorted((out / "labels" / "100k" / "val").glob("*.json"))]
    if args.train_list:
        names = [l.strip() for l in open(args.train_list) if l.strip()]
        members += [f"100k/train/{n}.jpg" for n in names]
    fetch(members, out)


if __name__ == "__main__":
    sys.exit(main())
