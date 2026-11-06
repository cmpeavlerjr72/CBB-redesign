"""Tolerant reader for the collector's multi-member gzip JSONL files (tape and orderbook).

A collector restart or kill can leave a truncated final gzip member; gzip.open then raises
mid-file. This reader yields every complete record and stops cleanly at the first corrupt member.

  python -I read_tape.py <file.jsonl.gz> [--summary]
Import: from read_tape import iter_records
"""
import gzip
import json
import sys
import zlib
from collections import Counter


def iter_records(path):
    with gzip.open(path, "rb") as f:
        buf = b""
        while True:
            try:
                chunk = f.read(1 << 20)
            except (EOFError, zlib.error, OSError):
                break
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                if ln:
                    try:
                        yield json.loads(ln)
                    except ValueError:
                        continue
        if buf.strip():
            try:
                yield json.loads(buf)
            except ValueError:
                pass


def summary(path):
    n, by_series, by_cat_size = 0, Counter(), Counter()
    for r in iter_records(path):
        n += 1
        tk = r.get("ticker") or ""
        by_series[tk.split("-")[0]] += 1
        cnt = float(r.get("count_fp") or r.get("count") or 0)
        by_cat_size["ge100" if cnt >= 100 else "lt100"] += 1
    print("records:", n)
    print("size split:", dict(by_cat_size))
    print("top series:", by_series.most_common(15))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if "--summary" in sys.argv:
        summary(sys.argv[1])
    else:
        for rec in iter_records(sys.argv[1]):
            print(json.dumps(rec))
