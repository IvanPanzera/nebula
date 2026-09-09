#!/usr/bin/env python3
"""Pinned Qwen assets, bounded GGUF header audit and resumable verified downloads.

No model code is executed. Large n-gram tables are accounted separately because
their disk size is not the working set of the row-gather cache.
"""
import argparse
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work/qwen"
MODELS = ROOT / "models/qwen"
REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
REVISION = "38bb39ee97821de2c9009abb7e93950eec396e66"
TARGET = "UD-Q4_K_XL/"
MTP = "MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"
FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
           7: "?", 10: "Q", 11: "q", 12: "d"}
TYPES = {0: (1, 4, "F32"), 1: (1, 2, "F16"), 2: (32, 18, "Q4_0"),
         3: (32, 20, "Q4_1"), 6: (32, 22, "Q5_0"), 7: (32, 24, "Q5_1"),
         8: (32, 34, "Q8_0"), 10: (256, 84, "Q2_K"),
         11: (256, 110, "Q3_K"), 12: (256, 144, "Q4_K"),
         13: (256, 176, "Q5_K"), 14: (256, 210, "Q6_K"),
         16: (256, 66, "IQ2_XXS"), 17: (256, 74, "IQ2_XS"),
         18: (256, 98, "IQ3_XXS"), 19: (256, 50, "IQ1_S"),
         20: (32, 18, "IQ4_NL"), 21: (256, 110, "IQ3_S"),
         22: (256, 82, "IQ2_S"), 23: (256, 136, "IQ4_XS"),
         24: (1, 1, "I8"), 25: (1, 2, "I16"), 26: (1, 4, "I32"),
         27: (1, 8, "I64"), 28: (1, 8, "F64"), 30: (1, 2, "BF16")}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def parse_header(data, file_size):
    """Read only the supplied bytes; EOF requests a larger bounded range."""
    f = io.BytesIO(data)

    def read(n):
        if n < 0 or n > 128 * 1024**2:
            raise ValueError("invalid header field length")
        b = f.read(n)
        if len(b) != n:
            raise EOFError("incomplete GGUF header")
        return b

    def number(fmt):
        return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]

    def string(keep=True):
        b = read(number("Q"))
        return b.decode("utf-8") if keep else None

    def value(kind, keep=True):
        if kind in FORMATS:
            return number(FORMATS[kind])
        if kind == 8:
            return string(keep)
        if kind == 9:
            item, count = number("I"), number("Q")
            if count > 10_000_000 or item == 9:
                raise ValueError("invalid GGUF array")
            if item in FORMATS and not keep:
                read(count * struct.calcsize("<" + FORMATS[item]))
                return None
            out = [] if keep else None
            for _ in range(count):
                v = value(item, keep)
                if keep:
                    out.append(v)
            return out
        raise ValueError(f"unknown metadata type {kind}")

    if read(4) != b"GGUF":
        raise ValueError("not GGUF")
    version, nt, nk = number("I"), number("Q"), number("Q")
    if version != 3 or nt > 100_000 or nk > 100_000:
        raise ValueError("unsupported GGUF header")
    metadata = {}
    for _ in range(nk):
        key = string()
        if key in metadata:
            raise ValueError(f"duplicate metadata: {key}")
        metadata[key] = value(number("I"), not key.startswith("tokenizer."))
    tensors = []
    names = set()
    for _ in range(nt):
        name, nd = string(), number("I")
        if name in names or not 1 <= nd <= 4:
            raise ValueError("duplicate tensor or invalid rank")
        names.add(name)
        dims = [number("Q") for _ in range(nd)]
        kind, offset = number("I"), number("Q")
        if kind not in TYPES or not all(0 < d <= 2**40 for d in dims):
            raise ValueError(f"unsupported tensor {name}: {kind}, {dims}")
        block, size, label = TYPES[kind]
        if dims[0] % block:
            raise ValueError(f"unaligned quantized row: {name}")
        tensors.append(dict(name=name, dims=dims, type=kind, format=label,
                            offset=offset, bytes=math.prod(dims) // block * size))
    alignment = metadata.get("general.alignment", 32)
    if alignment <= 0 or alignment & (alignment - 1) or alignment > 4096:
        raise ValueError("invalid GGUF alignment")
    start = (f.tell() + alignment - 1) // alignment * alignment
    end = start
    for t in sorted(tensors, key=lambda t: t["offset"]):
        absolute = start + t["offset"]
        if t["offset"] % alignment or absolute < end or absolute + t["bytes"] > file_size:
            raise ValueError(f"overlapping or out-of-bounds tensor: {t['name']}")
        t["file_offset"] = absolute
        end = absolute + t["bytes"]
    return dict(version=version, data_start=start, metadata=metadata, tensors=tensors)


def manifest():
    url = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    j = r.json()
    if j["sha"] != REVISION:
        raise ValueError("repository revision mismatch")
    files = []
    for x in j["siblings"]:
        name = x["rfilename"]
        if not (name.startswith(TARGET) and name.endswith(".gguf") or name == MTP):
            continue
        if ".." in Path(name).parts or Path(name).is_absolute():
            raise ValueError("unsafe asset path")
        lfs = x["lfs"]
        files.append(dict(name=name, size=lfs["size"], sha256=lfs["sha256"],
                          url=f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"))
    if len(files) != 5:
        raise ValueError("expected four target shards and one shared MTP")
    m = dict(repo=REPO, revision=REVISION, files=files)
    atomic_json(WORK / "assets_manifest.json", m)
    return m


def fetch_range(asset, start, end):
    # The cache-buster distinguishes byte ranges on CDNs which cache redirects.
    url = asset["url"] + f"?download=true&range_start={start}&range_end={end}"
    with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                      timeout=(30, 120), stream=True) as r:
        r.raise_for_status()
        if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{asset['size']}":
            raise ValueError("server did not honor the exact byte range")
        chunks, size = [], 0
        for b in r.iter_content(1024**2):
            size += len(b)
            if size > end - start + 1:
                raise ValueError("oversized range response")
            chunks.append(b)
        if size != end - start + 1:
            raise EOFError("truncated range")
        return b"".join(chunks)


def audit(asset):
    dst = WORK / "headers" / (Path(asset["name"]).name + ".bin")
    data = dst.read_bytes() if dst.exists() else b""
    size = max(len(data), min(asset["size"], 1024**2))
    while True:
        if not data or len(data) < size:
            data = fetch_range(asset, 0, size - 1)
        try:
            info = parse_header(data, asset["size"])
            break
        except EOFError:
            if size >= min(asset["size"], 128 * 1024**2):
                raise
            size = min(size * 2, asset["size"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    info.update(asset=asset, header_sha256=hashlib.sha256(data).hexdigest())
    atomic_json(dst.with_suffix(".json"), info)
    print(f"header {asset['name']}: {len(info['tensors'])} tensors", flush=True)
    return info


def download(asset, connections=2):
    dest = MODELS / asset["name"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    marker = dest.with_suffix(".gguf.verified.json")
    if dest.exists() and marker.exists():
        checked = json.loads(marker.read_text())
        stat = dest.stat()
        if (checked.get("sha256") == asset["sha256"] and stat.st_size == asset["size"]
                and checked.get("mtime_ns") == stat.st_mtime_ns):
            print(f"verified cached {dest.name}", flush=True)
            return
    partial = dest if dest.exists() else dest.with_suffix(".gguf.partial")
    position = partial.stat().st_size if partial.exists() else 0
    if position > asset["size"]:
        raise ValueError(f"oversized partial: {partial}")
    started, initial = time.monotonic(), position
    last_report = started
    def fetch(start,end):
        for attempt in range(6):
            try:return fetch_range(asset,start,end)
            except (requests.RequestException,EOFError) as e:
                if attempt==5:raise
                print(f"retry {dest.name}: {type(e).__name__}",flush=True)
                time.sleep(min(2**attempt,20))
    # Parallel reads, one ordered writer: a partial is always a valid contiguous
    # prefix, including after interruption. Bound buffered data to 64 MiB/connection.
    with ThreadPoolExecutor(max_workers=connections) as pool, partial.open("ab") as f:
        pending=collections.deque();requested=position
        while position<asset['size']:
            while len(pending)<connections and requested<asset['size']:
                end=min(requested+64*1024**2,asset['size'])-1
                pending.append(pool.submit(fetch,requested,end));requested=end+1
            data=pending.popleft().result()
            f.write(data)
            f.flush()
            position += len(data)
            now = time.monotonic()
            if now - last_report >= 20 or position == asset["size"]:
                rate = (position - initial) / max(now - started, 1) / 1024**2
                print(f"{dest.name}: {position / asset['size']:.1%}, {rate:.1f} MiB/s", flush=True)
                last_report = now
    digest = hashlib.sha256()
    with partial.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != asset["sha256"]:
        raise ValueError(f"SHA256 mismatch: retained for inspection at {partial}")
    if partial != dest:
        partial.replace(dest)
    atomic_json(marker, dict(sha256=digest.hexdigest(), size=dest.stat().st_size,
                            mtime_ns=dest.stat().st_mtime_ns, revision=REVISION))
    print(f"SHA256 OK {dest.name}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true")
    p.add_argument("--mtp-only", action="store_true")
    p.add_argument("--target-only", action="store_true")
    p.add_argument("--file-workers",type=int,default=3)
    p.add_argument("--connections-per-file",type=int,default=2)
    args = p.parse_args()
    if not 1<=args.file_workers<=3 or not 1<=args.connections_per_file<=4:
        p.error('file-workers 1..3 and connections-per-file 1..4 required')
    # Keep the lock open until every download/hash worker has completed.
    download_lock=None
    if args.download:
        import fcntl
        WORK.mkdir(parents=True,exist_ok=True);download_lock=(WORK/'download.lock').open('a')
        try:fcntl.flock(download_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('another Qwen asset download is already running')
    m = manifest()
    with ThreadPoolExecutor(max_workers=2) as pool:
        infos = list(pool.map(audit, m["files"]))
    groups = collections.Counter()
    formats = collections.Counter()
    names = set()
    for info in infos:
        is_mtp = info["asset"]["name"] == MTP
        for t in info["tensors"]:
            key = (is_mtp, t["name"])
            if key in names:
                raise ValueError(f"duplicate tensor across shards: {key}")
            names.add(key)
            group = ("mtp" if is_mtp else "ngram" if len(t["dims"]) == 2 and min(t["dims"]) == 160 and max(t["dims"]) > 1_000_000
                     else "routed" if "_exps." in t["name"] else "core")
            groups[group] += t["bytes"]
            formats[f"{group}/{t['format']}"] += t["bytes"]
    report = dict(revision=REVISION, groups_bytes=dict(groups), formats_bytes=dict(formats),
                  groups_GiB={k: v / 1024**3 for k, v in groups.items()},
                  target_non_ngram_GiB=(groups["routed"] + groups["core"]) / 1024**3)
    atomic_json(WORK / "gguf_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if args.download:
        if args.mtp_only and args.target_only:
            p.error("choose either --mtp-only or --target-only")
        chosen = [x for x in m["files"] if (not args.mtp_only or x["name"] == MTP)
                  and (not args.target_only or x["name"] != MTP)]
        with ThreadPoolExecutor(max_workers=args.file_workers) as pool:
            list(pool.map(lambda asset:download(asset,args.connections_per_file), chosen))


if __name__ == "__main__":
    main()
