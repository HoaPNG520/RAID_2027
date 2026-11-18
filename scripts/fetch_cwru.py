#!/usr/bin/env python3
"""Download the four CWRU recordings used in the paper from the public .npz mirror of the
CWRU Bearing Data Center (github.com/srigas/CWRU_Bearing_NumPy) and verify checksums."""
import argparse
import hashlib
import os
import sys
import urllib.parse
import urllib.request

BASE = "https://raw.githubusercontent.com/srigas/CWRU_Bearing_NumPy/main/Data/1797%20RPM/"
FILES = {
    "1797_IR_7_DE12.npz": "b1694c0ed4c77486c909a4fac8f512101cad21302358743486bc4edaa8e8cff3",
    "1797_OR@6_7_DE12.npz": "c44f3adcfd04026f21ccd081e43a3ba6b9631e92ad1940165d390b7be1909172",
    "1797_B_7_DE12.npz": "b3f85480c32e6e3e858dec74ed0f2c7f5c698ab53937e99d621bae6c1414937f",
    "1797_Normal.npz": "4cd7f673aced17ef1b1253159b32d6c3146ba76af1b6034b1e18c86aed6c0e26",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, digest in FILES.items():
        dst = os.path.join(a.out, name)
        if not (os.path.exists(dst) and sha256(dst) == digest):
            url = BASE + urllib.parse.quote(name)
            with urllib.request.urlopen(url, timeout=120) as r, open(dst + ".part", "wb") as f:
                f.write(r.read())
            os.replace(dst + ".part", dst)
        got = sha256(dst)
        if got != digest:
            sys.exit(f"checksum mismatch for {name}: {got}")
        print("ok", name)


if __name__ == "__main__":
    main()
