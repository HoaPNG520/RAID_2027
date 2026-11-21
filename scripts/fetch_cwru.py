#!/usr/bin/env python3
"""Download CWRU Bearing Data Center recordings (12 kHz drive-end channel) from the public .npz
mirror github.com/srigas/CWRU_Bearing_NumPy and verify their SHA-256 checksums.

  --set core : the four 1797 rpm recordings used in the main audit and comparison (default)
  --set wide : all 40 recordings of the wider audit (4 speeds x {IR, OR@6, B} x {7, 14, 21} mil
               + normal baseline)
"""
import argparse
import hashlib
import os
import sys
import urllib.parse
import urllib.request

BASE = "https://raw.githubusercontent.com/srigas/CWRU_Bearing_NumPy/main/Data/"
WIDE = {
    "1730_B_14_DE12.npz": "873bfea949155de79249d1603190b76b18daf83d3d9b2ceec5d151039f4cfe0c",
    "1730_B_21_DE12.npz": "ac66f158a5603f74372c3de7a1ea8cb8db37b0fe9ec62ba13a95dc6acc66f699",
    "1730_B_7_DE12.npz": "52f3c0676df6da0f9127e50f90c446b61624220ef8d73f69099937eb8c70b2a2",
    "1730_IR_14_DE12.npz": "372bf4155c42c652c3f451dd5b1eb3b6f5ffed92cdb359a01ab1be5c363f7d51",
    "1730_IR_21_DE12.npz": "f68d3e8bd1e0948ebe33d050edac5efc283ba0c87fe5009a9eba48753559e1c5",
    "1730_IR_7_DE12.npz": "807c5d48973c8c49dc58e27a4177f9e481e8f95f3d72ce5ebe2665aa01222dfe",
    "1730_Normal.npz": "e71083ee7e4971575b3deb99325b2eee058d1010a69652e3f0347673c45092b6",
    "1730_OR@6_14_DE12.npz": "6585557063bc55387b346424b3ae1afff82dfc898f9c998a2555ec414e6feb8d",
    "1730_OR@6_21_DE12.npz": "0654f52b747dc1af31e78de5d53613dedfd25fb0cad36335b5a4710e0d209dcd",
    "1730_OR@6_7_DE12.npz": "77ba53cbcf7d51781b5c09d4a9dba1491acb07b9e399f051f489a36c12adf253",
    "1750_B_14_DE12.npz": "51f71b777a4fc8ef26c97408c745ce57f4e37620b9a2714110c11865dd608515",
    "1750_B_21_DE12.npz": "d81fc5830e71a0b05dc34d3ce5b859a2ee9749a52e2a93ef1f214a1291c1c8c7",
    "1750_B_7_DE12.npz": "ddfbdcc5065492ffcef2cc917996ae1384321fa59dea313b978cc192e7514096",
    "1750_IR_14_DE12.npz": "c3a6e09970eb1857ccb6006cfcdb443d0d7af9ac1bb69747be6d248b6b0cb5bb",
    "1750_IR_21_DE12.npz": "5429f9a29c2ed4cfa4985d58f2371edb3450841a59787d54f026134e0f96f74e",
    "1750_IR_7_DE12.npz": "dd601c822e287065badf1963ab543404aac7156b986e0799afabd4dbea0aed5f",
    "1750_Normal.npz": "77830970288497a8f3dbf0c8a3d90d7142846bd7283735e3a795f85b49995a28",
    "1750_OR@6_14_DE12.npz": "6e137b06d95e12c1ba422663b5091cbd26d63cc35dbd4cc2a9c364b374bfca82",
    "1750_OR@6_21_DE12.npz": "f0d79a1ac896dbb0a042a5607b181bd27068f845079d46ee8d62f636d87a8d4a",
    "1750_OR@6_7_DE12.npz": "39890b0b01907e0fa15ddc1b2230fc5f0e08fb9d6fd9a777bc2af82324bca49b",
    "1772_B_14_DE12.npz": "031699ebcb6008543c857ca1c291442664c0f90c0c6bb1bb63944e7fba8c7e6c",
    "1772_B_21_DE12.npz": "6267c515becf3c8a7e156515bcf3460454029a12364518760a681b633c257ad5",
    "1772_B_7_DE12.npz": "7542b67fa52f17ac8b8720060c6ec63c46caa16ab5fbf64ffc59c763f5f0c109",
    "1772_IR_14_DE12.npz": "936e882c7a31c1bfa79533d365ec263949d619117fcd5fd03f15a0773fef618d",
    "1772_IR_21_DE12.npz": "6fcd1329cb438801897c06e995cf1e13e6c7c65942ccf3def773937f36d30aa1",
    "1772_IR_7_DE12.npz": "23b9d0110cabccee79d94f0b12942590540d8e66e66818b98a5c1470a7a654b7",
    "1772_Normal.npz": "ebb7cce9e15c260dacf2999c28551d0be43ce196f85dcd2227fd4cf3221df175",
    "1772_OR@6_14_DE12.npz": "ac28e7e2900c75225b343dcb05c38fc849ad1d30fe71804f8112350a1adc2c51",
    "1772_OR@6_21_DE12.npz": "6b33cd1d636fe363f12f8fa0cc79ff77671ff339b87553f28d3524e60516a9a2",
    "1772_OR@6_7_DE12.npz": "c93827a28e533730c6fad84d2cb2dd00cf881df1c0d77871595d740aaa0d1a1e",
    "1797_B_14_DE12.npz": "3e439e091b84ea353de1a722f6abb7d43df1924682e51afffd3b1d4eb9ea88f7",
    "1797_B_21_DE12.npz": "1147555cb94383cd6ae41541ed62ea95210a3b0c247093912c54905df1ea1818",
    "1797_B_7_DE12.npz": "b3f85480c32e6e3e858dec74ed0f2c7f5c698ab53937e99d621bae6c1414937f",
    "1797_IR_14_DE12.npz": "ca0722f375dc1f807a5e3b3ddd884b25bfa8756550872495fe70413da91d2e5b",
    "1797_IR_21_DE12.npz": "4fe4b0198a9ef77e2d249420ebd31c558a82f9d95db9f9fef1c26345d88b9c33",
    "1797_IR_7_DE12.npz": "b1694c0ed4c77486c909a4fac8f512101cad21302358743486bc4edaa8e8cff3",
    "1797_Normal.npz": "4cd7f673aced17ef1b1253159b32d6c3146ba76af1b6034b1e18c86aed6c0e26",
    "1797_OR@6_14_DE12.npz": "0886978ce0fb409d45494f9776a98db3aaa5e38b3f2d72340c4f5e9ceb3db4a1",
    "1797_OR@6_21_DE12.npz": "068d400e9fe360d6d49b188aca568fa60a4451128d724868d63f741e1f824c8c",
    "1797_OR@6_7_DE12.npz": "c44f3adcfd04026f21ccd081e43a3ba6b9631e92ad1940165d390b7be1909172",
}
CORE = ("1797_IR_7_DE12.npz", "1797_OR@6_7_DE12.npz", "1797_B_7_DE12.npz", "1797_Normal.npz")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, digest, out, tries=3):
    dst = os.path.join(out, name)
    if os.path.exists(dst) and sha256(dst) == digest:
        return dst
    url = BASE + urllib.parse.quote(name.split("_")[0] + " RPM/" + name)
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dst + ".part", "wb") as f:
                f.write(r.read())
            os.replace(dst + ".part", dst)
            if sha256(dst) == digest:
                return dst
            last = f"checksum mismatch: {sha256(dst)}"
        except OSError as exc:   # partial transfer, timeout: retry
            last = repr(exc)
    sys.exit(f"could not fetch {name}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", choices=("core", "wide"), default="core")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = CORE if a.set == "core" else tuple(sorted(WIDE))
    for name in names:
        fetch(name, WIDE[name], a.out)
        print("ok", name)


if __name__ == "__main__":
    main()
