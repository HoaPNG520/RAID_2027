"""
scripts/extract_pamap2.py against a synthetic PAMAP2 file: the real UCI zip is not fetched
automatically, so these tests pin the part that would otherwise go unchecked until someone
downloads it -- the inner-zip handling, the activity filter and, above all, the column layout
the PAMAP2 readme specifies (per IMU: temperature, then acc +-16 g, then acc +-6 g).

Each cell of the synthetic rows holds its own 0-based column index, so a wrong column pick
shows up as a wrong value rather than as plausible-looking numbers.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
EXTRACT = os.path.join(REPO_ROOT, "scripts", "extract_pamap2.py")

# 1-based column layout from the PAMAP2 readme: 2 activityID, 4-20 IMU hand, 21-37 chest,
# 38-54 ankle; inside an IMU 1 temperature, 2-4 acc +-16 g, 5-7 acc +-6 g. As 0-based indices:
EXPECTED_COLUMNS = {
    "hand16": [4, 5, 6], "hand6": [7, 8, 9],
    "chest16": [21, 22, 23], "chest6": [24, 25, 26],
    "ankle16": [38, 39, 40], "ankle6": [41, 42, 43],
}
KEPT, DROPPED = (5, 24), (1, 2)       # 5 and 24 are in the script's KEEP, 1 and 2 are not


def write_dat(activities):
    rows = []
    for act in activities:
        r = [float(j) for j in range(54)]
        r[1] = float(act)
        rows.append(" ".join(f"{v:g}" for v in r))
    return ("\n".join(rows) + "\n").encode()


def make_zip(path, members, nested=True):
    """members: {name inside the dataset zip: .dat bytes}. nested mirrors the UCI download,
    which is a zip holding PAMAP2_Dataset.zip."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    if not nested:
        path.write_bytes(inner.getvalue())
        return
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("PAMAP2_Dataset.zip", inner.getvalue())


def run_extract(zip_path, out_dir):
    return subprocess.run([sys.executable, "-I", EXTRACT, "--zip", str(zip_path), "--out", str(out_dir)],
                          check=True, capture_output=True, text=True)


@pytest.mark.fast
@pytest.mark.parametrize("nested", [True, False])
def test_reads_nested_and_plain_zip(tmp_path, nested):
    """The UCI download is a zip inside a zip; a bare dataset zip must work too."""
    z = tmp_path / "p.zip"
    make_zip(z, {"PAMAP2_Dataset/Protocol/subject101.dat": write_dat(KEPT)}, nested=nested)
    out = tmp_path / "out"
    run_extract(z, out)
    assert (out / "protocol_subject101.npz").exists()


@pytest.mark.fast
def test_columns_match_the_readme_layout(tmp_path):
    z = tmp_path / "p.zip"
    make_zip(z, {"PAMAP2_Dataset/Protocol/subject101.dat": write_dat(KEPT)})
    out = tmp_path / "out"
    run_extract(z, out)
    d = np.load(out / "protocol_subject101.npz")
    assert sorted(d.files) == ["act"] + sorted(EXPECTED_COLUMNS)
    for key, cols in EXPECTED_COLUMNS.items():
        assert d[key].shape == (len(KEPT), 3)
        for row in d[key]:
            assert row.tolist() == [float(c) for c in cols], key


@pytest.mark.fast
def test_keeps_only_the_high_impact_activities(tmp_path):
    z = tmp_path / "p.zip"
    acts = (DROPPED[0],) + KEPT + (DROPPED[1],) + (KEPT[0],)
    make_zip(z, {"PAMAP2_Dataset/Protocol/subject101.dat": write_dat(acts)})
    out = tmp_path / "out"
    run_extract(z, out)
    d = np.load(out / "protocol_subject101.npz")
    assert d["act"].tolist() == [a for a in acts if a in KEPT]


@pytest.mark.fast
def test_file_with_no_kept_activity_is_skipped(tmp_path):
    z = tmp_path / "p.zip"
    make_zip(z, {"PAMAP2_Dataset/Protocol/subject101.dat": write_dat(DROPPED),
                 "PAMAP2_Dataset/Protocol/subject102.dat": write_dat(KEPT)})
    out = tmp_path / "out"
    run_extract(z, out)
    assert not (out / "protocol_subject101.npz").exists()
    assert (out / "protocol_subject102.npz").exists()


@pytest.mark.fast
def test_session_prefix_and_manifest(tmp_path):
    z = tmp_path / "p.zip"
    make_zip(z, {"PAMAP2_Dataset/Protocol/subject101.dat": write_dat(KEPT),
                 "PAMAP2_Dataset/Optional/subject102.dat": write_dat(KEPT)})
    out = tmp_path / "out"
    run_extract(z, out)
    man = json.load(open(out / "MANIFEST.json"))
    assert set(man["files"]) == {"protocol_subject101.npz", "optional_subject102.npz"}
    for fn, entry in man["files"].items():
        assert entry["rows"] == len(KEPT)
        # the checksum must be of the file as written, so a corrupted copy is detectable
        assert entry["sha256"] == hashlib.sha256((out / fn).read_bytes()).hexdigest()
    assert man["activities_kept"] == [4, 5, 6, 12, 13, 24]
    assert "CC BY 4.0" in man["source"]


@pytest.mark.fast
def test_rejects_a_file_with_the_wrong_column_count(tmp_path):
    z = tmp_path / "p.zip"
    bad = b"\n".join(b" ".join(b"%d" % j for j in range(20)) for _ in range(3)) + b"\n"
    make_zip(z, {"PAMAP2_Dataset/Protocol/subject101.dat": bad})
    out = tmp_path / "out"
    with pytest.raises(subprocess.CalledProcessError) as e:
        run_extract(z, out)
    assert "54 columns" in e.value.stderr
