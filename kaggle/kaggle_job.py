#!/usr/bin/env python3
"""
Run a GPU job on Kaggle (T4 x2) from your own machine and fetch the results.

  export KAGGLE_USERNAME=<your kaggle username>   # never taken from the git remote
  python3 kaggle/kaggle_job.py list
  python3 kaggle/kaggle_job.py run smoke --approved     # push, wait, fetch, validate

What it does:
  1. checks: clean git tree, HEAD pushed (origin/<branch> == HEAD), weekly GPU budget,
     no run of the same kernel already queued or running;
  2. generates ONE self-contained script (kaggle/runner_template.py + the job config)
     and kernel-metadata.json in kaggle/build/<job_id>/ -- only this script is uploaded;
     inside Kaggle it clones the public repo at the exact commit and runs the commands;
  3. pushes it, polls until it finishes, downloads /kaggle/working into
     bench/raw/<job_id>/, validates job_summary.json (job id, commit, GPU, exit codes),
     and records the GPU time in bench/quota.json.
Only the kaggle CLI is called; the credential file is never read by this script.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MDMS_ROOT", os.path.dirname(HERE))   # tests point this at a temp repo
TEMPLATE = os.path.join(HERE, "runner_template.py")
QUOTA = os.path.join(ROOT, "bench", "quota.json")
WEEKLY_CAP_H = 20.0
KAGGLE = os.environ.get("MDMS_KAGGLE_BIN", "kaggle")   # tests point this at a fake CLI

PY = "python3 -I"
JOBS = {
    "smoke": dict(
        est_min=10,
        commands=[
            dict(name="env", cmd="nvidia-smi; nvcc --version; g++ --version | head -1; python3 -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'", timeout_sec=120),
            dict(name="build", cmd="make all", timeout_sec=600),
            dict(name="unit_tests", cmd=f"{PY} -m pytest -q -m fast tests/", timeout_sec=600),
            dict(name="sanitizer", cmd="mkdir -p out && sh scripts/sanitize.sh bin/mdms_gpu out/sanitizer.txt", timeout_sec=1200),
            dict(name="agree_gpu_vs_twin", cmd=f"mkdir -p out && {PY} gpu/bench.py --suite agree --gpu bin/mdms_gpu --twin bin/mdms_twin --out out/agree.jsonl", timeout_sec=1200),
        ],
        collect=["out"]),
    "validate": dict(
        est_min=25,
        commands=[
            dict(name="build", cmd="make all", timeout_sec=600),
            dict(name="fetch_cwru", cmd=f"{PY} scripts/fetch_cwru.py --out data", timeout_sec=600),
            dict(name="t5_t6_gpu", cmd=f"mkdir -p out && {PY} gpu/compare.py --bin bin/mdms_gpu --suite t5 --out out/t5_gpu.jsonl", timeout_sec=2400),
            dict(name="t8_gpu", cmd=f"{PY} gpu/compare.py --bin bin/mdms_gpu --suite t8 --out out/t8_gpu.jsonl --data-dir data", timeout_sec=2400),
        ],
        collect=["out"], stop_on_failure=False),
    "bench": dict(
        est_min=60,
        commands=[
            dict(name="build", cmd="make all", timeout_sec=600),
            dict(name="stumpy", cmd="pip install -q stumpy || true", timeout_sec=600),
            dict(name="speed_two_gpus", cmd=(
                f"mkdir -p out && ({PY} gpu/bench.py --suite speed --gpu bin/mdms_gpu --twin bin/mdms_twin "
                f"--out out/speed_d0.jsonl --device 0 --shard 0/2 --stumpy > out/speed_d0.log 2>&1 & p0=$!; "
                f"{PY} gpu/bench.py --suite speed --gpu bin/mdms_gpu --twin bin/mdms_twin "
                f"--out out/speed_d1.jsonl --device 1 --shard 1/2 > out/speed_d1.log 2>&1 & p1=$!; "
                f"wait $p0; r0=$?; wait $p1; r1=$?; exit $((r0 | r1)))"),
                timeout_sec=5400),
        ],
        collect=["out"], stop_on_failure=False),
    "bench2": dict(
        est_min=40,
        commands=[
            dict(name="build", cmd="make all", timeout_sec=600),
            dict(name="stumpy", cmd="pip install -q stumpy || true", timeout_sec=600),
            dict(name="speed2_and_stumpy_fair", cmd=(
                f"mkdir -p out && (({PY} gpu/bench.py --suite speed2 --gpu bin/mdms_gpu "
                f"--out out/speed2_d0.jsonl --device 0 --shard 0/2 > out/speed2_d0.log 2>&1 && "
                f"{PY} gpu/bench.py --suite stumpy_fair --gpu bin/mdms_gpu "
                f"--out out/stumpy_fair.jsonl --device 0 > out/stumpy_fair.log 2>&1) & p0=$!; "
                f"{PY} gpu/bench.py --suite speed2 --gpu bin/mdms_gpu "
                f"--out out/speed2_d1.jsonl --device 1 --shard 1/2 > out/speed2_d1.log 2>&1 & p1=$!; "
                f"wait $p0; r0=$?; wait $p1; r1=$?; exit $((r0 | r1)))"),
                timeout_sec=5400),
        ],
        collect=["out"], stop_on_failure=False),
}


# --------------------------------------------------------------------------- #
def run(cmd, check=True, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, **kw)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r


def git(*args, check=True):
    return run(["git", *args], check=check).stdout.strip()


def kaggle_username():
    u = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not u:
        out = run([KAGGLE, "config", "view"], check=False).stdout
        m = re.search(r"username:\s*(\S+)", out)
        u = m.group(1) if m else ""
    if not u or u.lower() in ("anonymous", "none"):
        raise SystemExit("set KAGGLE_USERNAME to your Kaggle username")
    return u


def https_url(url):
    m = re.match(r"git@github\.com:(.+?)(\.git)?$", url)
    if m:
        return f"https://github.com/{m.group(1)}.git"
    if url.startswith("https://") or url.startswith("file://"):
        return url
    raise SystemExit(f"unsupported remote URL {url!r}; the repo must be public on GitHub")


def preconditions(allow_dirty=False):
    if not allow_dirty and git("status", "--porcelain"):
        raise SystemExit("working tree not clean: commit first")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    sha = git("rev-parse", "HEAD")
    run(["git", "fetch", "--quiet", "origin", branch])
    remote_sha = git("rev-parse", f"origin/{branch}", check=False)
    if remote_sha != sha:
        raise SystemExit(f"HEAD {sha[:10]} is not on origin/{branch} ({remote_sha[:10]}): push first")
    return branch, sha, https_url(git("remote", "get-url", "origin"))


def load_quota():
    if os.path.exists(QUOTA):
        with open(QUOTA) as f:
            return json.load(f)
    return dict(weekly_cap_hours=WEEKLY_CAP_H, jobs=[])


def used_this_week(q):
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)
    return sum(j.get("gpu_hours", 0.0) for j in q["jobs"]
               if dt.datetime.fromisoformat(j["finished_utc"]) >= cutoff)


def record_quota(entry):
    q = load_quota()
    q["jobs"].append(entry)
    os.makedirs(os.path.dirname(QUOTA), exist_ok=True)
    with open(QUOTA, "w") as f:
        json.dump(q, f, indent=2)


STATUS_NEW = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")
STATUS_OLD = re.compile(r"has status \"?([A-Za-z_]+)")


def parse_status(text):
    """Map `kaggle kernels status` output to queued/running/complete/error/cancel/unknown.
    Current CLIs print '<id> has status "KernelWorkerStatus.COMPLETE"'; older ones print
    '<id> has status "complete"'. Only the status token is inspected, never the kernel id."""
    m = STATUS_NEW.search(text) or STATUS_OLD.search(text)
    if not m:
        return None
    tok = m.group(1).lower()
    for s in ("queued", "running", "complete", "error", "cancel"):
        if tok.startswith(s):
            return s
    return "unknown"


def kernel_status(kid):
    r = run([KAGGLE, "kernels", "status", kid], check=False)
    s = parse_status(r.stdout + "\n" + r.stderr)
    if s:
        return s
    return "unknown" if r.returncode == 0 else "none"


def build(job, job_id, user, branch, sha, url):
    spec = JOBS[job]
    cfg = dict(job=job, job_id=job_id, repo_url=url, branch=branch, sha=sha,
               default_timeout_sec=1800, stop_on_failure=spec.get("stop_on_failure", True),
               commands=spec["commands"], collect=spec["collect"])
    with open(TEMPLATE) as f:
        src = f.read()
    payload = json.dumps(cfg, indent=1)
    if "'''" in payload:
        raise SystemExit("job config may not contain triple quotes")
    src = src.replace("__JOB_JSON__", payload)
    d = os.path.join(ROOT, "kaggle", "build", job_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "run.py"), "w") as f:
        f.write(src)
    meta = dict(id=f"{user}/mdms-{job}", title=f"mdms {job}", code_file="run.py",
                language="python", kernel_type="script", is_private=True,
                enable_gpu=True, enable_internet=True, machine_shape="NvidiaTeslaT4",
                dataset_sources=[], competition_sources=[], kernel_sources=[], model_sources=[])
    with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return d, meta["id"]


def push(d):
    helptext = run([KAGGLE, "kernels", "push", "--help"], check=False).stdout
    cmd = [KAGGLE, "kernels", "push", "-p", d]
    if "--accelerator" in helptext:
        cmd += ["--accelerator", "NvidiaTeslaT4"]
    r = run(cmd, check=False)
    print(r.stdout.strip(), r.stderr.strip())
    if r.returncode != 0 or "error" in (r.stdout + r.stderr).lower():
        raise SystemExit("push failed")


def wait(kid, max_minutes):
    t0 = time.time()
    seen_active = False
    delay = 20
    while time.time() - t0 < max_minutes * 60:
        s = kernel_status(kid)
        print(f"[{dt.datetime.now():%H:%M:%S}] {kid}: {s}", flush=True)
        if s in ("queued", "running"):
            seen_active = True
        elif s in ("complete", "error", "cancel") and (seen_active or time.time() - t0 > 120):
            return s
        time.sleep(delay)
        delay = min(60, delay + 10)
    return "timeout"


def fetch_and_validate(kid, job_id, sha, out_root=None):
    out = out_root or os.path.join(ROOT, "bench", "raw", job_id)
    os.makedirs(out, exist_ok=True)
    run([KAGGLE, "kernels", "output", kid, "-p", out, "-o"], check=False)
    path = None
    for dirpath, _, files in os.walk(out):
        if "job_summary.json" in files:
            path = os.path.join(dirpath, "job_summary.json")
            break
    problems = []
    if not path:
        return None, ["job_summary.json missing from the outputs"], out
    with open(path) as f:
        s = json.load(f)
    if s.get("job_id") != job_id:
        problems.append(f"stale output: job_id {s.get('job_id')} != {job_id}")
    if s.get("checked_out_sha") != sha:
        problems.append(f"commit mismatch: {s.get('checked_out_sha')} != {sha}")
    if not any("T4" in g for g in s.get("gpus", [])):
        problems.append(f"expected Tesla T4, got {s.get('gpus')}")
    for st in s.get("steps", []):
        if st.get("rc") != 0:
            problems.append(f"step {st.get('name', st.get('cmd'))} rc={st.get('rc')} "
                            f"timed_out={st.get('timed_out')} log={st.get('log')}")
    if not s.get("ok"):
        problems.append("runner reported ok=false")
    return s, problems, out


def cmd_run(a):
    if a.job not in JOBS:
        raise SystemExit(f"unknown job {a.job}; see 'list'")
    user = kaggle_username()
    branch, sha, url = preconditions(a.allow_dirty)
    q = load_quota()
    used = used_this_week(q)
    est_h = JOBS[a.job]["est_min"] / 60
    print(f"job {a.job}: commit {sha[:10]} on {branch}; estimate {est_h:.2f} GPU-h; "
          f"used this week {used:.2f}/{q['weekly_cap_hours']} h")
    if used + est_h > q["weekly_cap_hours"]:
        raise SystemExit("weekly GPU budget would be exceeded")
    if not a.approved:
        raise SystemExit("not submitted: re-run with --approved after reading the line above")
    job_id = f"{a.job}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{sha[:8]}"
    d, kid = build(a.job, job_id, user, branch, sha, url)
    if kernel_status(kid) in ("queued", "running"):
        raise SystemExit(f"{kid} is already queued or running")
    t0 = time.time()
    status, summary, problems, out = "not_pushed", None, [], None
    try:
        push(d)
        status = wait(kid, a.max_minutes)
        summary, problems, out = fetch_and_validate(kid, job_id, sha)
    finally:
        secs = (summary or {}).get("seconds") or (time.time() - t0)
        record_quota(dict(job_id=job_id, job=a.job, kernel=kid, sha=sha, status=status,
                          gpu_hours=round(secs / 3600, 4),
                          finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    print(f"\nstatus: {status}; outputs: {out}")
    if summary:
        for st in summary.get("steps", []):
            print(f"  {st.get('name', st.get('cmd'))[:40]:40s} rc={st['rc']} {st['seconds']}s")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("VALID: job_summary matches the job id, the commit and the GPU; all steps rc=0")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("job")
    r.add_argument("--approved", action="store_true")
    r.add_argument("--max-minutes", type=int, default=180)
    r.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        q = load_quota()
        for k, v in JOBS.items():
            print(f"{k:10s} ~{v['est_min']} min: " + "; ".join(c["name"] for c in v["commands"]))
        print(f"GPU hours used in the last 7 days: {used_this_week(q):.2f} / {q['weekly_cap_hours']}")
        return
    cmd_run(a)


if __name__ == "__main__":
    main()
