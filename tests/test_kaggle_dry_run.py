"""
Dry run of the whole Kaggle path with a fake kaggle CLI: generate -> push (only run.py is
"uploaded") -> the generated script clones a real git repo at the exact commit, runs the job,
writes job_summary.json -> status -> output -> validation -> quota ledger.
"""
import importlib
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
FAKES = os.path.join(HERE, "fakes")


def sh(*cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    sh("git", "init", "--bare", "-q", str(origin), cwd=tmp_path)
    sh("git", "clone", "-q", str(origin), str(work), cwd=tmp_path)
    for k, v in (("user.email", "t@example.com"), ("user.name", "t")):
        sh("git", "config", k, v, cwd=work)
    (work / "hello.txt").write_text("hi\n")
    (work / ".gitignore").write_text("kaggle/build/\nbench/\n")
    sh("git", "add", ".", cwd=work)
    sh("git", "commit", "-q", "-m", "init", cwd=work)
    sh("git", "branch", "-M", "main", cwd=work)
    sh("git", "push", "-q", "-u", "origin", "main", cwd=work)
    monkeypatch.setenv("MDMS_ROOT", str(work))
    monkeypatch.setenv("MDMS_KAGGLE_BIN", os.path.join(FAKES, "kaggle"))
    monkeypatch.setenv("FAKE_KAGGLE_STATE", str(tmp_path / "kaggle_state"))
    monkeypatch.setenv("PATH", FAKES + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    os.makedirs(tmp_path / "kaggle_state")
    sys.path.insert(0, os.path.join(REPO_ROOT, "kaggle"))
    import kaggle_job
    kj = importlib.reload(kaggle_job)
    # origin is a local bare repo: let the runner clone it via file://
    kj.https_url = lambda url: "file://" + str(origin)
    kj.JOBS["testjob"] = dict(est_min=1, collect=["out"], commands=[
        dict(name="write", cmd="mkdir -p out && python3 -c \"open('out/result.txt','w').write(open('hello.txt').read())\"",
             timeout_sec=30)])
    kj.JOBS["slowjob"] = dict(est_min=1, collect=["out"], stop_on_failure=False, commands=[
        dict(name="sleep", cmd="sleep 30", timeout_sec=1)])
    kj.wait = lambda kid, max_minutes: kj.kernel_status(kid)   # fake runs synchronously
    yield kj, work, tmp_path
    sys.path.remove(os.path.join(REPO_ROOT, "kaggle"))


@pytest.mark.fast
def test_full_flow_valid(env, capsys):
    kj, work, tmp = env
    kj.main(["run", "testjob", "--approved", "--max-minutes", "1"])
    out = capsys.readouterr().out
    assert "VALID" in out
    raw = os.path.join(work, "bench", "raw")
    (job_dir,) = os.listdir(raw)
    summ = json.load(open(os.path.join(raw, job_dir, "out", "job_summary.json")))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=work, capture_output=True, text=True).stdout.strip()
    assert summ["checked_out_sha"] == sha and summ["job_id"] == job_dir and summ["ok"]
    assert open(os.path.join(raw, job_dir, "out", "out", "result.txt")).read() == "hi\n"
    # only run.py was uploaded, and it carries the job config
    build = os.path.join(work, "kaggle", "build", job_dir)
    meta = json.load(open(os.path.join(build, "kernel-metadata.json")))
    assert meta["id"] == "tester/mdms-testjob" and meta["code_file"] == "run.py"
    assert job_dir in open(os.path.join(build, "run.py")).read()
    q = json.load(open(os.path.join(work, "bench", "quota.json")))
    assert q["jobs"][-1]["job_id"] == job_dir


@pytest.mark.fast
def test_requires_approval(env):
    kj, work, tmp = env
    with pytest.raises(SystemExit, match="approved"):
        kj.main(["run", "testjob"])
    assert not os.path.exists(os.path.join(tmp, "kaggle_state", "tester__mdms-testjob"))


@pytest.mark.fast
def test_rejects_dirty_tree_and_unpushed_head(env):
    kj, work, tmp = env
    (work / "new.txt").write_text("x")
    with pytest.raises(SystemExit, match="not clean"):
        kj.main(["run", "testjob", "--approved"])
    sh("git", "add", ".", cwd=work)
    sh("git", "commit", "-q", "-m", "x", cwd=work)
    with pytest.raises(SystemExit, match="push first"):
        kj.main(["run", "testjob", "--approved"])


@pytest.mark.fast
def test_timeout_kills_and_fails_validation(env, capsys):
    kj, work, tmp = env
    with pytest.raises(SystemExit):
        kj.main(["run", "slowjob", "--approved"])
    out = capsys.readouterr().out
    assert "timed_out=True" in out


@pytest.mark.fast
def test_stale_output_detected(env):
    kj, work, tmp = env
    kj.main(["run", "testjob", "--approved"])
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=work, capture_output=True, text=True).stdout.strip()
    s, problems, _ = kj.fetch_and_validate("tester/mdms-testjob", "some-other-job-id", sha,
                                           out_root=str(tmp / "o2"))
    assert any("stale" in p for p in problems)


@pytest.mark.fast
def test_username_never_from_git_remote(env, monkeypatch):
    kj, work, tmp = env
    monkeypatch.setenv("MDMS_KAGGLE_BIN", "/nonexistent-kaggle")
    monkeypatch.setattr(kj, "KAGGLE", "/bin/false")
    with pytest.raises(SystemExit, match="KAGGLE_USERNAME"):
        kj.kaggle_username()
