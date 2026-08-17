#!/usr/bin/env python3
"""Poll this repository, benchmark mmq-style norm updates, and push results."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


REMOTE = "origin"
BENCHMARK_SCRIPT = "bench_mmq_style_norm_after_attn_npu.py"
OUTPUT_CSV = "mmq_style_norm_after_attn_all.csv"
ERROR_LOG = "mmq_style_norm_after_attn_run_error.log"
DEFAULT_INTERVAL_SECONDS = 60
AUTO_COMMIT_MARKER = "Auto-Benchmark: true"


class GitCommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class PendingPush:
    branch: str
    base_sha: str
    commit_sha: str
    created_commit: bool


REPO = Path(__file__).resolve().parent
BENCHMARK_PATH = REPO / BENCHMARK_SCRIPT
CSV_PATH = REPO / OUTPUT_CSV
ERROR_PATH = REPO / ERROR_LOG


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def run_command(
    command: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=REPO,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture_output,
        check=False,
    )
    if check and result.returncode != 0:
        rendered = shlex.join(command)
        detail = (result.stderr or result.stdout or "no error output").strip()
        raise GitCommandError(f"command failed ({result.returncode}): {rendered}\n{detail}")
    return result


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command(["git", *args], check=check)


def git_text(*args: str) -> str:
    return git(*args).stdout.strip()


def current_branch() -> str:
    branch = git_text("symbolic-ref", "--quiet", "--short", "HEAD")
    if not branch:
        raise GitCommandError("HEAD is detached; check out a branch first")
    return branch


def fetch(branch: str) -> None:
    git("fetch", "--quiet", REMOTE, branch)


def remote_sha(branch: str) -> str:
    return git_text("rev-parse", f"refs/remotes/{REMOTE}/{branch}")


def is_ancestor(older: str, newer: str) -> bool:
    return git("merge-base", "--is-ancestor", older, newer, check=False).returncode == 0


def pull_if_updated(branch: str) -> str | None:
    fetch(branch)
    local = git_text("rev-parse", "HEAD")
    remote = remote_sha(branch)
    if local == remote:
        return None
    if is_ancestor(local, remote):
        log(f"remote update found: {local[:12]} -> {remote[:12]}")
        git("pull", "--ff-only", REMOTE, branch)
        pulled = git_text("rev-parse", "HEAD")
        log(f"git pull completed at {pulled[:12]}")
        return pulled
    if is_ancestor(remote, local):
        log("local branch is ahead of origin; no new remote commit to run")
        return None
    raise GitCommandError("local and remote branches diverged; resolve manually")


def benchmark_command(device: str) -> list[str]:
    python_executable = os.environ.get("BENCH_PYTHON", sys.executable)
    return [
        python_executable,
        BENCHMARK_SCRIPT,
        "--mode",
        "both",
        "--cases",
        "all",
        "--scope",
        "kernel",
        "--device",
        device,
        "--event-diagnostic",
        "off",
        "--capture-msprof-op",
        "on",
        "--output-csv",
        OUTPUT_CSV,
    ]


def write_error_log(command: Sequence[str], returncode: int, output: str, reason: str) -> None:
    content = (
        f"time: {now()}\n"
        f"command: {shlex.join(command)}\n"
        f"return_code: {returncode}\n"
        f"reason: {reason}\n\n"
        "===== stdout + stderr =====\n"
        f"{output}"
    )
    if output and not output.endswith("\n"):
        content += "\n"
    temporary_path = ERROR_PATH.with_suffix(ERROR_PATH.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, ERROR_PATH)


def run_benchmark(device: str) -> bool:
    """Do no polling while the benchmark or error-file write is in progress."""
    command = benchmark_command(device)
    CSV_PATH.unlink(missing_ok=True)
    log(f"starting benchmark: {shlex.join(command)}")
    returncode = -1
    captured = ""
    launch_error = ""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stream:
        try:
            result = subprocess.run(
                command,
                cwd=REPO,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
            returncode = result.returncode
        except OSError as exc:
            launch_error = f"could not start benchmark: {exc}"
        finally:
            stream.seek(0)
            captured = stream.read()

    csv_is_valid = CSV_PATH.is_file() and CSV_PATH.stat().st_size > 0
    if returncode == 0 and csv_is_valid and not launch_error:
        ERROR_PATH.unlink(missing_ok=True)
        log(f"benchmark succeeded; generated {OUTPUT_CSV}")
        return True
    CSV_PATH.unlink(missing_ok=True)
    if launch_error:
        reason = launch_error
    elif returncode != 0:
        reason = f"benchmark exited with status {returncode}"
    else:
        reason = f"benchmark did not create a non-empty {OUTPUT_CSV}"
    write_error_log(command, returncode, captured, reason)
    log(f"benchmark failed; details written to {ERROR_LOG}")
    return False


def artifact_status_paths() -> list[str]:
    return [
        path
        for path in (OUTPUT_CSV, ERROR_LOG)
        if git("status", "--porcelain", "--", path).stdout.strip()
    ]


def commit_artifacts(branch: str, base_sha: str, succeeded: bool) -> PendingPush:
    changed = artifact_status_paths()
    created_commit = False
    if changed:
        git("add", "-A", "--", *changed)
        result_word = "success" if succeeded else "failure"
        message = (
            f"chore: update mmq norm benchmark {result_word}\n\n"
            f"Benchmark-Base: {base_sha}\n"
            f"{AUTO_COMMIT_MARKER}"
        )
        git("commit", "--only", "-m", message, "--", *changed)
        created_commit = True
        log(f"committed benchmark artifacts: {', '.join(changed)}")
    else:
        log("benchmark artifacts are unchanged; no new commit was needed")
    return PendingPush(
        branch=branch,
        base_sha=base_sha,
        commit_sha=git_text("rev-parse", "HEAD"),
        created_commit=created_commit,
    )


def restore_artifacts_from_head() -> None:
    for name, path in ((OUTPUT_CSV, CSV_PATH), (ERROR_LOG, ERROR_PATH)):
        tracked = git("ls-files", "--error-unmatch", "--", name, check=False).returncode == 0
        if tracked:
            git("restore", "--source=HEAD", "--staged", "--worktree", "--", name)
        else:
            git("reset", "--quiet", "HEAD", "--", name, check=False)
            path.unlink(missing_ok=True)


def rewind_own_commit(pending: PendingPush) -> None:
    if not pending.created_commit:
        return
    git("update-ref", f"refs/heads/{pending.branch}", pending.base_sha, pending.commit_sha)
    restore_artifacts_from_head()
    log("remote changed before push; discarded the stale auto-commit")


def try_push(pending: PendingPush) -> tuple[PendingPush | None, bool]:
    result = git("push", REMOTE, f"HEAD:refs/heads/{pending.branch}", check=False)
    if result.returncode == 0:
        log(f"git push {REMOTE} completed")
        return None, False
    detail = (result.stderr or result.stdout or "no error output").strip()
    log(f"git push failed: {detail}")
    try:
        fetch(pending.branch)
        latest_remote = remote_sha(pending.branch)
    except GitCommandError as exc:
        log(f"could not verify remote; will retry in one minute: {exc}")
        return pending, False
    if latest_remote != pending.base_sha:
        rewind_own_commit(pending)
        return None, True
    log("remote did not move; retaining the result commit for a push retry")
    return pending, False


def detect_interrupted_pending_push(branch: str) -> PendingPush | None:
    message = git_text("log", "-1", "--format=%B")
    if AUTO_COMMIT_MARKER not in message:
        return None
    commit_sha = git_text("rev-parse", "HEAD")
    parent = git("rev-parse", "HEAD^", check=False)
    if parent.returncode != 0:
        return None
    base_sha = parent.stdout.strip()
    fetch(branch)
    latest_remote = remote_sha(branch)
    if latest_remote == commit_sha or is_ancestor(commit_sha, latest_remote):
        return None
    log("found an unpushed automatic benchmark commit from an earlier run")
    return PendingPush(branch, base_sha, commit_sha, True)


def validate_repository() -> None:
    if git("rev-parse", "--is-inside-work-tree", check=False).stdout.strip() != "true":
        raise GitCommandError(f"{REPO} is not inside a Git work tree")
    root = Path(git_text("rev-parse", "--show-toplevel")).resolve()
    if root != REPO:
        raise GitCommandError(f"put this monitor at repository root {root}")
    if not BENCHMARK_PATH.is_file():
        raise FileNotFoundError(f"benchmark script not found: {BENCHMARK_PATH}")
    if git("remote", "get-url", REMOTE, check=False).returncode != 0:
        raise GitCommandError(f"Git remote {REMOTE!r} is not configured")
    for key in ("user.name", "user.email"):
        if not git("config", "--get", key, check=False).stdout.strip():
            raise GitCommandError(f"Git {key} must be configured")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="benchmark synchronized HEAD immediately, then keep monitoring",
    )
    parser.add_argument("--device", default="npu:5")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    return args


def main() -> int:
    args = parse_args()
    validate_repository()
    branch = current_branch()
    log(
        f"monitoring {REMOTE}/{branch} every {args.interval:g}s; "
        f"benchmark={BENCHMARK_SCRIPT}, device={args.device}, artifact={OUTPUT_CSV}"
    )
    pending = detect_interrupted_pending_push(branch)
    force_run = args.run_now and pending is None
    while True:
        retry_immediately = False
        try:
            if pending is not None:
                pending, retry_immediately = try_push(pending)
            else:
                base_sha = pull_if_updated(branch)
                if force_run and base_sha is None:
                    local = git_text("rev-parse", "HEAD")
                    remote = remote_sha(branch)
                    if local != remote:
                        raise GitCommandError(
                            "--run-now requires local HEAD to equal origin"
                        )
                    base_sha = local
                    log(f"--run-now selected current HEAD {base_sha[:12]}")
                force_run = False
                if base_sha is not None:
                    succeeded = run_benchmark(args.device)
                    fetch(branch)
                    if remote_sha(branch) != base_sha:
                        restore_artifacts_from_head()
                        log("remote changed during benchmark; rerunning newest commit")
                        retry_immediately = True
                    else:
                        pending = commit_artifacts(branch, base_sha, succeeded)
                        pending, retry_immediately = try_push(pending)
        except (GitCommandError, OSError) as exc:
            log(f"cycle error: {exc}")
        if retry_immediately:
            continue
        if args.once:
            return 0 if pending is None else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("stopped by user")
        raise SystemExit(130)
