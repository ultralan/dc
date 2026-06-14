from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch and restart the RealMAN 8ch downloader.")
    parser.add_argument("--progress-jsonl", type=Path, default=Path("realman_demo/realman_ring2_8ch_progress.jsonl"))
    parser.add_argument("--watch-log", type=Path, default=Path("realman_demo/realman_ring2_8ch_watchdog.jsonl"))
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--connections", type=int, default=4)
    parser.add_argument("--check-interval", type=float, default=120.0)
    parser.add_argument("--max-silent-seconds", type=float, default=600.0)
    parser.add_argument("--min-mib-s", type=float, default=2.0)
    parser.add_argument("--slow-checks", type=int, default=3)
    return parser.parse_args()


def emit(path: Path, event: str, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_latest_event(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    latest: dict[str, object] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest = record
    return latest


def read_latest_progress(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    latest: dict[str, object] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "download_progress":
                latest = record
    return latest


def event_age_seconds(record: dict[str, object] | None) -> float | None:
    if not record or not isinstance(record.get("ts"), str):
        return None
    try:
        ts = datetime.fromisoformat(record["ts"])
    except ValueError:
        return None
    return (datetime.now() - ts).total_seconds()


def matching_pids() -> list[int]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -in @('python.exe','curl.exe','tar.exe') -and "
        "($_.CommandLine -like '*prepare_realman_ring2_8ch.py*' -or "
        "$_.CommandLine -like '*datasets/AISHELL/RealMAN*' -or "
        "$_.CommandLine -like '*realman_demo\\\\tmp_download*') } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def stop_download_tree() -> None:
    pids = matching_pids()
    if not pids:
        return
    joined = ",".join(str(pid) for pid in pids)
    command = f"Stop-Process -Id {joined} -Force -ErrorAction SilentlyContinue"
    subprocess.run(["powershell", "-NoProfile", "-Command", command], check=False)


def start_downloader(args: argparse.Namespace) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/prepare_realman_ring2_8ch.py",
        "--retries",
        "20",
        "--retry-sleep",
        "10",
        "--proxy",
        args.proxy,
        "--connections",
        str(args.connections),
        "--progress-jsonl",
        str(args.progress_jsonl),
        "--progress-interval",
        "10",
        "--stall-timeout",
        "180",
    ]
    creationflags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags |= subprocess.CREATE_NO_WINDOW
    subprocess.Popen(cmd, creationflags=creationflags)


def main() -> None:
    args = parse_args()
    slow_count = 0
    emit(
        args.watch_log,
        "watchdog_start",
        proxy=args.proxy,
        connections=args.connections,
        min_mib_s=args.min_mib_s,
        max_silent_seconds=args.max_silent_seconds,
    )
    while True:
        latest = read_latest_event(args.progress_jsonl)
        latest_progress = read_latest_progress(args.progress_jsonl)
        if latest and latest.get("event") == "run_done":
            emit(args.watch_log, "watchdog_done")
            return

        pids = matching_pids()
        latest_event_age = event_age_seconds(latest)
        latest_progress_age = event_age_seconds(latest_progress)
        avg_mib_s = latest_progress.get("avg_mib_s") if latest_progress else None
        archive_index = latest_progress.get("archive_index") if latest_progress else None
        rel = latest_progress.get("rel") if latest_progress else None
        proxy = latest_progress.get("proxy") if latest_progress else None

        reason: str | None = None
        if not pids:
            reason = "no_process"
        elif latest_event_age is not None and latest_event_age > args.max_silent_seconds:
            reason = "silent"
        elif isinstance(avg_mib_s, (int, float)) and avg_mib_s < args.min_mib_s:
            slow_count += 1
            if slow_count >= args.slow_checks:
                reason = "slow"
        else:
            slow_count = 0

        emit(
            args.watch_log,
            "watchdog_check",
            pids=pids,
            latest_event_age_s=round(latest_event_age, 1) if latest_event_age is not None else None,
            latest_progress_age_s=round(latest_progress_age, 1) if latest_progress_age is not None else None,
            avg_mib_s=avg_mib_s,
            archive_index=archive_index,
            rel=rel,
            proxy=proxy,
            slow_count=slow_count,
            reason=reason,
        )

        if reason:
            stop_download_tree()
            time.sleep(5)
            start_downloader(args)
            slow_count = 0
            emit(args.watch_log, "watchdog_restart", reason=reason)

        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
