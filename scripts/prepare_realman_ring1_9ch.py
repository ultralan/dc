from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


DEFAULT_REPO_BASE = "https://hf-mirror.com/datasets/AISHELL/RealMAN/resolve/main"
CHANNEL_IDS = tuple(range(0, 9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract RealMAN ring1 9ch data (CH0-8, ring1+center).")
    parser.add_argument("--raw-dir", type=Path, default=Path("realman_demo/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("D:/RealMAN/ring1_9ch/extracted"))
    parser.add_argument("--temp-dir", type=Path, default=Path("D:/RealMAN/ring1_9ch/tmp_download"))
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        default=[
            Path("realman_demo/raw/merged/train_val_moving_source_location.csv"),
            Path("realman_demo/raw/merged/train_val_static_source_location.csv"),
        ],
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--speech-kind",
        choices=["csv", "ma_noisy_speech", "ma_speech", "all"],
        default="csv",
        help="csv uses the speech archive kind referenced by the location CSV.",
    )
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-scenes", nargs="*", default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--repo-base", default=DEFAULT_REPO_BASE)
    parser.add_argument("--proxy", default=None, help="Optional curl proxy, e.g. http://127.0.0.1:7890")
    parser.add_argument("--progress-jsonl", type=Path, default=Path("realman_demo/realman_ring2_8ch_progress.jsonl"))
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--stall-timeout", type=float, default=180.0)
    parser.add_argument("--connections", type=int, default=1, help="Parallel range connections per archive.")
    return parser.parse_args()


class ProgressLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **payload: object) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)


def size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path.resolve().anchor or ".")
    return round(usage.free / (1024**3), 2)


def completed_rels_from_progress(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "extract_done" and isinstance(record.get("rel"), str):
                completed.add(record["rel"])
    return completed


def required_packages(csv_paths: list[Path], splits: set[str]) -> dict[tuple[str, str], set[str]]:
    packages: dict[tuple[str, str], set[str]] = {}
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                filename = row["filename"].replace("\\", "/")
                parts = filename.split("/")
                if len(parts) < 4:
                    continue
                split, speech_kind, scene = parts[0], parts[1], parts[2]
                if split not in splits:
                    continue
                packages.setdefault((split, "dp_speech"), set()).add(scene)
                packages.setdefault((split, speech_kind), set()).add(scene)
    return packages


def package_url(repo_base: str, split: str, kind: str, scene: str) -> str:
    rel = f"{split}/{kind}/{scene}.rar"
    return f"{repo_base.rstrip('/')}/{quote(rel)}?download=true"


def curl_base_cmd(proxy: str | None) -> list[str]:
    cmd = ["curl.exe"]
    if proxy:
        cmd.extend(["--proxy", proxy])
    return cmd


def probe_remote_size(url: str, proxy: str | None) -> int | None:
    cmd = curl_base_cmd(proxy) + [
        "-L",
        "-I",
        "--connect-timeout",
        "30",
        "--max-time",
        "120",
        url,
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    size: int | None = None
    for line in result.stdout.splitlines():
        lower = line.lower()
        if lower.startswith("x-linked-size:") or lower.startswith("content-length:"):
            try:
                size = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return size


def download_range_chunk(
    url: str,
    proxy: str | None,
    chunk_path: Path,
    start: int,
    end: int,
) -> None:
    expected = end - start + 1
    if chunk_path.exists() and chunk_path.stat().st_size == expected:
        return
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    if chunk_path.exists():
        chunk_path.unlink()
    cmd = curl_base_cmd(proxy) + [
        "-L",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "1024",
        "--speed-time",
        "60",
        "-r",
        f"{start}-{end}",
        "-o",
        str(chunk_path),
        url,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"curl range {start}-{end} failed with exit code {result.returncode}")
    actual = chunk_path.stat().st_size if chunk_path.exists() else 0
    if actual != expected:
        raise RuntimeError(f"curl range {start}-{end} got {actual} bytes, expected {expected}")


def append_file(dest: Path, src: Path) -> None:
    with dest.open("ab") as out_handle, src.open("rb") as in_handle:
        shutil.copyfileobj(in_handle, out_handle, length=1024 * 1024)


def download_archive(
    repo_base: str,
    split: str,
    kind: str,
    scene: str,
    dest: Path,
    *,
    retries: int,
    retry_sleep: float,
    proxy: str | None,
    progress: ProgressLogger,
    progress_context: dict[str, object],
    progress_interval: float,
    stall_timeout: float,
    connections: int,
) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        progress.emit("download_skip_existing", **progress_context, archive=str(dest), size_bytes=size_bytes(dest))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = package_url(repo_base, split, kind, scene)
    for attempt in range(1, retries + 1):
        started = time.monotonic()
        start_size = size_bytes(tmp)
        try:
            remote_size = probe_remote_size(url, proxy) if connections > 1 else None
            progress.emit(
                "download_start",
                **progress_context,
                attempt=attempt,
                retries=retries,
                archive=str(dest),
                part=str(tmp),
                existing_part_bytes=start_size,
                remote_size_bytes=remote_size,
                connections=connections,
                proxy=proxy or "none",
                free_gb=free_gb(dest.parent),
            )
            if connections > 1 and remote_size and start_size < remote_size:
                remaining_start = start_size
                remaining = remote_size - remaining_start
                chunk_count = min(connections, remaining)
                chunk_size = max(1, remaining // chunk_count)
                ranges: list[tuple[int, int, Path]] = []
                for chunk_index in range(chunk_count):
                    chunk_start = remaining_start + chunk_index * chunk_size
                    chunk_end = (
                        remote_size - 1
                        if chunk_index == chunk_count - 1
                        else remaining_start + (chunk_index + 1) * chunk_size - 1
                    )
                    chunk_path = tmp.with_name(f"{tmp.name}.{chunk_start}-{chunk_end}.chunk")
                    ranges.append((chunk_start, chunk_end, chunk_path))
                progress.emit(
                    "download_parallel_start",
                    **progress_context,
                    attempt=attempt,
                    part=str(tmp),
                    existing_part_bytes=start_size,
                    remote_size_bytes=remote_size,
                    remaining_bytes=remaining,
                    chunks=len(ranges),
                    proxy=proxy or "none",
                )
                last_total = start_size + sum(size_bytes(chunk_path) for _, _, chunk_path in ranges)
                last_growth = time.monotonic()
                with ThreadPoolExecutor(max_workers=chunk_count) as executor:
                    futures = [
                        executor.submit(download_range_chunk, url, proxy, chunk_path, chunk_start, chunk_end)
                        for chunk_start, chunk_end, chunk_path in ranges
                    ]
                    while not all(future.done() for future in futures):
                        time.sleep(progress_interval)
                        current_total = start_size + sum(size_bytes(chunk_path) for _, _, chunk_path in ranges)
                        now = time.monotonic()
                        delta = current_total - last_total
                        if delta > 0:
                            last_growth = now
                        elapsed = max(now - started, 1e-6)
                        progress.emit(
                            "download_progress",
                            **progress_context,
                            attempt=attempt,
                            part=str(tmp),
                            size_bytes=current_total,
                            size_gb=round(current_total / (1024**3), 3),
                            delta_bytes=delta,
                            avg_mib_s=round((current_total - start_size) / elapsed / (1024**2), 3),
                            seconds_since_growth=round(now - last_growth, 1),
                            proxy=proxy or "none",
                            connections=chunk_count,
                            free_gb=free_gb(dest.parent),
                        )
                        last_total = current_total
                        if now - last_growth >= stall_timeout:
                            raise RuntimeError(f"parallel download stalled for {stall_timeout}s")
                    for future in as_completed(futures):
                        future.result()
                for _, _, chunk_path in sorted(ranges, key=lambda item: item[0]):
                    append_file(tmp, chunk_path)
                    chunk_path.unlink()
                final_size = size_bytes(tmp)
                if final_size != remote_size:
                    raise RuntimeError(f"parallel download assembled {final_size} bytes, expected {remote_size}")
                tmp.replace(dest)
                progress.emit(
                    "download_done",
                    **progress_context,
                    attempt=attempt,
                    archive=str(dest),
                    size_bytes=final_size,
                    size_gb=round(final_size / (1024**3), 3),
                    elapsed_s=round(time.monotonic() - started, 1),
                    proxy=proxy or "none",
                    connections=chunk_count,
                    free_gb=free_gb(dest.parent),
                )
                return
            cmd = [
                "curl.exe",
                "-L",
                "-C",
                "-",
                "--retry",
                "5",
                "--retry-delay",
                "5",
                "--connect-timeout",
                "30",
                "--speed-limit",
                "1024",
                "--speed-time",
                "60",
                "-o",
                str(tmp),
                url,
            ]
            if proxy:
                cmd[1:1] = ["--proxy", proxy]
            proc = subprocess.Popen(cmd)
            last_size = size_bytes(tmp)
            last_growth = time.monotonic()
            while proc.poll() is None:
                time.sleep(progress_interval)
                current_size = size_bytes(tmp)
                now = time.monotonic()
                delta = current_size - last_size
                if delta > 0:
                    last_growth = now
                elapsed = max(now - started, 1e-6)
                progress.emit(
                    "download_progress",
                    **progress_context,
                    attempt=attempt,
                    part=str(tmp),
                    size_bytes=current_size,
                    size_gb=round(current_size / (1024**3), 3),
                    delta_bytes=delta,
                    avg_mib_s=round((current_size - start_size) / elapsed / (1024**2), 3),
                    seconds_since_growth=round(now - last_growth, 1),
                    proxy=proxy or "none",
                    free_gb=free_gb(dest.parent),
                )
                last_size = current_size
                if now - last_growth >= stall_timeout:
                    progress.emit(
                        "download_stalled_restart",
                        **progress_context,
                        attempt=attempt,
                        part=str(tmp),
                        size_bytes=current_size,
                        stall_timeout=stall_timeout,
                    )
                    proc.terminate()
                    time.sleep(5)
                    if proc.poll() is None:
                        proc.kill()
                    break
            returncode = proc.wait()
            if returncode != 0:
                raise RuntimeError(f"curl failed with exit code {returncode}")
            final_size = size_bytes(tmp)
            tmp.replace(dest)
            progress.emit(
                "download_done",
                **progress_context,
                attempt=attempt,
                archive=str(dest),
                size_bytes=final_size,
                size_gb=round(final_size / (1024**3), 3),
                elapsed_s=round(time.monotonic() - started, 1),
                proxy=proxy or "none",
                free_gb=free_gb(dest.parent),
            )
            return
        except Exception as exc:
            progress.emit(
                "download_retry",
                **progress_context,
                attempt=attempt,
                retries=retries,
                error=str(exc),
                part=str(tmp),
                part_bytes=size_bytes(tmp),
                proxy=proxy or "none",
                free_gb=free_gb(dest.parent),
            )
            if attempt >= retries:
                raise
            time.sleep(retry_sleep)


def archive_members(archive: Path) -> list[str]:
    result = subprocess.run(
        ["tar", "-tf", str(archive)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def selected_members(kind: str, members: list[str]) -> list[str]:
    if kind == "dp_speech":
        return [member for member in members if member.endswith(".flac") and "_CH" not in Path(member).name]
    suffixes = tuple(f"_CH{channel_id}.flac" for channel_id in CHANNEL_IDS)
    return [member for member in members if member.endswith(suffixes)]


def extract_selected(archive: Path, members: list[str], output_dir: Path) -> None:
    if not members:
        print(f"[skip] no selected members in {archive}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = archive.with_suffix(".members.txt")
    manifest.write_text("\n".join(members) + "\n", encoding="utf-8")
    print(f"[extract] {archive.name}: {len(members)} files")
    subprocess.run(
        ["tar", "-xf", str(archive), "-C", str(output_dir), "-T", str(manifest)],
        check=True,
    )
    manifest.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    splits = set(args.splits)
    packages = required_packages(args.csv, splits)
    if args.limit_scenes:
        allowed = set(args.limit_scenes)
        packages = {
            key: {scene for scene in scenes if scene in allowed}
            for key, scenes in packages.items()
        }
    wanted: list[tuple[str, str, str]] = []
    for (split, kind), scenes in sorted(packages.items()):
        if args.speech_kind == "csv":
            include_kind = kind == "dp_speech" or kind in {"ma_noisy_speech", "ma_speech"}
        elif args.speech_kind == "all":
            include_kind = kind in {"dp_speech", "ma_noisy_speech", "ma_speech"}
        else:
            include_kind = kind in {"dp_speech", args.speech_kind}
        if not include_kind:
            continue
        for scene in sorted(scenes):
            wanted.append((split, kind, scene))
    print(f"Need {len(wanted)} archives.")
    if args.dry_run:
        for item in wanted:
            print("/".join(item) + ".rar")
        return
    progress = ProgressLogger(args.progress_jsonl)
    completed_rels = completed_rels_from_progress(args.progress_jsonl)
    progress.emit(
        "run_start",
        total_archives=len(wanted),
        completed_archives=len(completed_rels),
        progress_jsonl=str(args.progress_jsonl),
    )
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    for archive_index, (split, kind, scene) in enumerate(wanted, start=1):
        context = {
            "archive_index": archive_index,
            "total_archives": len(wanted),
            "split": split,
            "kind": kind,
            "scene": scene,
            "rel": f"{split}/{kind}/{scene}.rar",
        }
        archive = args.temp_dir / split / kind / f"{scene}.rar"
        local_archive = args.raw_dir / split / kind / f"{scene}.rar"
        try:
            if context["rel"] in completed_rels:
                progress.emit("archive_skip_completed", **context, free_gb=free_gb(args.output_dir))
                continue
            progress.emit("archive_start", **context, free_gb=free_gb(args.output_dir))
            active_archive = local_archive if local_archive.exists() else archive
            if not active_archive.exists():
                download_archive(
                    args.repo_base,
                    split,
                    kind,
                    scene,
                    archive,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                    proxy=args.proxy,
                    progress=progress,
                    progress_context=context,
                    progress_interval=args.progress_interval,
                    stall_timeout=args.stall_timeout,
                    connections=args.connections,
                )
                active_archive = archive
            else:
                progress.emit(
                    "archive_use_local",
                    **context,
                    archive=str(active_archive),
                    size_bytes=size_bytes(active_archive),
                )
            members = archive_members(active_archive)
            chosen = selected_members(kind, members)
            extract_started = time.monotonic()
            progress.emit("extract_start", **context, archive=str(active_archive), selected_files=len(chosen))
            extract_selected(active_archive, chosen, args.output_dir)
            progress.emit(
                "extract_done",
                **context,
                archive=str(active_archive),
                selected_files=len(chosen),
                elapsed_s=round(time.monotonic() - extract_started, 1),
                free_gb=free_gb(args.output_dir),
            )
        except Exception as exc:
            print(f"[error] {split}/{kind}/{scene}: {exc}")
            progress.emit("archive_error", **context, error=str(exc), free_gb=free_gb(args.output_dir))
            continue
        finally:
            if not args.keep_archives and archive.exists():
                deleted_size = size_bytes(archive)
                archive.unlink()
                progress.emit("archive_deleted", **context, archive=str(archive), size_bytes=deleted_size)
            empty_parent = archive.parent
            while empty_parent != args.temp_dir and empty_parent.exists():
                try:
                    empty_parent.rmdir()
                except OSError:
                    break
                empty_parent = empty_parent.parent
    if args.temp_dir.exists() and not any(args.temp_dir.rglob("*")):
        shutil.rmtree(args.temp_dir)
    progress.emit("run_done", total_archives=len(wanted), free_gb=free_gb(args.output_dir))


if __name__ == "__main__":
    main()
