"""Watch one existing PBS benchmark and write a report; never submit jobs."""
import argparse
from datetime import datetime
from pathlib import Path
import re
import subprocess
import time


def summarize(logs):
    phases = {}
    for rank, content in enumerate(logs):
        for phase, progress_name in (("train", "train"), ("validation", "valid")):
            results = re.findall(
                rf"BENCH_RESULT phase={phase} rank={rank} measured_steps=(\d+) "
                rf"events=(\d+) seconds=([\d.]+) total_seconds=([\d.]+)", content)
            totals = re.findall(rf"{progress_name}_batch=\d+/(\d+)", content)
            if results and totals:
                steps, events, seconds, total_seconds = map(float, results[-1])
                if steps > 0:
                    phases.setdefault(phase, []).append(dict(
                        rank=rank, seconds_per_batch=seconds / steps,
                        epoch_seconds=int(totals[-1]) * seconds / steps,
                        events=int(events), measured_steps=int(steps),
                        startup_seconds=total_seconds - seconds,
                        rss_mib=max([float(x) for x in re.findall(r'host_tree_rss_mib=([\d.]+)', content)] or [0.])))
    return phases


def summarize_configs(logs):
    configs = {}
    for rank, content in enumerate(logs):
        parts = re.split(r'BENCH_CONFIG chunk=(\d+) buffer=(\d+) workers=(\d+)\n', content)
        if len(parts) == 1:
            configs.setdefault('baseline', [''] * 4)[rank] = content
        for i in range(1, len(parts), 4):
            key = '/'.join(parts[i:i + 3])
            configs.setdefault(key, [''] * 4)[rank] = parts[i + 3]
    return {key: summarize(value) for key, value in configs.items()}


def job_status(job_id):
    for opts in (("-f",), ("-f", "-H")):
        result = subprocess.run(["qstat", *opts, job_id], capture_output=True,
                                text=True, timeout=30)
        state = re.search(r"job_state = (\w+)", result.stdout)
        if state:
            return state[1], result.stdout
    return "unknown", ""


def write_report(job_id, repo, state, status):
    run_dir = Path(f"/work/gw25/w25002/checkpoint/energy_4gpu_{job_id}.opbs")
    paths = [run_dir / "ranks" / f"rank{rank}.log" for rank in range(4)]
    logs = [p.read_text(errors="replace") if p.exists() else "" for p in paths]
    configs = summarize_configs(logs)
    lines = [f"# Miyabi 4 GPU benchmark — job {job_id}", "",
             f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
             f"PBS state: {state}", "",
             "Data: 1,960,000 training events; 20,000 validation events.",
             "4 nodes × 1 GPU; 32 events/GPU/batch; FP32; full beta and energy losses.",
             "Loader sweep (chunk/buffer/workers): 32/256/2, 8/256/2, 8/256/4, 8/128/4.",
             "Each configuration/phase: 2 warm-up batches + 10 measured batches per rank.",
             "Each configuration resets model, optimizer and Torch RNG; fixed learning rate for timing.",
             "No training checkpoints are saved by this benchmark.", ""]
    for key in ("estimated.start_time", "stime", "resources_used.walltime", "Exit_status"):
        value = re.search(rf"^\s*{re.escape(key)} = (.+)$", status, re.M)
        if value:
            lines.append(f"{key}: {value[1]}")
    finished = {key: phases for key, phases in configs.items()
                if all(len(phases.get(p, [])) == 4 for p in ('train', 'validation'))}
    complete = len(finished) == 4
    if finished:
        lines += ["", f"Completed configurations: {len(finished)}/4.", "",
                  "| Chunk/buffer/workers | Train s/batch (slowest) | Warm-up train s (slowest) | Host tree RSS GiB (max) | Estimated epoch h | 500 epochs days |",
                  "|---|---:|---:|---:|---:|---:|"]
        epochs = {}
        for key, phases in finished.items():
            epoch = sum(max(r['epoch_seconds'] for r in phases[p]) for p in ('train', 'validation'))
            epochs[key] = epoch
            speed = max(r['seconds_per_batch'] for r in phases['train'])
            warmup = max(r['startup_seconds'] for r in phases['train'])
            rss = max(r['rss_mib'] for p in phases.values() for r in p) / 1024
            lines.append(f"| {key} | {speed:.3f} | {warmup:.1f} | {rss:.2f} | {epoch / 3600:.3f} | {epoch * 500 / 86400:.2f} |")
        if complete:
            best = min(epochs, key=epochs.get)
            lines += ["", f"Fastest measured candidate (provisional): **{best}**."]
        lines += ["", "Host tree RSS sums parent/worker RSS and may double-count shared pages. Warm-up includes loading and two batches.",
                  "", "This is a small-sample projection, not a completed-epoch measurement. "
                  "It excludes queue wait, initial loading, checkpoint I/O and later data-chunk loading. "
                  "Later candidates benefit from warmed filesystem caches. Event order/mixing changes with loader settings; "
                  "a longer chunk-boundary test and training-quality check are needed before calling any setting optimal. "
                  "The projection uses the slowest rank per phase and its full batch count."]
    else:
        lines += ["", "Timing results are not complete yet; no training-duration estimate is available."]
    output_log = repo / f"pfa-bench4.o{job_id}"
    lines += ["", f"PBS output: `{output_log}`", f"Rank logs: `{run_dir / 'ranks'}`"]
    if state == "F" and not complete:
        lines += ["", "The job finished without completing all loader comparisons. Check the logs below."]
        for path, content in [(output_log, output_log.read_text(errors='replace') if output_log.exists() else ''), *zip(paths, logs)]:
            if content:
                lines += ["", f"## {path.name}", "```text", *content.splitlines()[-30:], "```"]
    report = repo / f"benchmark_miyabi_4gpu_{job_id}.md"
    temporary = report.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(report)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not args.job_id.isdigit():
        parser.error('Use the numeric PBS job ID')
    repo = Path(__file__).resolve().parents[1]
    deadline = time.monotonic() + 24 * 3600
    while True:
        try:
            state, status = job_status(args.job_id)
            complete = write_report(args.job_id, repo, state, status)
            print(datetime.now().isoformat(timespec='seconds'), state, 'timing_complete=', complete, flush=True)
            if state == 'F' or args.once or time.monotonic() >= deadline:
                return
        except (OSError, subprocess.TimeoutExpired) as error:
            print(type(error).__name__, str(error), flush=True)
            if args.once or time.monotonic() >= deadline:
                raise
        time.sleep(45)


if __name__ == '__main__':
    main()
