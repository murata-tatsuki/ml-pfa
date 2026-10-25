#!/usr/bin/env python3
"""Run frozen ROOT jobs with several existing writers per selected GPU."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import socket
import sys

sys.dont_write_bytecode = True
import run_server as legacy
from scan_common import (completed_state, load_plan, state_path,
                         verify_implementation)
import epoch_jobs

epoch_jobs.install()
verify_implementation = epoch_jobs.verify_implementation


class ParallelRunner(legacy.Runner):
    """Keep the existing job lifecycle, sharing one GPU lease across its slots."""

    def consume(self, jobs, gpu):
        while not self.stop.is_set():
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                result = self.run_job(job, gpu)
                with self.guard:
                    self.counts[result] += 1
            finally:
                jobs.task_done()

    def worker(self, jobs, gpu):
        token = re.sub(r'[^a-zA-Z0-9_-]', '_', socket.gethostname() + '_' + gpu)
        lock = Path(self.plan['scan_dir']) / 'state' / 'gpu_locks' / (token + '.lock')
        owner = dict(pid=os.getpid(), started=legacy.utc_now(),
                     workers_per_gpu=self.args.workers_per_gpu)
        with legacy.exclusive_lock(lock, owner) as acquired:
            if not acquired:
                self.report(f'[busy] Another runner owns GPU {gpu} on this host for this scan')
                with self.guard:
                    self.counts['busy_gpu'] += 1
                return
            with ThreadPoolExecutor(max_workers=self.args.workers_per_gpu) as pool:
                futures = [pool.submit(self.consume, jobs, gpu)
                           for _ in range(self.args.workers_per_gpu)]
                for future in futures:
                    future.result()


def parser():
    result = legacy.parser()
    result.description = __doc__
    result.add_argument('--workers-per-gpu', type=int, default=1,
                        help='Concurrent save_root.py processes on each GPU (default: 1)')
    return result


def selected_jobs(plan, args, argument_parser):
    jobs = [job for job in plan['jobs']
            if args.server is None or job['server'] == args.server]
    if args.job_id:
        selected = set(args.job_id)
        jobs = [job for job in jobs if job['id'] in selected]
        if {job['id'] for job in jobs} != selected:
            argument_parser.error('Some --job-id values are absent or assigned to a different server')
    if not jobs:
        argument_parser.error('No matching jobs; check --server and --job-id')
    return jobs[:args.max_jobs] if args.max_jobs else jobs


def show_status(plan, jobs):
    counts = Counter()
    for job in jobs:
        try:
            state = json.loads(state_path(plan, job).read_text())
            status = state.get('status', 'unknown')
            if status == 'completed' and not completed_state(plan, job):
                status = 'outputs_changed'
        except (OSError, ValueError):
            status = 'pending'
        counts[(job['server'], status)] += 1
    for (server, status), count in sorted(counts.items()):
        print(f'{server}\t{status}\t{count}')


def run_jobs(plan, jobs, args, gpus):
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    runner = ParallelRunner(plan, args)
    interrupted_signal = [signal.SIGINT]
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def terminate(signum, frame):
        interrupted_signal[0] = signum
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    pool = ThreadPoolExecutor(max_workers=len(gpus))
    try:
        futures = [pool.submit(runner.worker, pending, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    except KeyboardInterrupt:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        runner.report('Stopping new jobs and interrupting active writers...')
        runner.interrupt()
        return 128 + interrupted_signal[0]
    except BaseException:
        runner.interrupt()
        raise
    finally:
        pool.shutdown(wait=True)
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    print(json.dumps(dict(runner.counts), sort_keys=True), flush=True)
    return int(bool(pending.qsize() or any(runner.counts[name]
               for name in ('failed', 'interrupted', 'busy', 'busy_gpu'))))


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if min(args.workers_per_gpu, args.pid_threads, args.root_chunk_events) < 1 or args.pipeline_depth < 0:
        argument_parser.error('Workers/threads/chunk size must be positive; pipeline depth must be nonnegative')
    if args.max_jobs is not None and args.max_jobs < 1:
        argument_parser.error('--max-jobs must be positive')
    plan = load_plan(args.plan)
    jobs = selected_jobs(plan, args, argument_parser)
    if args.status:
        show_status(plan, jobs)
        return 0
    if not args.server or not args.gpus:
        argument_parser.error('--server and --gpus are required for execution/dry-run')
    gpus = [gpu.strip() for gpu in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or any(
            not re.fullmatch(r'(?:[0-9]+|GPU-[a-zA-Z0-9-]+|MIG-[a-zA-Z0-9/-]+|cpu)', gpu)
            for gpu in gpus) or ('cpu' in gpus and len(gpus) != 1):
        argument_parser.error('Use unique GPU numbers/UUIDs, or the single value cpu')
    verify_implementation(plan)
    print(f"[parallel] GPUs={','.join(gpus)} workers-per-gpu={args.workers_per_gpu} "
          f"max-writers={len(gpus) * args.workers_per_gpu} jobs={len(jobs)}", flush=True)
    if args.dry_run:
        for index, job in enumerate(jobs):
            gpu = gpus[index % len(gpus)]
            prefix = '' if gpu == 'cpu' else 'CUDA_VISIBLE_DEVICES=' + shlex.quote(gpu) + ' '
            print(f"# {job['id']} {job['model']} epoch={job['epoch']}")
            print(prefix + shlex.join(legacy.command_for(plan, job, args, gpu)))
        return 0
    return run_jobs(plan, jobs, args, gpus)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('error: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
