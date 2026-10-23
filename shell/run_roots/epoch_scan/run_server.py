#!/usr/bin/env python3
"""Run one server's immutable jobs, with one subprocess per selected GPU."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
from scan_common import (atomic_json, completed_state, expected_outputs,
                         file_identity, job_digest, load_plan, sha256_file,
                         state_path, verify_implementation)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive_lock(path, owner):
    """Never unlink a lock: a new inode could admit a concurrent second owner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+', encoding='utf-8') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            stream.seek(0)
            stream.truncate()
            json.dump(owner, stream)
            stream.flush()
            yield True
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def command_for(plan, job, args, gpu):
    command = [args.python, '-B', str(Path(plan['repo']) / 'save_root.py'),
               '@' + job['input_list'], job['checkpoint'], job['output_dir'],
               str(job['nstart']), str(job['nend']), 'False',
               str(job['input_dim']), str(job['output_dim']), '--multi-h5']
    command += job['writer_args']
    command += ['--device', 'cpu' if gpu == 'cpu' else 'cuda:0',
                '--pid-threads', str(args.pid_threads),
                '--root-chunk-events', str(args.root_chunk_events),
                '--pipeline-depth', str(args.pipeline_depth)]
    if args.replace:
        command.append('--replace')
    return command


class Runner:
    def __init__(self, plan, args):
        self.plan, self.args = plan, args
        self.stop = threading.Event()
        self.guard = threading.Lock()
        self.hash_guard = threading.Lock()
        self.hashes = {}
        self.active = set()
        self.counts = Counter()

    def report(self, text):
        with self.guard:
            print(text, flush=True)

    def verify_job(self, job):
        verify_implementation(self.plan)
        current = file_identity(job['checkpoint'])
        if current != job['checkpoint_stat']:
            raise ValueError('Checkpoint changed since planning: ' + job['checkpoint'])
        with self.hash_guard:
            key = (job['checkpoint'], current['size'], current['mtime_ns'])
            if key not in self.hashes:
                self.hashes[key] = sha256_file(job['checkpoint'])
            if self.hashes[key] != job['checkpoint_sha256']:
                raise ValueError('Checkpoint SHA256 differs from plan: ' + job['checkpoint'])
        for item in job['inputs']:
            if file_identity(item['path']) != {k: item[k] for k in ('size', 'mtime_ns')}:
                raise ValueError('Input changed since planning: ' + item['path'])
        listed = [line.strip() for line in Path(job['input_list']).read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith('#')]
        if listed != [item['path'] for item in job['inputs']]:
            raise ValueError('Input list differs from plan: ' + job['input_list'])

    def run_job(self, job, gpu):
        owner = dict(host=socket.gethostname(), pid=os.getpid(),
                     server=self.args.server, gpu=gpu, started=utc_now())
        lock = Path(self.plan['scan_dir']) / 'state' / 'locks' / (job['id'] + '.lock')
        with exclusive_lock(lock, owner) as acquired:
            if not acquired:
                self.report('[busy] ' + job['id'])
                return 'busy'
            state = dict(owner, job_id=job['id'], job_digest=job_digest(job), status='running')
            log_dir = Path(self.plan['scan_dir']) / 'logs' / self.args.server
            log_dir.mkdir(parents=True, exist_ok=True)
            log = log_dir / (job['id'] + '.log')
            state['log'] = str(log)
            try:
                self.verify_job(job)
                if not self.args.recheck and not self.args.replace and completed_state(self.plan, job):
                    self.report('[skip] ' + job['id'])
                    return 'skipped'
                atomic_json(state_path(self.plan, job), state)
                command = command_for(self.plan, job, self.args, gpu)
                env = os.environ.copy()
                env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
                if gpu != 'cpu':
                    env['CUDA_VISIBLE_DEVICES'] = gpu
                self.report(f"[start] GPU={gpu} {job['model']} epoch={job['epoch']} "
                            f"{job['energy']}GeV {job['method']} {job['id']}")
                with log.open('a', encoding='utf-8') as stream:
                    stream.write('\n' + utc_now() + '\n' + shlex.join(command) + '\n')
                    stream.flush()
                    with self.guard:
                        if self.stop.is_set():
                            state.update(status='interrupted', finished=utc_now())
                            atomic_json(state_path(self.plan, job), state)
                            return 'interrupted'
                        proc = subprocess.Popen(command, cwd=self.plan['repo'], env=env,
                                                stdout=stream, stderr=subprocess.STDOUT,
                                                start_new_session=True)
                        self.active.add(proc)
                    try:
                        code = proc.wait()
                    finally:
                        with self.guard:
                            self.active.discard(proc)
                if code != 0:
                    raise RuntimeError(f'Writer exited with status {code}; see {log}')
                self.verify_job(job)
                outputs = []
                for path in expected_outputs(job):
                    identity = file_identity(path)
                    if identity['size'] == 0:
                        raise RuntimeError('Writer left an empty output: ' + str(path))
                    outputs.append(dict(path=str(path), identity=identity))
                state.update(status='completed', returncode=0, finished=utc_now(), outputs=outputs)
                atomic_json(state_path(self.plan, job), state)
                self.report('[done] ' + job['id'])
                return 'completed'
            except Exception as error:
                state.update(status='interrupted' if self.stop.is_set() else 'failed',
                             finished=utc_now(), error=str(error))
                atomic_json(state_path(self.plan, job), state)
                self.report(f"[{state['status']}] {job['id']}: {error}")
                if self.args.fail_fast:
                    self.stop.set()
                return state['status']

    def worker(self, jobs, gpu):
        token = re.sub(r'[^a-zA-Z0-9_-]', '_', socket.gethostname() + '_' + gpu)
        lock = Path(self.plan['scan_dir']) / 'state' / 'gpu_locks' / (token + '.lock')
        with exclusive_lock(lock, dict(pid=os.getpid(), started=utc_now())) as acquired:
            if not acquired:
                self.report(f'[busy] Another runner owns GPU {gpu} on this host for this scan')
                with self.guard:
                    self.counts['busy_gpu'] += 1
                return
            while not self.stop.is_set():
                try:
                    job = jobs.get_nowait()
                except queue.Empty:
                    break
                try:
                    result = self.run_job(job, gpu)
                    with self.guard:
                        self.counts[result] += 1
                finally:
                    jobs.task_done()

    def interrupt(self):
        self.stop.set()
        with self.guard:
            active = list(self.active)
        for proc in active:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        for proc in active:
            try:
                proc.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 3
        for proc in active:
            try:
                proc.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--server', help='Assigned server name; need not equal hostname')
    p.add_argument('--gpus', help='Comma-separated CUDA_VISIBLE_DEVICES identifiers, or cpu')
    p.add_argument('--python', default=os.environ.get('PYTHON_BIN', sys.executable))
    p.add_argument('--pid-threads', type=int, default=2)
    p.add_argument('--root-chunk-events', type=int, default=10)
    p.add_argument('--pipeline-depth', type=int, default=40)
    p.add_argument('--max-jobs', type=int)
    p.add_argument('--job-id', action='append', default=[], help='Restrict to this job ID; repeatable')
    p.add_argument('--dry-run', action='store_true', help='Print commands without writing or running')
    p.add_argument('--status', action='store_true', help='Show saved states without running')
    p.add_argument('--recheck', action='store_true', help='Ask writer to revalidate completed outputs')
    p.add_argument('--replace', action='store_true', help='Explicitly regenerate selected ROOTs')
    p.add_argument('--fail-fast', action='store_true')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.pid_threads, args.root_chunk_events) < 1 or args.pipeline_depth < 0:
        p.error('Threads/chunk size must be positive; pipeline depth must be nonnegative')
    if args.max_jobs is not None and args.max_jobs < 1:
        p.error('--max-jobs must be positive')
    plan = load_plan(args.plan)
    jobs = [job for job in plan['jobs'] if args.server is None or job['server'] == args.server]
    if args.job_id:
        selected = set(args.job_id)
        jobs = [job for job in jobs if job['id'] in selected]
        if {job['id'] for job in jobs} != selected:
            p.error('Some --job-id values are absent or assigned to a different server')
    if not jobs:
        p.error('No matching jobs; check --server and --job-id')
    if args.max_jobs:
        jobs = jobs[:args.max_jobs]
    if args.status:
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
        return 0
    if not args.server or not args.gpus:
        p.error('--server and --gpus are required for execution/dry-run')
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or any(
            not re.fullmatch(r'(?:[0-9]+|GPU-[a-zA-Z0-9-]+|MIG-[a-zA-Z0-9/-]+|cpu)', gpu)
            for gpu in gpus) or ('cpu' in gpus and len(gpus) != 1):
        p.error('Use unique GPU numbers/UUIDs, or the single value cpu')
    verify_implementation(plan)
    if args.dry_run:
        for index, job in enumerate(jobs):
            gpu = gpus[index % len(gpus)]
            prefix = '' if gpu == 'cpu' else 'CUDA_VISIBLE_DEVICES=' + shlex.quote(gpu) + ' '
            print(f"# {job['id']} {job['model']} epoch={job['epoch']}")
            print(prefix + shlex.join(command_for(plan, job, args, gpu)))
        return 0
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    runner = Runner(plan, args)
    interrupted_signal = [signal.SIGINT]
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def terminate(signum, frame):
        interrupted_signal[0] = signum
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
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


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('error: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
