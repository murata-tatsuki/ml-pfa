#!/usr/bin/env python3
"""Keep a pool of existing ROOT writers alive across checkpoint/epoch jobs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import re
import selectors
import shlex
import signal
import socket
import subprocess
import sys
import threading

sys.dont_write_bytecode = True
import run_parallel as parallel
import run_server as legacy
from scan_common import (atomic_json, completed_state, expected_outputs,
                         file_identity, job_digest, load_plan, state_path,
                         verify_implementation)


class WorkerProcess:
    """One slot: JSON requests on stdin, replies on a separate pipe from ROOT logs."""

    def __init__(self, runner, gpu):
        self.runner, self.gpu = runner, gpu
        self.proc = self.selector = None
        self.reply_fd = None
        self.buffer = b''

    def start(self):
        runner = self.runner
        bootstrap = (Path(runner.plan['scan_dir']) / 'logs' / runner.args.server /
                     'workers' / f'{socket.gethostname()}_{self.gpu}_{threading.get_ident()}.log')
        bootstrap.parent.mkdir(parents=True, exist_ok=True)
        read_fd, write_fd = os.pipe()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        if self.gpu != 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = self.gpu
        command = [runner.args.python, '-B', str(Path(__file__).with_name('persistent_worker.py')),
                   '--repo', runner.plan['repo'], '--reply-fd', str(write_fd)]
        try:
            with runner.guard:
                if runner.stop.is_set():
                    raise InterruptedError('Runner is stopping')
                with bootstrap.open('a') as output:
                    self.proc = subprocess.Popen(command, cwd=runner.plan['repo'], env=env,
                        stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT,
                        text=True, pass_fds=(write_fd,), start_new_session=True)
                runner.active.add(self.proc)
            self.reply_fd = read_fd
            self.selector = selectors.DefaultSelector()
            self.selector.register(read_fd, selectors.EVENT_READ)
        except BaseException:
            if self.reply_fd is None:
                os.close(read_fd)
            self.close(force=True)
            raise
        finally:
            os.close(write_fd)

    def request(self, job, argv, log):
        if self.proc is None or self.proc.poll() is not None:
            self.close(force=True)
            self.start()
        request = dict(job_id=job['id'], argv=argv, log=str(log),
                       writer_mode='epoch' if job.get('output_file') else 'legacy')
        self.proc.stdin.write(json.dumps(request) + '\n')
        self.proc.stdin.flush()
        while True:
            if b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                response = json.loads(line)
                if response.get('job_id') != job['id']:
                    raise RuntimeError('Worker returned a response for another job')
                return response
            if self.runner.stop.is_set():
                raise InterruptedError('Runner is stopping')
            if not self.selector.select(timeout=.2):
                if self.proc.poll() is not None:
                    raise RuntimeError(f'Persistent worker exited with status {self.proc.returncode}; see {log}')
                continue
            block = os.read(self.reply_fd, 65536)
            if not block:
                raise RuntimeError(f'Persistent worker closed its reply pipe; see {log}')
            self.buffer += block
            if len(self.buffer) > 1024 * 1024:
                raise RuntimeError('Persistent worker response is too large')

    def close(self, force=False):
        proc = self.proc
        if proc is not None:
            if proc.poll() is None:
                if force:
                    try:
                        os.killpg(proc.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                else:
                    try:
                        proc.stdin.write('{"shutdown": true}\n')
                        proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait()
            if proc.stdin:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            with self.runner.guard:
                self.runner.active.discard(proc)
        if self.selector is not None:
            self.selector.close()
        if self.reply_fd is not None:
            os.close(self.reply_fd)
        self.proc = self.selector = self.reply_fd = None
        self.buffer = b''


class PersistentRunner(parallel.ParallelRunner):
    def consume(self, jobs, gpu):
        worker = WorkerProcess(self, gpu)
        try:
            while not self.stop.is_set():
                try:
                    job = jobs.get_nowait()
                except queue.Empty:
                    return
                try:
                    result = self.run_persistent_job(job, gpu, worker)
                    with self.guard:
                        self.counts[result] += 1
                finally:
                    jobs.task_done()
        finally:
            worker.close(force=self.stop.is_set())

    def run_persistent_job(self, job, gpu, worker):
        owner = dict(host=socket.gethostname(), pid=os.getpid(), server=self.args.server,
                     gpu=gpu, started=legacy.utc_now(), worker_mode='persistent')
        lock = Path(self.plan['scan_dir']) / 'state' / 'locks' / (job['id'] + '.lock')
        with legacy.exclusive_lock(lock, owner) as acquired:
            if not acquired:
                self.report('[busy] ' + job['id'])
                return 'busy'
            log = Path(self.plan['scan_dir']) / 'logs' / self.args.server / (job['id'] + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            state = dict(owner, job_id=job['id'], job_digest=job_digest(job), status='running', log=str(log))
            try:
                self.verify_job(job)
                if not self.args.recheck and not self.args.replace and completed_state(self.plan, job):
                    self.report('[skip] ' + job['id'])
                    return 'skipped'
                atomic_json(state_path(self.plan, job), state)
                command = legacy.command_for(self.plan, job, self.args, gpu)
                with log.open('a') as stream:
                    stream.write('\n' + legacy.utc_now() + '\n' + shlex.join(command) + '\n')
                self.report(f"[start] GPU={gpu} {job['model']} epoch={job['epoch']} "
                            f"{job['energy']}GeV {job['method']} {job['id']} persistent")
                response = worker.request(job, command[3:], log)
                state['worker_pid'] = response['pid']
                if not response.get('ok'):
                    raise RuntimeError(response.get('error', 'Persistent writer failed') + f'; see {log}')
                self.verify_job(job)
                outputs = []
                for path in expected_outputs(job):
                    identity = file_identity(path)
                    if identity['size'] == 0:
                        raise RuntimeError('Writer left an empty output: ' + str(path))
                    outputs.append(dict(path=str(path), identity=identity))
                state.update(status='completed', returncode=0, finished=legacy.utc_now(),
                             outputs=outputs, model_cache=response.get('cache', {}))
                atomic_json(state_path(self.plan, job), state)
                self.report(f"[done] {job['id']} worker={response['pid']} cache={response.get('cache', {})}")
                return 'completed'
            except Exception as error:
                worker.close(force=True)
                state.update(status='interrupted' if self.stop.is_set() else 'failed',
                             finished=legacy.utc_now(), error=str(error))
                atomic_json(state_path(self.plan, job), state)
                self.report(f"[{state['status']}] {job['id']}: {error}")
                if self.args.fail_fast:
                    self.stop.set()
                return state['status']


def run_jobs(plan, jobs, args, gpus):
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    runner = PersistentRunner(plan, args)
    interrupted = [signal.SIGINT]
    previous_term, previous_int = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)

    def terminate(signum, frame):
        interrupted[0] = signum
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
        runner.report('Stopping new jobs and interrupting persistent writers...')
        runner.interrupt()
        return 128 + interrupted[0]
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
    parser = parallel.parser()
    parser.description = __doc__
    args = parser.parse_args(argv)
    if min(args.workers_per_gpu, args.pid_threads, args.root_chunk_events) < 1 or args.pipeline_depth < 0:
        parser.error('Workers/threads/chunk size must be positive; pipeline depth must be nonnegative')
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error('--max-jobs must be positive')
    plan = load_plan(args.plan)
    jobs = parallel.selected_jobs(plan, args, parser)
    if args.status:
        parallel.show_status(plan, jobs)
        return 0
    if not args.server or not args.gpus:
        parser.error('--server and --gpus are required for execution/dry-run')
    gpus = [gpu.strip() for gpu in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or any(
            not re.fullmatch(r'(?:[0-9]+|GPU-[a-zA-Z0-9-]+|MIG-[a-zA-Z0-9/-]+|cpu)', gpu)
            for gpu in gpus) or ('cpu' in gpus and len(gpus) != 1):
        parser.error('Use unique GPU numbers/UUIDs, or the single value cpu')
    verify_implementation(plan)
    print(f"[persistent] GPUs={','.join(gpus)} workers-per-gpu={args.workers_per_gpu} "
          f"max-workers={len(gpus) * args.workers_per_gpu} jobs={len(jobs)}", flush=True)
    if args.dry_run:
        for index, job in enumerate(jobs):
            gpu = gpus[index % len(gpus)]
            prefix = '' if gpu == 'cpu' else 'CUDA_VISIBLE_DEVICES=' + shlex.quote(gpu) + ' '
            print(f"# persistent job {job['id']} {job['model']} epoch={job['epoch']}")
            print(prefix + shlex.join(legacy.command_for(plan, job, args, gpu)))
        return 0
    return run_jobs(plan, jobs, args, gpus)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('error: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
