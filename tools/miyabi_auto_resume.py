"""Continue at a predicted epoch boundary, with planned timeout as a fallback.

No login-node daemon, GPU tests, or submission happens by importing this module.
PBS dependency ensures a successor starts only after this allocation has ended.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / 'shell/miyabi_train_energy_auto.sh'
TRAIN_SCRIPT = PROJECT / 'shell/miyabi_train_energy_fast.sh'
# Export only application settings; never propagate old PBS/MPI process state.
SETTINGS = (
    'RUN_DIR', 'EXPECTED_NODES', 'BATCH_SIZE', 'EPOCHS', 'LEARNING_RATE',
    'WEIGHT_DECAY', 'LR_POLICY', 'CLIP_VALUE', 'CLIP_MODE', 'TRAIN_SEED',
    'TRAIN_DIR', 'VALID_DIR', 'STREAM_FILES_PER_CHUNK', 'STREAM_SHUFFLE_BUFFER',
    'NUM_WORKERS', 'AMP', 'AMP_DTYPE', 'LR_WARMUP_EPOCHS', 'EPOCHS_NOBETA',
    'EPOCHS_NOLE', 'RANK_LOG_INTERVAL', 'OMP_NUM_THREADS', 'MASTER_PORT',
    'AUTO_MAX_JOBS', 'AUTO_JOB_INDEX',
    'CLUSTER_ENERGY_POOLING', 'CLUSTER_ENERGY_SOURCE',
)


def seconds(value):
    h, m, s = map(int, value.split(':'))
    return h * 3600 + m * 60 + s


def job_resources(text):
    fields = dict(re.findall(r'^\s*(\S+) = ([^\n]+)', text, re.M))
    walltime = fields['Resource_List.walltime'].strip()
    used = fields.get('resources_used.walltime', '00:00:00').strip()
    return walltime, seconds(walltime) - seconds(used), int(fields['Resource_List.nodect'])


def saved_epoch(run_dir):
    latest = run_dir / 'checkpoints/last.pth.tar'
    if not latest.exists():
        if os.path.lexists(latest):
            raise RuntimeError('Broken checkpoint symlink: ' + str(latest))
        return None
    name = latest.resolve(strict=True).name
    if name == 'ckpt_initial.pth.tar':
        return -1
    match = re.fullmatch(r'ckpt_(\d+)_1\.pth\.tar', name)
    if match is None:
        raise RuntimeError('Expected last.pth.tar from TrainingCheckpoint: ' + str(latest))
    return int(match[1])


def log_event(run_dir, **event):
    event.update(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    line = json.dumps(event, ensure_ascii=False)
    print('AUTO_RESUME ' + line, flush=True)
    with (run_dir / 'auto_jobs.jsonl').open('a') as stream:
        stream.write(line + '\n')


def successor_command(env, job_id, walltime):
    variables = dict((key, env[key]) for key in SETTINGS if key in env)
    variables['AUTO_JOB_INDEX'] = str(int(env['AUTO_JOB_INDEX']) + 1)
    for key, value in variables.items():
        if any(c in value for c in ',\n\r'):
            raise ValueError(f'{key} cannot contain commas or newlines for qsub -v')
    return ['qsub', '-q', 'regular-g', '-W', 'group_list=gw25',
            '-l', f'select={env["EXPECTED_NODES"]}:mpiprocs=1',
            '-l', f'walltime={walltime}', '-W', f'depend=afterany:{job_id}',
            '-v', ','.join(f'{k}={v}' for k, v in variables.items()), str(SCRIPT)]


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def supervise(command, env, budget, stop_file):
    """Distinguish OUR timeout from failures, even application exit code 124."""
    deadline = time.monotonic() + budget
    process = subprocess.Popen(command, cwd=PROJECT, env=env, start_new_session=True)
    try:
        while True:
            if stop_file.exists():
                return 'manual_stop', None
            result = process.poll()
            if result is not None:
                return ('complete' if result == 0 else 'error'), result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 'time_limit', None
            try:
                process.wait(timeout=min(30, remaining))
            except subprocess.TimeoutExpired:
                pass
    finally:
        stop_process(process)


def run(env):
    job_id = env['PBS_JOBID']
    if not env.get('PBS_NODEFILE'):
        raise RuntimeError('Submit with qsub; do not run training on a login node.')
    env.setdefault('EXPECTED_NODES', '4')
    env.setdefault('BATCH_SIZE', '32')
    env.setdefault('RUN_DIR', f'/work/gw25/w25002/checkpoint/energy_auto_{job_id}')
    env.setdefault('AUTO_MAX_JOBS', '25')
    env.setdefault('AUTO_JOB_INDEX', '1')
    if int(env.get('BENCHMARK_BATCHES', '0')) != 0:
        raise ValueError('Automatic continuation is for training, not benchmarks.')
    env['BENCHMARK_BATCHES'] = '0'
    run_dir = Path(env['RUN_DIR'])
    if not run_dir.is_absolute():
        raise ValueError('RUN_DIR must be an absolute path.')
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep this file descriptor for the entire job; fail instead of racing another run.
    with (run_dir / '.auto_resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run_dir / 'STOP_AUTO').exists():
            log_event(run_dir, job=job_id, event='manual_stop')
            return 0
        index, maximum = int(env['AUTO_JOB_INDEX']), int(env['AUTO_MAX_JOBS'])
        if not 1 <= index <= maximum:
            raise ValueError('AUTO_JOB_INDEX must be between 1 and AUTO_MAX_JOBS.')
        # Miyabi's supported qstat wrapper emits PBS-style attributes.
        details = subprocess.run(['qstat', '-f', job_id], check=True, text=True,
                                 capture_output=True, timeout=30).stdout
        walltime, remaining, nodes = job_resources(details)
        if nodes != int(env['EXPECTED_NODES']):
            raise ValueError('PBS node count differs from EXPECTED_NODES.')
        budget = remaining - 600  # Leave time for MPI cleanup and qsub.
        if budget <= 0:
            raise ValueError('Automatic continuation needs more than 10 minutes remaining walltime.')
        before = saved_epoch(run_dir)
        latest = run_dir / 'checkpoints/last.pth.tar'
        if index > 1 and before is None:
            raise RuntimeError('Continuation checkpoint is missing; refusing to start from scratch.')
        requested = env.get('RESUME_CHECKPOINT')
        if requested and Path(requested).resolve() != latest.resolve():
            raise ValueError('Automatic continuation resumes RUN_DIR/checkpoints/last.pth.tar only.')
        if before is not None:
            env['RESUME_CHECKPOINT'] = str(latest)
        # Validate successor arguments before starting a potentially long run.
        command = successor_command(env, job_id, walltime)
        # A unique marker prevents an old successful stop from hiding a new
        # failure. These per-allocation fields are not exported to successors.
        boundary_file = run_dir / f'epoch_boundary_{job_id}.json'
        boundary_file.unlink(missing_ok=True)
        env['AUTO_EPOCH_STOP_FILE'] = str(boundary_file)
        env['AUTO_TRAIN_DEADLINE'] = str(time.time() + budget)
        log_event(run_dir, job=job_id, event='start', index=index,
                  training_seconds=budget, completed_epoch=before)
        outcome, code = supervise(['bash', str(TRAIN_SCRIPT)], env, budget, run_dir / 'STOP_AUTO')
        if outcome == 'complete' and boundary_file.exists():
            boundary = json.loads(boundary_file.read_text())
            completed = saved_epoch(run_dir)
            if completed is None or boundary['next_epoch'] != completed + 1:
                raise RuntimeError('Epoch boundary signal does not match the latest checkpoint.')
            outcome = 'epoch_boundary'
        log_event(run_dir, job=job_id, event=outcome, returncode=code)
        if outcome not in ('time_limit', 'epoch_boundary') or (run_dir / 'STOP_AUTO').exists():
            return 1 if outcome == 'error' else 0
        after = saved_epoch(run_dir)
        if after is None or after <= (-1 if before is None else before):
            raise RuntimeError('No completed epoch advanced in this job; stopping automatic retries.')
        if after + 1 >= int(env.get('EPOCHS', '500')):
            return 0
        if index >= maximum:
            log_event(run_dir, job=job_id, event='job_cap_reached', maximum=maximum)
            return 0
        # Do not blindly retry an ambiguous qsub result: that could submit twice.
        result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=60)
        next_job = result.stdout.strip()
        (run_dir / 'next_job_id').write_text(next_job + '\n')
        log_event(run_dir, job=job_id, event='submitted', successor=next_job, completed_epoch=after)
        return 0


def interrupted(signum, frame):
    raise InterruptedError(f'Received signal {signum}; automatic continuation cancelled.')


def main():
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        return run(dict(os.environ))
    except Exception as exc:
        print(f'AUTO_RESUME stopped: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
