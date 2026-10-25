#!/usr/bin/env python3
"""Prepare/resume one manually selected model/energy/epoch block, then run it."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from plan_scan import (absolute_path, make_plan, parse_epochs, read_config,
                       require_name, write_plan)
from scan_common import load_plan
from scan_layout import EPOCH_LAYOUT, epoch_plan
from epoch_jobs import freeze_execution, verify_implementation

FILES_PER_FLAVOR = {40: 3, 91: 3, 200: 8, 350: 8, 500: 17}
DEFAULT_OUTPUT = '/data/suehara/mldata/pfa/murata/output/jer_epoch_scan'


def digest_request(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def effective_request(args):
    config = read_config(absolute_path(str(args.config)))
    selected = [model for model in config['models'] if model['name'] == args.model]
    if not selected:
        names = ', '.join(model['name'] for model in config['models'])
        raise ValueError(f'Unknown model {args.model!r}; choose {names}')
    if args.energy not in FILES_PER_FLAVOR:
        raise ValueError('energy must be one of 40, 91, 200, 350, 500 GeV')
    epochs = parse_epochs(args.epochs)
    count = FILES_PER_FLAVOR[args.energy] if args.files_per_flavor is None else args.files_per_flavor
    config = copy.deepcopy(config)
    config['energies'] = [args.energy]
    config['models'] = selected
    request = dict(schema='epoch-setting-1', config=config,
                   epochs='all' if epochs is None else sorted(epochs),
                   files_per_flavor=count, files_per_job=args.files_per_job)
    return config, epochs, count, request


def output_path(args, config, epochs, count, request):
    if args.scan_dir:
        return absolute_path(args.scan_dir)
    base = absolute_path(args.output_base)
    model = config['models'][0]
    if args.tag:
        require_name(args.tag, 'tag')
        base /= args.tag
    return base / (model['name'] + '_' + model['run_id']) / f'{args.energy}GeV'


def report_legacy_results(args, config, request, scan_dir):
    """Make an existing legacy scan visible before starting a separate output."""
    if args.scan_dir:
        return
    base = absolute_path(args.output_base)
    model = config['models'][0]
    relative = (f"*/{model['name']}_{model['run_id']}/{args.energy}GeV/"
                'epochs_*/plan.json')
    found = set()
    for root in (base / 'manual', base):
        for path in root.glob(relative):
            try:
                saved = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if saved.get('manual_request') == request:
                found.add(str(path.parent))
    for old_scan in sorted(found):
        print(f'[legacy] Existing results remain at {old_scan}\n'
              f'         New output: {scan_dir}\n'
              f'         To continue the existing scan, use --scan-dir {old_scan}',
              file=sys.stderr, flush=True)


def verify_request(plan, request):
    saved = plan.get('manual_request')
    if not isinstance(saved, dict) or plan.get('manual_request_sha256') != digest_request(saved):
        raise ValueError('Existing plan has no valid frozen setting.sh request')
    requested = copy.deepcopy(request)
    comparable = copy.deepcopy(saved)
    if plan.get('layout') == EPOCH_LAYOUT:
        # A new job always contains all selected H5s; legacy shard size has no
        # effect on the selected events, writer settings or new output identity.
        requested.pop('files_per_job', None)
        comparable.pop('files_per_job', None)
    if comparable != requested:
        raise ValueError('Existing plan has different settings or was not made by setting.sh. '
                         'Choose another --tag/--scan-dir; existing results were not changed.')
    if any(job['server'] != 'local' for job in plan['jobs']):
        raise ValueError('Manual setting plan must assign every job to local')


@contextmanager
def plan_lock(scan_dir):
    """Serialize first-time preparation on shared /data without creating scan_dir."""
    scan_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = scan_dir.with_name('.' + scan_dir.name + '.prepare.lock')
    with lock_path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def prepare_plan(args, config, epochs, count, request, scan_dir):
    inspect_only = args.status or args.collect
    with plan_lock(scan_dir):
        if scan_dir.is_symlink():
            raise ValueError(f'Refusing a symlink scan directory: {scan_dir}')
        plan_path = scan_dir / '_scan' / 'plan.json'
        legacy_path = scan_dir / 'plan.json'
        if plan_path.is_file() and legacy_path.is_file():
            raise ValueError(f'Both new and legacy plans exist under {scan_dir}; finish migration first')
        if not plan_path.is_file() and legacy_path.is_file():
            plan_path = legacy_path
        if plan_path.is_file():
            plan = load_plan(plan_path)
            verify_request(plan, request)
            verify_implementation(plan)
            print(f'[resume] {plan_path}', flush=True)
            return plan_path, plan
        if inspect_only:
            raise ValueError(f'No existing plan at {plan_path}; run --plan-only or --dry-run first')
        if scan_dir.exists():
            raise ValueError(f'Scan directory exists without a complete plan: {scan_dir}')
        model = config['models'][0]
        if count:
            root = Path(model.get('input_root', config['input_root'])) / f'{args.energy}GeV'
            for flavor in config['flavors']:
                available = len(list(root.glob(flavor + '*.h5')))
                if available < count:
                    raise ValueError(f'{root}: {flavor} has {available} H5 files, '
                                     f'but files-per-flavor={count} was requested')
        report_legacy_results(args, config, request, scan_dir)
        management = scan_dir / '_scan'
        plan, filelists = make_plan(config, management, [{'name': 'local', 'weight': 1.0}],
                                   epochs, None, count, sys.maxsize)
        plan, filelists = epoch_plan(plan, scan_dir, filelists)
        plan['manual_request'] = request
        plan['manual_request_sha256'] = digest_request(request)
        freeze_execution(plan)
        try:
            write_plan(management, plan, filelists)
        except BaseException:
            # write_plan removes its own newly created metadata directory.
            # Remove only the empty result parent created by this invocation.
            try:
                scan_dir.rmdir()
            except OSError:
                pass
            raise
        print(f'[created] {plan_path}', flush=True)
        return plan_path, plan


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--energy', required=True, type=int)
    p.add_argument('--epochs', default='all', help='all or 0,5,10:21:2; range stop excluded')
    p.add_argument('--gpus', default='0', help='Local GPU numbers/UUIDs separated by commas')
    p.add_argument('--workers-per-gpu', type=int, default=1,
                   help='Concurrent save_root processes per GPU (setting.sh default: 10)')
    p.add_argument('--worker-mode', choices=('persistent', 'subprocess'), default='persistent',
                   help='Reuse writer processes across jobs/epochs (default), or start one process per job')
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('scan_config.json'))
    p.add_argument('--output-base', default=os.environ.get('SCAN_OUTPUT_BASE', DEFAULT_OUTPUT))
    p.add_argument('--scan-dir', help='Optional explicit new/reusable scan directory')
    p.add_argument('--tag', default=os.environ.get('SCAN_TAG') or None,
                   help='Optional named group under output-base; omitted by default')
    p.add_argument('--files-per-flavor', type=int, help='Override energy mapping; 0 means all H5s')
    p.add_argument('--files-per-job', type=int, default=20,
                   help='Legacy shard size; new scans use all selected H5s per epoch/method')
    p.add_argument('--pid-threads', type=int, default=2)
    p.add_argument('--root-chunk-events', type=int, default=10)
    p.add_argument('--pipeline-depth', type=int, default=40)
    p.add_argument('--max-jobs', type=int)
    actions = p.add_mutually_exclusive_group()
    actions.add_argument('--dry-run', action='store_true', help='Prepare/reuse plan, print commands; no inference')
    actions.add_argument('--plan-only', action='store_true', help='Prepare/reuse plan without inference')
    actions.add_argument('--status', action='store_true', help='Inspect existing plan without creating one')
    actions.add_argument('--collect', action='store_true', help='Collect JER from existing ROOT outputs')
    p.add_argument('--recheck', action='store_true')
    p.add_argument('--replace', action='store_true')
    p.add_argument('--fail-fast', action='store_true')
    p.add_argument('--allow-incomplete', action='store_true', help='Only with --collect; partial JER diagnostics')
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers_per_gpu < 1:
        parser.error('--workers-per-gpu must be positive')
    if args.files_per_flavor is not None and args.files_per_flavor < 0:
        parser.error('--files-per-flavor must be nonnegative')
    if min(args.files_per_job, args.pid_threads, args.root_chunk_events) < 1 or args.pipeline_depth < 0:
        parser.error('File/thread/chunk counts must be positive; pipeline depth must be nonnegative')
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error('--max-jobs must be positive')
    if args.allow_incomplete and not args.collect:
        parser.error('--allow-incomplete is only used with --collect')
    gpus = [gpu.strip() for gpu in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or any(not re.fullmatch(
            r'(?:[0-9]+|GPU-[a-zA-Z0-9-]+|MIG-[a-zA-Z0-9/-]+|cpu)', gpu) for gpu in gpus
            ) or ('cpu' in gpus and len(gpus) != 1):
        parser.error('Use unique GPU numbers/UUIDs, or cpu')
    config, epochs, count, request = effective_request(args)
    scan_dir = output_path(args, config, epochs, count, request)
    plan_path, plan = prepare_plan(args, config, epochs, count, request, scan_dir)
    unique_inputs = {item['path'] for job in plan['jobs'] for item in job['inputs']}
    frozen_epochs = sorted({job['epoch'] for job in plan['jobs']})
    print(f"model={args.model} energy={args.energy}GeV gpus={','.join(gpus)} "
          f"workers-per-gpu={args.workers_per_gpu} max-workers={len(gpus) * args.workers_per_gpu} "
          f"worker-mode={args.worker_mode} "
          f"files-per-flavor={count} H5={len(unique_inputs)} "
          f"epochs={len(frozen_epochs)} ({frozen_epochs[0]}..{frozen_epochs[-1]}) "
          f"jobs={len(plan['jobs'])}", flush=True)
    print(f'output={scan_dir}', flush=True)
    if args.plan_only:
        return 0
    package = Path(__file__).resolve().parent
    if args.collect:
        command = [sys.executable, '-B', str(package / 'collect_scan.py'), '--plan', str(plan_path)]
        if args.allow_incomplete:
            command.append('--allow-incomplete')
    else:
        runner = 'run_persistent.py' if args.worker_mode == 'persistent' else 'run_parallel.py'
        command = [sys.executable, '-B', str(package / runner), '--plan', str(plan_path),
                   '--server', 'local', '--gpus', ','.join(gpus),
                   '--workers-per-gpu', str(args.workers_per_gpu)]
        if args.status:
            command.append('--status')
        else:
            command += ['--pid-threads', str(args.pid_threads),
                        '--root-chunk-events', str(args.root_chunk_events),
                        '--pipeline-depth', str(args.pipeline_depth)]
            for flag in ('dry_run', 'recheck', 'replace', 'fail_fast'):
                if getattr(args, flag):
                    command.append('--' + flag.replace('_', '-'))
            if args.max_jobs is not None:
                command += ['--max-jobs', str(args.max_jobs)]
    os.execv(sys.executable, command)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('setting: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
