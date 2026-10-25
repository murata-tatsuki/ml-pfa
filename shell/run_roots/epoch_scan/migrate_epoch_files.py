#!/usr/bin/env python3
"""Convert a completed per-H5 scan, preserving and validating its readouts.

No inference is run. Originals remain until every output is validated. A
journal under _scan allows interrupted conversions and cleanup to resume.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import struct
import sys
import tempfile

sys.dont_write_bytecode = True
from scan_common import (atomic_json, completed_state, file_identity, job_digest,
                         load_plan, sha256_file)
from scan_layout import EPOCH_LAYOUT, epoch_plan, summary_directory
import epoch_jobs


def now():
    return datetime.now(timezone.utc).isoformat()


def root_identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular file: ' + str(path))
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                device=stat.st_dev, inode=stat.st_ino)


def lease(stack, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+')
    stat = os.fstat(stream.fileno())
    key = stat.st_dev, stat.st_ino
    held = getattr(stack, '_held_inodes', set())
    if key in held:
        stream.close()
        return
    stack.enter_context(stream)
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('Scan is active; stop it before migrating: ' + str(path)) from error
    held.add(key)
    stack._held_inodes = held


def group_key(job):
    return int(job['epoch']), job['method']


def legacy_readouts(plan, jobs):
    import collect_jer as core
    _, _, _, helper, _, _ = deps = core._dependencies(plan['repo'])
    sources = sorted((item['path'], job, item) for job in jobs for item in job['inputs'])
    result, digest = [], hashlib.sha256()
    for path, job, item in sources:
        root = Path(job['output_dir']) / (Path(path).stem + '.root')
        info = core._root_info(root, core._check_identity(item), job, plan, deps)
        selected = ~(info['reco']['cond_beta'] < 0.0)
        saved = helper._totals(info['reco']['event'], info['energies'], selected, info['events'])
        rows = [dict(event=int(event), truth_energy=float(truth),
                     thrust=info['angles'].get(int(event), float('nan')), reco_energy=saved[int(event)])
                for event, truth in zip(info['events'], info['truth'])]
        modes = {'saved': saved}
        if info['pid']:
            alternate, _, _ = helper._file_readouts(root, info['particle'], rows, 0.0)
            modes.update(alternate)
        ordered = sorted(rows, key=lambda row: row['event'])
        digest.update(str(Path(path).resolve()).encode() + b'\0')
        for row in ordered:
            angle = row['thrust']
            digest.update(struct.pack('<qdd', row['event'], row['truth_energy'],
                                      angle if math.isfinite(angle) else float('nan')))
        result.append((core._identity(item), ordered, modes))
    return result, digest.hexdigest()


def compare_readouts(reference, digest, data):
    import numpy as np
    import collect_jer as core
    ranges = {entry['input']['path']: entry for entry in data['source_ranges']}
    if len(ranges) != len(reference) or data['event_selection_sha256'] != digest:
        raise ValueError('Merged event selection differs from the original scan')
    total = 0
    for identity, rows, modes in reference:
        segment = ranges[identity['path']]
        if core._identity(segment['input']) != identity:
            raise ValueError('Merged source identity differs')
        offset, count = segment['event_offset'], segment['event_count']
        actual = sorted((row for row in data['rows'] if offset <= row['event'] < offset + count),
                        key=lambda row: row['event'])
        if set(modes) != set(data['modes']) or len(actual) != len(rows):
            raise ValueError('Merged readout/event coverage differs')
        if [r['event'] for r in rows] != [r['event'] - offset for r in actual]:
            raise ValueError('Merged local event IDs differ')
        for key in ('truth_energy', 'thrust'):
            if not np.array_equal(np.asarray([r[key] for r in rows], dtype=np.float64),
                                  np.asarray([r[key] for r in actual], dtype=np.float64), equal_nan=True):
                raise ValueError('Merged ' + key + ' differs')
        for mode in modes:
            before = np.asarray([modes[mode][r['event']] for r in rows], dtype=np.float64)
            after = np.asarray([data['modes'][mode][r['event']] for r in actual], dtype=np.float64)
            if not np.array_equal(before, after, equal_nan=True):
                raise ValueError('Merged energy readout differs: ' + mode)
        total += len(rows)
    if total != len(data['rows']):
        raise ValueError('Unmatched events in the merged ROOT')


def merge_one(old, new, job, old_jobs):
    sys.path.insert(0, new['repo'])
    from epoch_root import merge_roots
    from epoch_collect import read_epoch_data, collect_epoch_group
    epoch_jobs.install()
    reference, digest = legacy_readouts(old, old_jobs)
    paths = {item['path']: Path(part['output_dir']) / (Path(item['path']).stem + '.root')
             for part in old_jobs for item in part['inputs']}
    merge_roots([paths[item['path']] for item in job['inputs']],
                Path(job['output_file']), job['inputs'])
    data = read_epoch_data(new, job)
    compare_readouts(reference, digest, data)
    metrics = collect_epoch_group(new, [job], output_directory=summary_directory(new, job), cached_data=data)
    # CSV preserves these spellings; durable JSON records must stay strict JSON.
    metrics = [{key: str(value) if isinstance(value, float) and not math.isfinite(value) else value
                for key, value in row.items()} for row in metrics]
    return dict(job_id=job['id'], output=job['output_file'], identity=root_identity(job['output_file']),
                events=len(data['rows']), event_selection_sha256=digest,
                readouts=sorted(data['modes']), metrics=metrics)


def remove_empty(root):
    root = Path(root)
    if root.exists():
        for path in sorted((p for p in root.rglob('*') if p.is_dir()),
                           key=lambda p: len(p.parts), reverse=True):
            path.rmdir()
        root.rmdir()


def write_metrics(result, records):
    rows = [row for record in records for row in record['metrics']]
    fields = list(dict.fromkeys(name for row in rows for name in row))
    fd, temporary = tempfile.mkstemp(prefix='.metrics.', suffix='.tmp', dir=result)
    try:
        with os.fdopen(fd, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, result / 'metrics.csv')
    finally:
        Path(temporary).unlink(missing_ok=True)


def migrate(result, workers=2, dry_run=False):
    result = Path(result).expanduser().resolve()
    management = result / '_scan'
    journal_path = management / 'migration.json'
    target_plan_path = management / 'plan.json'
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else None
    if journal and journal.get('schema') != 'epoch-file-migration-1':
        raise ValueError('Unknown migration journal')
    epoch_jobs.install()
    if journal and journal.get('phase') == 'complete':
        plan = load_plan(target_plan_path)
        epoch_jobs.verify_implementation(plan)
        if not all(completed_state(plan, job) for job in plan['jobs']):
            raise ValueError('Migrated outputs changed since completion')
        print('[complete] This scan is already migrated', flush=True)
        return journal
    source_plan_path = result / 'plan.json'
    old = journal['source_plan'] if journal else load_plan(source_plan_path)
    if old.get('layout') == EPOCH_LAYOUT or Path(old['scan_dir']) != result:
        raise ValueError('Expected a legacy plan at SCAN_DIR/plan.json')
    epoch_jobs.verify_implementation(old)
    if not journal and target_plan_path.exists():
        raise ValueError('A different new-format plan already exists')
    lists = journal['filelists'] if journal else {
        job['input_list']: Path(job['input_list']).read_text() for job in old['jobs']}
    new, rewritten_lists = epoch_plan(old, result, lists)
    groups = {}
    for job in old['jobs']:
        groups.setdefault(group_key(job), []).append(job)
    source_roots = {str(Path(job['output_dir']) / (Path(item['path']).stem + '.root'))
                    for job in old['jobs'] for item in job['inputs']}
    if len(source_roots) != sum(len(j['inputs']) for j in old['jobs']):
        raise ValueError('Duplicate legacy ROOT path')
    published = target_plan_path.exists()
    if not published:
        if not all(completed_state(old, job) for job in old['jobs']):
            raise ValueError('Only fully completed scans can be migrated; finish the scan first')
        if {str(p) for p in (result / 'roots').rglob('*.root')} != source_roots:
            raise ValueError('Legacy ROOT inventory differs from the plan')
    if dry_run:
        print(json.dumps(dict(scan=str(result), old_roots=len(source_roots),
                              new_roots=len(new['jobs']), inference=False), indent=2))
        return
    if any(p.is_symlink() for p in result.rglob('*')):
        raise ValueError('Refusing symlinks inside the scan')
    epoch_jobs.freeze_execution(new)
    if journal and journal.get('execution_implementation') != new['execution_implementation']:
        raise ValueError('Epoch writer changed during migration; restore the recorded version before resuming')
    if not journal:
        if any((result / name).exists() for name in ('metrics.csv', 'summaries')):
            raise ValueError('Existing analysis products need an explicit migration strategy')
        journal = dict(schema='epoch-file-migration-1', phase='merging', started=now(),
                       source_plan_sha256=sha256_file(source_plan_path), source_plan=old, filelists=lists,
                       execution_implementation=new['execution_implementation'],
                       roots={path: root_identity(path) for path in sorted(source_roots)})
    # Refuse unexpected data before publication/retirement, not during rmdir.
    expected_state = {result / 'state' / (j['id'] + '.json') for j in old['jobs']}
    expected_locks = {result / 'state/locks' / (j['id'] + '.lock') for j in old['jobs']}
    expected_locks.update((result / 'state/gpu_locks').glob('*.lock'))
    for directory, allowed in (('roots', {Path(p) for p in source_roots}),
                               ('filelists', {Path(p) for p in lists}),
                               ('state', expected_state | expected_locks)):
        unknown = {p for p in (result / directory).rglob('*') if p.is_file()} - allowed
        if unknown:
            raise ValueError('Unexpected legacy files: ' + str(sorted(map(str, unknown))))
    work = management / 'migration'
    with ExitStack() as stack:
        lease(stack, result.with_name('.' + result.name + '.prepare.lock'))
        lease(stack, management.with_name('._scan.prepare.lock'))
        if not published:
            if sha256_file(source_plan_path) != journal['source_plan_sha256']:
                raise ValueError('Legacy plan changed during migration')
            if not all(completed_state(old, job) for job in old['jobs']):
                raise ValueError('Legacy completion markers changed')
        for path in (result / 'state/gpu_locks').glob('*.lock'):
            lease(stack, path)
        for job in old['jobs']:
            lock = result / 'state/locks' / (job['id'] + '.lock')
            if not published or lock.exists():
                lease(stack, lock)
        for directory in ('state/locks', 'state/gpu_locks'):
            for old_lock in (result / directory).glob('*.lock'):
                target = management / old_lock.relative_to(result)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    os.link(old_lock, target)
                elif old_lock.stat().st_ino != target.stat().st_ino:
                    raise ValueError('Migration lock namespaces differ')
        for job in new['jobs']:
            lease(stack, management / 'state/locks' / (job['id'] + '.lock'))
        if not published:
            for path, identity in journal['roots'].items():
                if root_identity(path) != identity:
                    raise ValueError('Legacy ROOT changed: ' + path)
        atomic_json(journal_path, journal)
        work.mkdir(exist_ok=True)
        for name, contents in rewritten_lists.items():
            file = Path(name)
            file.parent.mkdir(parents=True, exist_ok=True)
            if file.exists() and file.read_text() != contents:
                raise ValueError('Migration input list differs')
            if not file.exists():
                file.write_text(contents)
                file.chmod(0o444)
        if not published:
            done, pending = {}, []
            for job in new['jobs']:
                record_path = work / (job['id'] + '.json')
                record = json.loads(record_path.read_text()) if record_path.exists() else None
                if record and Path(job['output_file']).is_file() and record['identity'] == root_identity(job['output_file']):
                    done[job['id']] = record
                else:
                    pending.append(job)
            print(f'[migration] {len(done)}/{len(new["jobs"])} ready; workers={workers}', flush=True)
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = {pool.submit(merge_one, old, new, j, groups[group_key(j)]): j for j in pending}
                try:
                    for future in as_completed(futures):
                        job, record = futures[future], future.result()
                        atomic_json(work / (job['id'] + '.json'), record)
                        done[job['id']] = record
                        print(f'[verified {len(done)}/{len(new["jobs"])}] epoch={job["epoch"]} '
                              f'{job["method"]} events={record["events"]}', flush=True)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
            records = [done[job['id']] for job in new['jobs']]
            for job, record in zip(new['jobs'], records):
                if root_identity(job['output_file']) != record['identity']:
                    raise ValueError('New ROOT changed after validation')
                source_jobs = groups[group_key(job)]
                first = json.loads((result / 'state' / (source_jobs[0]['id'] + '.json')).read_text())
                state = copy.deepcopy(first)
                state.update(job_id=job['id'], job_digest=job_digest(job), status='completed',
                             outputs=[dict(path=job['output_file'], identity=file_identity(job['output_file']))],
                             migrated=now(), source_jobs=[j['id'] for j in source_jobs])
                if state.get('log'):
                    state['log'] = str(management / Path(state['log']).relative_to(result))
                atomic_json(management / 'state' / (job['id'] + '.json'), state)
            new['migration'] = dict(schema=journal['schema'], source_plan_sha256=journal['source_plan_sha256'],
                                    source_roots=len(source_roots), output_roots=len(records),
                                    validated_readouts=True, completed=now())
            write_metrics(result, records)
            atomic_json(target_plan_path, new)
            target_plan_path.chmod(0o444)
            journal.update(phase='published', outputs={r['output']: r['identity'] for r in records})
            atomic_json(journal_path, journal)
        else:
            new = load_plan(target_plan_path)
            if new.get('migration', {}).get('source_plan_sha256') != journal['source_plan_sha256']:
                raise ValueError('Published plan does not belong to this migration')
            if 'outputs' not in journal:
                journal['outputs'] = {j['output_file']: json.loads((work / (j['id'] + '.json')).read_text())['identity']
                                      for j in new['jobs']}
            journal['phase'] = 'published'
            atomic_json(journal_path, journal)
        epoch_jobs.verify_implementation(new)
        if not all(completed_state(new, job) for job in new['jobs']):
            raise ValueError('Published completion records do not validate')
        for path, identity in journal['outputs'].items():
            if root_identity(path) != identity:
                raise ValueError('Published ROOT changed before source retirement')
        if (result / 'logs').exists():
            if (management / 'logs').exists():
                # A legacy status/restart may have recreated an empty log tree.
                if any(p.is_file() for p in (result / 'logs').rglob('*')):
                    raise ValueError('Both original and new log directories contain files')
                remove_empty(result / 'logs')
            else:
                os.rename(result / 'logs', management / 'logs')
        for path, identity in journal['roots'].items():
            original = Path(path)
            if original.exists():
                if root_identity(original) != identity:
                    raise ValueError('Legacy ROOT changed before retirement')
                original.unlink()
        if source_plan_path.exists():
            if sha256_file(source_plan_path) != journal['source_plan_sha256']:
                raise ValueError('Legacy plan changed before retirement')
            source_plan_path.unlink()
        for path in lists:
            Path(path).unlink(missing_ok=True)
        for job in old['jobs']:
            (result / 'state' / (job['id'] + '.json')).unlink(missing_ok=True)
        for directory in ('state/locks', 'state/gpu_locks'):
            for path in (result / directory).glob('*.lock'):
                path.unlink()
    # NFS removes .nfs aliases only after all open lock descriptors close.
    for name in ('roots', 'filelists', 'state'):
        remove_empty(result / name)
    for job in new['jobs']:
        (work / (job['id'] + '.json')).unlink(missing_ok=True)
    work.rmdir()
    summary = dict(schema=journal['schema'], phase='complete', started=journal['started'], completed=now(),
                   source_plan_sha256=journal['source_plan_sha256'], original_roots=len(source_roots),
                   output_roots=len(new['jobs']), original_bytes=sum(i['size'] for i in journal['roots'].values()),
                   output_bytes=sum(Path(j['output_file']).stat().st_size for j in new['jobs']),
                   validated_readouts=True)
    atomic_json(journal_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-dir', required=True, type=Path, help='Completed legacy energy directory')
    parser.add_argument('--workers', default=2, type=int, help='CPU merge processes; no GPU inference')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    migrate(args.scan_dir, args.workers, args.dry_run)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('migration: ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
