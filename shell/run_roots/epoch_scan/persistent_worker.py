#!/usr/bin/env python3
"""Run existing save_root requests sequentially in one Python/CUDA process."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True


def flush_output():
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    # ROOT writes to the C/C++ file descriptors, independently of sys.stdout.
    ctypes.CDLL(None).fflush(None)


@contextmanager
def job_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flush_output()
    saved = [os.dup(1), os.dup(2)]
    try:
        with path.open('a', buffering=1) as output:
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            try:
                yield
            finally:
                flush_output()
    finally:
        for target, original in zip((1, 2), saved):
            os.dup2(original, target)
            os.close(original)


class Writer:
    def __init__(self, repo):
        sys.path.insert(0, str(repo))
        import save_root
        from root_export import cedric
        from persistent_model import ModelCache

        self.cli = save_root
        self.models = ModelCache(cedric.get_model)
        # Local to this dedicated process. The original CLI/files stay intact.
        cedric.get_model = self.models

    def run(self, argv, mode='legacy'):
        if mode == 'epoch':
            import epoch_writer
            result = epoch_writer.main(argv)
        elif mode == 'legacy':
            result = self.cli.main(argv)
        else:
            raise ValueError('Unknown writer mode: ' + str(mode))
        if result not in (None, 0):
            raise RuntimeError(f'save_root returned {result}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--reply-fd', required=True, type=int)
    args = parser.parse_args()
    writer = None
    with os.fdopen(args.reply_fd, 'w', buffering=1) as reply:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('shutdown'):
                return 0
            response = dict(ok=False, pid=os.getpid(), job_id=request['job_id'])
            with job_output(request['log']):
                try:
                    print(f'[persistent worker] pid={os.getpid()} job={request["job_id"]}', flush=True)
                    if writer is None:
                        writer = Writer(args.repo)
                    writer.run(request['argv'], request.get('writer_mode', 'legacy'))
                    response.update(ok=True, cache=writer.models.stats)
                    print('[persistent model] ' + json.dumps(writer.models.stats), flush=True)
                except BaseException as error:
                    traceback.print_exc()
                    response['error'] = f'{type(error).__name__}: {error}'
            reply.write(json.dumps(response, allow_nan=False) + '\n')
            reply.flush()
            if not response['ok']:
                # Never carry partially modified weights/ROOT state into another job.
                return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
