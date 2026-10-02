"""Predict whether another full epoch fits before a job's training deadline."""
import json
import math
import os
from pathlib import Path
import time


class EpochTimeBudget:
    def __init__(self, deadline=0, stop_file=None):
        self.deadline = deadline
        self.stop_file = stop_file
        self.durations = []

    def record(self, elapsed):
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('Epoch duration must be finite and positive.')
        self.durations = (self.durations + [elapsed])[-5:]

    def should_stop(self, next_epoch, now=None):
        if not self.deadline:
            return False
        remaining = self.deadline - (time.time() if now is None else now)
        # Include train, validation, checkpoint I/O and DDP waits. The slowest
        # recent epoch plus 20% margin tolerates ordinary I/O/timing variation.
        estimate = max(self.durations, default=0) * 1.2
        stop = remaining <= estimate
        print(f'EPOCH_BUDGET next_epoch={next_epoch} remaining_seconds={remaining:.3f} '
              f'predicted_seconds={estimate:.3f} decision={"stop" if stop else "start"}',
              flush=True)
        if stop:
            path = Path(self.stop_file)
            payload = dict(next_epoch=next_epoch, remaining_seconds=remaining,
                           predicted_seconds=estimate, recent_epoch_seconds=self.durations)
            with open(str(path) + '.tmp', 'w') as stream:
                json.dump(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(str(path) + '.tmp', path)
        return stop
