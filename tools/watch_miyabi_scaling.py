"""Report a fixed, budgeted set of PBS jobs. Never submit or retry jobs."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time

from watch_miyabi_benchmark import job_status, summarize


def completed_cases(logs, gpus):
    """Require matching config and completed train+validation on every rank."""
    cases = {}
    for rank, content in enumerate(logs):
        parts = re.split(r'BENCH_CONFIG chunk=(\d+) buffer=(\d+) workers=(\d+)\n', content)
        for i in range(1, len(parts), 4):
            body = parts[i + 3]
            metadata = re.search(r'BENCH_RUN gpus=(\d+) batch_per_gpu=(\d+) '
                                 r'global_batch=(\d+) optimizer_lr=(\S+) precision=(\S+)', body)
            if not metadata:
                continue
            nodes, batch, global_batch, lr, precision = metadata.groups()
            if int(nodes) != gpus or int(global_batch) != gpus * int(batch):
                continue
            key = (*map(int, parts[i:i + 3]), int(batch), float(lr), precision)
            cases.setdefault(key, [''] * gpus)[rank] = body
    rows = []
    for key, bodies in cases.items():
        phases = summarize(bodies)
        if not all(len(phases.get(p, [])) == gpus for p in ('train', 'validation')):
            continue
        if any(re.search(r'(?:loss=|Returning\s+)(?:nan|[+-]?inf)\b', b, re.I) for b in bodies):
            continue
        seconds = sum(max(r['epoch_seconds'] for r in phases[p]) for p in ('train', 'validation'))
        if seconds <= 0 or not math.isfinite(seconds):
            continue
        train_seconds = max(r['seconds_per_batch'] for r in phases['train'])
        peak = max([float(v) for b in bodies for v in
                    re.findall(r'gpu_peak_reserved_mib=([\d.]+)', b)] or [0.])
        allocated = max([float(v) for b in bodies for v in
                         re.findall(r'gpu_peak_allocated_mib=([\d.]+)', b)] or [0.])
        rows.append(dict(gpus=gpus, chunk=key[0], buffer=key[1], workers=key[2],
                         batch=key[3], lr=key[4], precision=key[5], global_batch=gpus*key[3],
                         events_per_second=gpus*key[3]/train_seconds,
                         epoch_hours=seconds/3600, gpu_hours=seconds*gpus/3600,
                         gpu_reserved_gib=peak/1024, gpu_allocated_gib=allocated/1024))
    return rows


def write_report(manifest, repo):
    rows, states, status_lines = [], [], []
    used_gpu_seconds = manifest.get('failed_gpu_seconds', 0)
    for job in manifest['jobs']:
        state, status = job_status(job['id'])
        states.append(state)
        used = re.search(r'^\s*resources_used.walltime = (\d+):(\d+):(\d+)', status, re.M)
        if used:
            hours, minutes, seconds = map(int, used.groups())
            used_gpu_seconds += job['gpus']*(3600*hours+60*minutes+seconds)
        root = Path(job['run_dir'])
        logs = [(root/'ranks'/f'rank{r}.log').read_text(errors='replace')
                if (root/'ranks'/f'rank{r}.log').exists() else '' for r in range(job['gpus'])]
        found = completed_cases(logs, job['gpus'])
        rows.extend(dict(row, job_id=job['id']) for row in found)
        details = []
        for field in ('estimated.start_time', 'resources_used.walltime', 'Exit_status'):
            match = re.search(rf'^\s*{re.escape(field)} = (.+)$', status, re.M)
            if match:
                details.append(f'{field}: {match[1]}')
        status_lines.append(f"- {job['id']}: {job['gpus']} GPU, state={state}, "
                            f"completed cases={len(found)}; {'; '.join(details)}")
        for rank, content in enumerate(logs):
            if 'torch.cuda.OutOfMemoryError:' in content:
                batches = re.findall(r'BENCH_RUN gpus=\d+ batch_per_gpu=(\d+)', content)
                status_lines.append(f"  rank {rank}: batch/GPU={batches[-1] if batches else '?'}でGPUメモリ不足。後続条件は未測定。")
        if state == 'F' and not found:
            status_lines.append(f"  No complete timing result. Inspect `{root / 'ranks'}` and `{job['pbs_output']}`.")
    lines = ['# Miyabi GPU数・batch size速度比較', '',
             f"更新: {datetime.now().astimezone().isoformat(timespec='seconds')}", '',
             manifest.get('objective', '目的: 精度を維持してGPU時間を削減。今回は速度測定のみで、精度は未評価。'),
             f"投入枠の上限: {manifest['maximum_gpu_minutes']} GPU分 / 60 GPU分。自動再投入なし。", '',
             f"PBSが報告した消費（終了済み失敗分も含む）: {used_gpu_seconds/60:.2f} GPU分。", '',
             *status_lines, '',
             '各条件・各phaseはwarm-up 2 batch + 測定10 batch。LR=4e-4を固定。precisionは各行に記載。',
             'Batch/GPUは1 GPUあたりのイベント数。Global batch = GPU数 × Batch/GPU。', '',
             '| Job | GPU | Batch/GPU | Global batch | Precision | Chunk/buffer/workers | Train events/s | GPU実使用ピーク GiB | 推定epoch時間 h | 推定GPU時間/epoch h |',
             '|---|---:|---:|---:|---|---|---:|---:|---:|---:|']
    for r in sorted(rows, key=lambda r:(r['gpus'],r['batch'],r['chunk'],r['workers'],r['buffer'])):
        lines.append(f"| {r['job_id']} | {r['gpus']} | {r['batch']} | {r['global_batch']} | "
                     f"{r['precision']} | {r['chunk']}/{r['buffer']}/{r['workers']} | {r['events_per_second']:.2f} | "
                     f"{r['gpu_allocated_gib']:.2f} | {r['epoch_hours']:.3f} | {r['gpu_hours']:.3f} |")
    if not rows:
        lines += ['', '完了した測定はまだありません。最適なGPU数・batch sizeは未確定です。']
    else:
        # A hardware-efficiency comparison must keep the effective batch fixed.
        groups = {}
        for r in rows:
            k = (r['global_batch'], r['chunk'], r['buffer'], r['workers'], r['lr'], r['precision'])
            groups.setdefault(k, []).append(r)
        matched = [group for group in groups.values() if len({r['gpus'] for r in group}) >= 2]
        for group in matched:
            fastest = manifest.get('ranking_metric') == 'walltime'
            best = min(group, key=lambda r:r['epoch_hours'] if fastest else r['gpu_hours'])
            criterion = '実時間最短' if fastest else 'GPU時間最小'
            lines += ['', f"同一global batch={best['global_batch']}・precision={best['precision']}・loader条件での暫定{criterion}: "
                      f"{best['gpus']} GPU × batch {best['batch']} "
                      f"({best['epoch_hours']:.3f} 時間/epoch、{best['gpu_hours']:.3f} GPU時間/epoch)。精度維持の確認は未実施。"]
        if not matched:
            lines += ['', '同一global batch・loader条件で複数GPU数の測定が揃っていないため、GPU数の推奨は保留します。']
    lines += ['', '## 測定の限界', '',
              '- train 1,960,000 / validation 20,000イベントへの短区間からの外挿です。全epochを測っていません。',
              '- rankごとの進捗batch数から各phaseの最大推定時間を採用します。',
              '- 初回読み込み、後続chunk読み込み、checkpoint保存、キュー待ちは外挿に含みません。',
              '- 各条件のデータ分割・イベント数・ファイルキャッシュ状態は異なり、速度差にはその影響も含まれます。',
              '- 大きなbatchではイベントあたりの更新回数が減ります。GPU時間/epochが短くても、目標精度までの総時間が短いとは限りません。',
              '- 今回のbatch比較ではLRを固定し、速度のみ評価しています。loss値からLRやbatchの最適性を選びません。',
              '- GPUメモリはPyTorch allocatorの最大allocated値です。NCCLなどの外部割当は含みません。reserved値もJSONに記録します。',
              '- GPUメモリ不足・時間切れが起きた条件は未完了として扱い、自動再実行しません。', '',
              '## 後続の精度比較候補（今回実行しない）', '',
              '- 基準: global batch 128、実際のoptimizer LR 4e-4、AdamW weight decay 1e-4、元のloss設定。',
              '- LR: 1e-4 / 2e-4 / 4e-4、warm-up: 0 / 2 / 5 epochを段階的に比較。',
              '- 有望な条件でweight decay: 1e-5 / 1e-4 / 1e-3を比較。全組合せの一括探索はしない。',
              '- 同じ学習イベント数・固定validationでenergy lossとclustering lossを別々に評価。',
              '- beta/energy lossの有効化時期を揃え、LR scheduler・warm-upを短い試験用に都合よく変更しない。',
              '- 精度評価と許容差を決めてから本番設定を選ぶ。元のloss定義やqminなどは今回変更しない。']
    report_name = manifest.get('report_name', 'benchmark_miyabi_scaling')
    target = repo/f'{report_name}.md'
    temp = target.with_suffix('.tmp')
    temp.write_text('\n'.join(lines)+'\n')
    temp.replace(target)
    (repo/f'{report_name}_results.json').write_text(json.dumps(rows, indent=2)+'\n')
    return all(state == 'F' for state in states), states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    repo = Path(__file__).resolve().parents[1]
    deadline = time.monotonic()+24*3600
    while True:
        done, states = write_report(manifest, repo)
        print(datetime.now().isoformat(), states, flush=True)
        if done or args.once or time.monotonic() >= deadline:
            return
        time.sleep(45)


if __name__ == '__main__':
    main()
