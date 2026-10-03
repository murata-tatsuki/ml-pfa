"""Ablate KNN changes 1, 3 and 5 on identical coordinates, without training.

Inputs are either --coordinates (a torch-saved dict with CPU ptr and a list of
coordinate tensors) or --cached-inputs + --checkpoint (the saved audit batches).
--describe-only reports launch counts on CPU, NOT GPU timings or speedups.
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from knn_event_parallel import (prepare_plan, select_knn, select_knn_baseline,
                                knn_graph, load_extension, BLOCK_SIZES)


def load_cpu(path):
    # Audit/checkpoint files are user-owned trusted artifacts, not remote input.
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # PyTorch 1.12
        return torch.load(path, map_location='cpu')


def launch_counts(ptr):
    return {str(block): dict(rectangular_blocks=p[3]*(len(ptr)-1),
                            packed_blocks=p[1].size(1), hits=p[4])
            for block in BLOCK_SIZES for p in [prepare_plan(ptr, block_size=block)]}


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter()-start)*1000
    del value
    return elapsed


def measurements(calls, repeats):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    for _ in range(3):
        for call in calls.values():
            value = call()
            del value
    samples = {name: [] for name in calls}
    rng = random.Random(1004)
    for _ in range(repeats):
        order = list(calls)
        rng.shuffle(order)
        for name in order:
            samples[name].append(timed(calls[name]))
    return {name: dict(median_ms=statistics.median(values), samples_ms=values)
            for name, values in samples.items()}


def ablate(coordinates, ptr, repeats=12):
    if not coordinates:
        raise ValueError("At least one coordinate tensor is required")
    load_extension()
    device = coordinates[0].device
    plans = {block: prepare_plan(ptr, device, block) for block in BLOCK_SIZES}
    batch = torch.arange(len(ptr)-1, device=device).repeat_interleave((ptr[1:]-ptr[:-1]).to(device))
    # Sequential comparisons isolate one change at a time. Single-5 is also
    # measured against 1-only to expose interactions with block packing.
    modes = {'1_only': (1024, False, True),
             '1_3': (1024, True, True),
             '1_5': (1024, False, False),
             '1_3_5': (1024, True, False)}
    modes.update({'1_3_5_block'+str(b): (b, True, False) for b in BLOCK_SIZES if b != 1024})
    exact = []
    for layer, x in enumerate(coordinates):
        expected_i, expected_d = select_knn_baseline(x, 41, batch)
        expected_edges = knn_graph(x, 40, batch, baseline=True)
        bits = torch.int32 if x.dtype == torch.float32 else torch.int64
        for name, (block, packed, zero_init) in modes.items():
            options = dict(plan=plans[block], packed=packed, zero_init=zero_init)
            i, d = select_knn(x, 41, batch, **options)
            edges = knn_graph(x, 40, batch, **options)
            counts = dict(layer=layer, mode=name,
                index_mismatches=int((i != expected_i).sum()),
                distance_bit_mismatches=int((d.view(bits) != expected_d.view(bits)).sum()),
                edge_order_equal=torch.equal(edges, expected_edges))
            exact.append(counts)
            if counts['index_mismatches'] or counts['distance_bit_mismatches'] or not counts['edge_order_equal']:
                raise AssertionError('Exact check failed: ' + str(counts))

    def sweep(name, graph=False, include_preparation=False):
        if name == 'baseline':
            fn = (lambda x: knn_graph(x, 40, batch, baseline=True)) if graph else (
                 lambda x: select_knn_baseline(x, 41, batch))
        else:
            block, packed, zero_init = modes[name]
            plan = prepare_plan(ptr, device, block) if include_preparation else plans[block]
            fn = (lambda x: knn_graph(x, 40, batch, plan=plan, packed=packed, zero_init=zero_init)) if graph else (
                 lambda x: select_knn(x, 41, batch, plan=plan, packed=packed, zero_init=zero_init))
        for x in coordinates:
            value = fn(x)
            del value

    names = ['baseline'] + list(modes)
    search = measurements({n: lambda n=n: sweep(n) for n in names}, repeats)
    graph = measurements({n: lambda n=n: sweep(n, graph=True) for n in names}, repeats)
    # Includes CPU plan construction + H2D once, then all layers. It excludes
    # model dense/scatter/loss/backward and is NOT full training throughput.
    inclusive = measurements({n: lambda n=n: sweep(n, graph=True, include_preparation=True)
                              for n in names}, repeats)
    comparisons = []
    for before, after, label in [('baseline', '1_only', '1'),
                                  ('1_only', '1_3', '3_after_1'),
                                  ('1_3', '1_3_5', '5_after_1_3'),
                                  ('1_only', '1_5', '5_after_1'),
                                  ('baseline', '1_3_5', 'all')]:
        a, b = inclusive[before]['median_ms'], inclusive[after]['median_ms']
        comparisons.append(dict(change=label, before=before, after=after,
                                saved_ms=a-b, speedup=a/b))
    return dict(exact_checks=exact, layers=len(coordinates), search_only=search,
                graph=graph, graph_with_once_per_batch_preparation=inclusive,
                comparisons=comparisons, launch_counts=launch_counts(ptr))


def learned_coordinates(cpu_batch, checkpoint, epoch):
    from model import get_model, _extract_state_dict, _infer_multihead_config
    from gravnet_conv import GravNetConv
    payload = load_cpu(checkpoint)
    state = _extract_state_dict(payload)
    _, output_dim, _, _ = _infer_multihead_config(state)
    model = get_model(str(checkpoint), jit=False, input_dim=cpu_batch.x.size(1),
                      output_dim=output_dim).cuda().eval()
    if hasattr(model, 'model'):
        model = model.model
    coordinates, handles = [], []
    for module in model.modules():
        if isinstance(module, GravNetConv):
            handles.append(module.lin_s.register_forward_hook(
                lambda module, inputs, output: coordinates.append(output.detach().float().clone())))
    b = cpu_batch.clone().to('cuda')
    try:
        with torch.no_grad():
            model(b.x, b.batch, epoch=epoch)
    finally:
        for handle in handles:
            handle.remove()
    del model, b
    return coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--coordinates')
    group.add_argument('--cached-inputs')
    parser.add_argument('--checkpoint')
    parser.add_argument('--batch-index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=27)
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--output', required=True)
    parser.add_argument('--describe-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1004)
    if args.coordinates:
        payload = load_cpu(args.coordinates)
        ptr = payload['ptr'].cpu()
    else:
        batches, _ = load_cpu(args.cached_inputs)
        cpu_batch = batches[args.batch_index]
        ptr = cpu_batch.ptr
    result = dict(torch=torch.__version__, config=vars(args), launch_counts=launch_counts(ptr))
    if args.describe_only:
        result['status'] = 'CPU launch counts only; no GPU performance measured'
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: no GPU timing or bitwise claim can be made')
        if args.coordinates:
            coordinates = [x.cuda() for x in payload['coordinates']]
        else:
            if not args.checkpoint:
                parser.error('--checkpoint is required to capture learned coordinates')
            coordinates = learned_coordinates(cpu_batch, args.checkpoint, args.epoch)
        result.update(gpu=torch.cuda.get_device_name(), **ablate(coordinates, ptr, args.repeats))
        result['status'] = 'all KNN index/distance-bit/edge-order checks passed'
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('exact_checks','search_only','graph','graph_with_once_per_batch_preparation')}, indent=2))


if __name__ == '__main__':
    main()
