"""Bounded GPU comparison using a real batch of 32 and learned GravNet coordinates.

Run on a GPU inside the training container. JSON includes exact-match checks and
paired wall-clock timings. This does not train or overwrite the source checkpoint.
Full training throughput should additionally be measured with train.py's benchmark.
"""
import argparse
import json
import statistics
import time
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

import torch
from torch_geometric.loader import DataLoader
from torch_cmspepr import select_knn as legacy_select, knn_graph as legacy_graph
from knn_event_parallel import select_knn, knn_graph, load_extension
from dataset_ilc_streaming import ILCStreamingDataset
from gravnet_model import GravNetModelMultiHead
from gravnet_conv import GravNetConv, configure_knn_backend


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000, result


def compare_pair(old, new, repeats):
    for _ in range(3):
        old(); new()
    times = {'legacy': [], 'event-parallel': []}
    for i in range(repeats):
        order = [('legacy', old), ('event-parallel', new)]
        for name, call in order[::1 if i % 2 == 0 else -1]:
            ms, _ = timed(call)
            times[name].append(ms)
    medians = {key: statistics.median(values) for key, values in times.items()}
    return dict(samples_ms=times, median_ms=medians,
                speedup=medians['legacy'] / medians['event-parallel'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=12)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.manual_seed(1009)
    load_extension()
    dataset = ILCStreamingDataset(args.data, thetaphi=True, test_mode=True,
        momentum=True, momentumAmp=True, extended_h5_input=True, exclude_gap_hits=True,
        shuffle=False, files_per_chunk=1, ddp_rank=0, ddp_world_size=1)
    data = next(iter(DataLoader(dataset, batch_size=32, num_workers=0))).to('cuda')
    assert data.num_graphs == 32
    model = GravNetModelMultiHead(input_dim=data.x.shape[1], output_dim=5, n_heads=6,
        regression_output_dims=[1]*5, interaction_mode='none', pid_head=True,
        five_particle_energy_heads=True).cuda().eval()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model'], strict=True)
    coordinates, hooks = [], []
    for layer in model.modules():
        if isinstance(layer, GravNetConv):
            hooks.append(layer.lin_s.register_forward_hook(
                lambda module, inputs, output: coordinates.append(output.detach().clone())))
    with torch.no_grad():
        old_output = model(data.x, data.batch)
    for hook in hooks:
        hook.remove()
    result = dict(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        batch_size=data.num_graphs, precision='fp32', checkpoint=args.checkpoint,
        hits_per_event=(data.ptr[1:] - data.ptr[:-1]).tolist(), layers=[])
    for number, x in enumerate(coordinates):
        old_i, old_d = legacy_select(x, 41, data.batch)
        new_i, new_d = select_knn(x, 41, data.batch)
        index_mismatches = int((old_i != new_i).sum())
        distance_bit_mismatches = int((old_d.view(torch.int32) != new_d.view(torch.int32)).sum())
        assert index_mismatches == distance_bit_mismatches == 0
        assert torch.equal(legacy_graph(x, 40, data.batch), knn_graph(x, 40, data.batch))
        mask = torch.ones(x.shape[0], dtype=torch.int32, device='cuda')
        rows = data.ptr.to(dtype=torch.int32)
        raw_old = lambda: torch.ops.select_knn_cuda.select_knn_cuda(x, rows, mask, 41, 1e9, 1)
        raw_new = lambda: torch.ops.pfa_knn_event_parallel.select_knn_cuda(x, rows, mask, 41, 1e9, 1)
        record = dict(layer=number, index_mismatches=index_mismatches,
            distance_bit_mismatches=distance_bit_mismatches,
            extension=compare_pair(raw_old, raw_new, args.repeats),
            graph=compare_pair(lambda: legacy_graph(x, 40, data.batch),
                               lambda: knn_graph(x, 40, data.batch), args.repeats))
        result['layers'].append(record)
        print('LAYER', number, 'extension_speedup', record['extension']['speedup'], flush=True)
    repeat_coordinates, new_coordinates = [], []

    def forward_capture(storage):
        handles = [layer.lin_s.register_forward_hook(
            lambda module, inputs, output: storage.append(output.detach().clone()))
            for layer in model.modules() if isinstance(layer, GravNetConv)]
        try:
            with torch.no_grad():
                return model(data.x, data.batch)
        finally:
            for handle in handles:
                handle.remove()

    repeat_output = forward_capture(repeat_coordinates)
    configure_knn_backend(model, 'event-parallel')
    new_output = forward_capture(new_coordinates)
    differences = []
    baseline_differences = []
    for old, repeat, new in zip(old_output['all_heads'] + [old_output['pid_logits']],
                        repeat_output['all_heads'] + [repeat_output['pid_logits']],
                        new_output['all_heads'] + [new_output['pid_logits']]):
        differences.append(float((old-new).abs().max()))
        baseline_differences.append(float((old-repeat).abs().max()))
    print('MODEL_DIFF old_new=', differences, 'old_old=', baseline_differences, flush=True)
    result['legacy_repeat_max_abs_differences'] = baseline_differences
    result['model_head_max_abs_differences'] = differences
    result['knn_checks_passed'] = True
    result['model_close'] = all(torch.allclose(a,b,atol=1e-4,rtol=1e-4) for a,b in zip(old_output['all_heads']+[old_output['pid_logits']],new_output['all_heads']+[new_output['pid_logits']]))
    result['coordinate_max_abs_differences'] = {
        'legacy_repeat': [float((a-b).abs().max()) for a,b in zip(coordinates, repeat_coordinates)],
        'legacy_parallel': [float((a-b).abs().max()) for a,b in zip(coordinates, new_coordinates)],
    }
    # Diagnose the discontinuity of rebuilding KNN after nondeterministic GPU
    # reductions. This is validation only: never freeze coordinates in training.
    handles = []
    for layer, frozen in zip((m for m in model.modules() if isinstance(m, GravNetConv)), coordinates):
        handles.append(layer.lin_s.register_forward_hook(
            lambda module, inputs, output, frozen=frozen: frozen))
    fixed_outputs = []
    try:
        for backend in ('legacy', 'event-parallel'):
            configure_knn_backend(model, backend)
            with torch.no_grad():
                fixed_outputs.append(model(data.x, data.batch))
    finally:
        for handle in handles:
            handle.remove()
    fixed_differences = []
    for a, b in zip(fixed_outputs[0]['all_heads'] + [fixed_outputs[0]['pid_logits']],
                    fixed_outputs[1]['all_heads'] + [fixed_outputs[1]['pid_logits']]):
        fixed_differences.append(float((a-b).abs().max()))
    result['fixed_coordinate_model_head_max_abs_differences'] = fixed_differences
    # Holding coordinates fixed still leaves GPU scatter's atomic reductions.
    # For a full-model equivalence check, use deterministic CPU reductions only
    # in this diagnostic. KNN and all dense layers still run on the GPU. These
    # forwards are NOT part of any speed measurement or training configuration.
    import gravnet_model
    import gravnet_conv

    def cpu_reference(function):
        def call(src, index, *args, **kwargs):
            value = function(src.cpu(), index.cpu(), *args, **kwargs)
            if isinstance(value, tuple):
                return tuple(v.to(src.device) for v in value)
            return value.to(src.device)
        return call

    reference_outputs = []
    torch.set_num_threads(1)
    with ExitStack() as stack:
        for module, name in ((gravnet_model, 'scatter_mean'), (gravnet_model, 'scatter_min'),
                             (gravnet_model, 'scatter_max'), (gravnet_conv, 'scatter')):
            stack.enter_context(patch.object(module, name, cpu_reference(getattr(module, name))))
        for backend in ('legacy', 'event-parallel'):
            configure_knn_backend(model, backend)
            with torch.no_grad():
                reference_outputs.append(model(data.x, data.batch))
    reference_differences = []
    for a, b in zip(reference_outputs[0]['all_heads'] + [reference_outputs[0]['pid_logits']],
                    reference_outputs[1]['all_heads'] + [reference_outputs[1]['pid_logits']]):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
        reference_differences.append(float((a-b).abs().max()))
    result['cpu_reference_reduction_model_head_max_abs_differences'] = reference_differences
    result['cpu_reference_reduction_model_checks_passed'] = True
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print('RESULT', args.output, flush=True)


if __name__ == '__main__':
    main()
