"""Read-only hooks for the coordinates actually used by each GravNet kNN graph."""
import numpy as np


class SpaceCapture:
    def __init__(self, model, coordinates=False, features=False, layer=None):
        self.handles = []
        self.values = {}
        self.specs = {}
        modules = dict(model.named_modules())
        targets = {}
        if coordinates or features:
            blocks = [(name, module) for name, module in modules.items()
                      if name.endswith('gravnet_blocks')]
            if len(blocks) != 1:
                raise ValueError('Expected exactly one GravNet block stack')
            prefix, stack = blocks[0]
            for i, block in enumerate(stack):
                if coordinates:
                    key = f'gravnet_{i+1}_coords'
                    targets[key] = f'{prefix}.{i}.gravnet_layer.lin_s'
                    self.specs[key] = dict(kind='knn_coordinates', block=i+1,
                        description='Learned coordinates used inside this block for kNN; before aggregation')
                if features:
                    key = f'gravnet_{i+1}_features'
                    targets[key] = f'{prefix}.{i}'
                    self.specs[key] = dict(kind='block_output_features', block=i+1,
                        description='Features at the end of this block; not spatial coordinates')
        if layer:
            targets['intermediate'] = layer
            self.specs['intermediate'] = dict(kind='module_output', description='Requested module output')
        for key, name in targets.items():
            if name not in modules:
                raise ValueError(f'Unknown module {name}; available: {list(modules)}')
            self.specs[key]['module'] = name
        try:
            for key, name in targets.items():
                self.handles.append(modules[name].register_forward_hook(self._hook(key)))
        except Exception:
            self.close()
            raise

    def _hook(self, key):
        def capture(module, inputs, output):
            import torch
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise ValueError(f'{key} must return an (N, D) tensor')
            self.values.setdefault(key, []).append(output.detach().cpu().clone().numpy())
            # No return value: hooks must never replace model outputs.
        return capture

    def clear(self):
        self.values.clear()

    def collect(self, rows):
        result = {}
        for key in self.specs:
            values = self.values.get(key, [])
            if len(values) != 1 or values[0].shape[0] != rows:
                raise ValueError(f'{key} must be called once with {rows} event-aligned rows')
            if not np.isfinite(values[0]).all():
                raise ValueError(f'{key} contains non-finite values')
            result[key] = values[0]
            self.specs[key]['dimensions'] = values[0].shape[1]
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
