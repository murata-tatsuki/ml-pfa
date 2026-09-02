import re

import torch
import torch.nn as nn
from tools.readtext import ReadText 


def _strip_module_prefix(state_dict):
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def _extract_state_dict(checkpoint):
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    return _strip_module_prefix(state_dict)


def _is_multihead_state_dict(state_dict):
    return any(
        key.startswith("head_postgn_dense.") or key.startswith("head_output.")
        for key in state_dict
    )


def _infer_multihead_interaction_mode(state_dict):
    if any(".fusion." in key for key in state_dict if key.startswith("interaction_blocks.")):
        return "concat"
    if any(".projection." in key for key in state_dict if key.startswith("interaction_blocks.")):
        return "add"
    if any(".gate." in key for key in state_dict if key.startswith("interaction_blocks.")):
        return "gate"
    return "none"


def _infer_multihead_config(state_dict):
    head_pattern = re.compile(r"^head_output\.(\d+)\.4\.weight$")
    head_dims = {}
    for key, value in state_dict.items():
        match = head_pattern.match(key)
        if match:
            head_dims[int(match.group(1))] = int(value.shape[0])

    if not head_dims or 0 not in head_dims:
        raise ValueError("Failed to infer multi-head configuration from checkpoint")

    n_heads = max(head_dims) + 1
    clustering_output_dim = head_dims[0]
    regression_output_dims = [head_dims[i] for i in range(1, n_heads)]
    interaction_mode = _infer_multihead_interaction_mode(state_dict)
    return n_heads, clustering_output_dim, regression_output_dims, interaction_mode


class LegacyCompatibleMultiHeadAdapter(nn.Module):
    def __init__(
        self,
        model,
        use_charge_track_likeness=False,
        energy_regression=False,
        energy_regression_cluster=False,
    ):
        super().__init__()
        self.model = model
        self.use_charge_track_likeness = use_charge_track_likeness
        self.energy_regression = energy_regression
        self.energy_regression_cluster = energy_regression_cluster

    def forward(self, x, batch, epoch=None, return_dict=False):
        result = self.model(x, batch, epoch=epoch, return_dict=True)
        if return_dict:
            return result

        clustering = result["clustering"]
        regressions = result.get("regressions", [])

        parts = [clustering[:, :1]]
        coord_start = 1

        if self.use_charge_track_likeness:
            parts.append(clustering[:, 1:2])
            coord_start = 2

        if self.energy_regression:
            if len(regressions) < 1:
                raise RuntimeError("Multi-head model does not contain tracker-energy head")
            parts.append(regressions[0].reshape(-1, 1))

        if self.energy_regression and self.energy_regression_cluster:
            if len(regressions) < 2:
                raise RuntimeError("Multi-head model does not contain cluster-energy head")
            parts.append(regressions[1].reshape(-1, 1))

        parts.append(clustering[:, coord_start:])
        return torch.cat(parts, dim=1)


def get_model(
    ckpt = None,
    jit = True,
    input_dim = 5,
    output_dim = 3,
    ddp = False,
    use_charge_track_likeness = False,
    energy_regression = False,
    energy_regression_cluster = False,
    energy_regression_weight = False,
    model_variant = "auto",
):
    # from torch_cmspepr.gravnet_model import GravnetModel
    from gravnet_model import GravnetModel
    #model = GravnetModelWithNoiseFilter(input_dim=9, output_dim=6, k=50, signal_threshold=.05)

    ckpt = ReadText("Grav_ILC_setting.txt")["Output Model File"] if ckpt is None else ckpt 
    print(f"Loading model from {ckpt=}")

    if jit:
        model = torch.jit.load(ckpt, map_location=torch.device('cpu'))

    else:
        print(f'{input_dim=}')
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        state_dict = _extract_state_dict(checkpoint)

        if model_variant not in {"auto", "legacy", "multihead"}:
            raise ValueError(f"Unknown model_variant: {model_variant}")

        checkpoint_is_multihead = _is_multihead_state_dict(state_dict)
        use_multihead = checkpoint_is_multihead if model_variant == "auto" else (model_variant == "multihead")

        if use_multihead and not checkpoint_is_multihead:
            raise ValueError("Requested multihead model, but checkpoint does not look like multi-head")
        if model_variant == "legacy" and checkpoint_is_multihead:
            raise ValueError("Requested legacy model, but checkpoint looks like multi-head")

        if use_multihead:
            if energy_regression_weight:
                raise ValueError("Multi-head inference does not support --energy-regression-weight yet")

            from gravnet_model import GravNetModelMultiHead

            n_heads, clustering_output_dim, regression_output_dims, interaction_mode = _infer_multihead_config(state_dict)
            inferred_extra_dims = output_dim - clustering_output_dim
            inferred_energy_regression = energy_regression
            inferred_energy_regression_cluster = energy_regression_cluster

            if not inferred_energy_regression and not inferred_energy_regression_cluster:
                if inferred_extra_dims == 1:
                    inferred_energy_regression = True
                elif inferred_extra_dims == 2:
                    inferred_energy_regression = True
                    inferred_energy_regression_cluster = True
                elif inferred_extra_dims not in (0,):
                    raise ValueError(
                        "Could not infer legacy-compatible layout for multi-head checkpoint: "
                        f"requested output_dim={output_dim}, clustering_output_dim={clustering_output_dim}"
                    )

            print(
                "Using multi-head model: "
                f"heads={n_heads}, clustering_out={clustering_output_dim}, "
                f"regression_out={regression_output_dims}, interaction={interaction_mode}"
            )
            model = GravNetModelMultiHead(
                input_dim=input_dim,
                output_dim=clustering_output_dim,
                n_heads=n_heads,
                regression_output_dims=regression_output_dims,
                interaction_mode=interaction_mode,
            )
            model.load_state_dict(state_dict)
            model = LegacyCompatibleMultiHeadAdapter(
                model,
                use_charge_track_likeness=use_charge_track_likeness,
                energy_regression=inferred_energy_regression,
                energy_regression_cluster=inferred_energy_regression_cluster,
            )
        else:
            print("Using legacy GravnetModel")
            model=GravnetModel(input_dim=input_dim,output_dim=output_dim)#,k=50)
            if not ddp:
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict, strict=False)

    return model


def get_model_branch(ckpt = None, jit = True, input_dim = 5, output_dim = 3, ddp = False):
    # from torch_cmspepr.gravnet_model import GravnetModel
    from gravnet_model import GravNetModelBranch
    #model = GravnetModelWithNoiseFilter(input_dim=9, output_dim=6, k=50, signal_threshold=.05)

    ckpt = ReadText("Grav_ILC_setting.txt")["Output Model File"] if ckpt is None else ckpt 
    print(f"Loading model from {ckpt=}")

    if jit:
        model = torch.jit.load(ckpt, map_location=torch.device('cpu'))

    else:
        print(f'{input_dim=}')
        model=GravNetModelBranch(input_dim=input_dim,output_dim=output_dim,b_energy_branch=True)#,k=50)
        if not ddp:
            model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])        
        else:
            model.module.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])        

    return model


def get_clustering_model(ckpt = None, jit = True, input_dim = 5, output_dim = 3, ddp = False, lcr_block = True, pid = False, score_raw = False):
    # from torch_cmspepr.gravnet_model import GravnetModel
    from lcr_module import LCR, LCR_withPID, LCR_withClass, LCR_Block, LCR_Block_modifiedOutput, LCR_Block_modifiedOutput_moreParameters, LCR_Block_modifiedOutput_moreParameters_trackQuery, hungarian_set_loss, hungarian_set_loss_bbox_only
    #model = GravnetModelWithNoiseFilter(input_dim=9, output_dim=6, k=50, signal_threshold=.05)
    # LCR(embed_dim_=4,embed_dim=128, num_heads=8, K=256, feat_dim=4)
    # LCR_Block(embed_dim_=7,embed_dim=128, num_heads=8, num_layers=4, K=256, feat_dim=4)

    ckpt = ReadText("Grav_ILC_setting.txt")["Output Model File"] if ckpt is None else ckpt 
    print(f"Loading model from {ckpt=}")

    if jit:
        model = torch.jit.load(ckpt, map_location=torch.device('cpu'))

    else:
        if lcr_block:
            print(f'{input_dim=}')
            if not pid:
                model=LCR_Block(embed_dim_=7,embed_dim=128, num_heads=8, num_layers=4, feat_dim=4)
                # model=LCR_Block_modifiedOutput(embed_dim_=7,embed_dim=128, num_heads=8, num_layers=4, feat_dim=4)
            else:
                # model=LCR_Block_modifiedOutput_moreParameters(embed_dim_=17,embed_dim=256, num_heads=8, num_layers=4, feat_dim=4, num_particle_classes=5)
                model=LCR_Block_modifiedOutput_moreParameters_trackQuery(embed_dim_=17,embed_dim=256, num_heads=8, num_layers=8, feat_dim=4, num_particle_classes=5, score_raw=score_raw)
            if not ddp:
                model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])
            else:
                model.module.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])
        else:
            print(f'{input_dim=}')
            model=LCR(embed_dim_=4,embed_dim=128, num_heads=8, K=256, feat_dim=4)
            if not ddp:
                model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])
            else:
                model.module.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])

    return model