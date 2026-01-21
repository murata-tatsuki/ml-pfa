import torch
from tools.readtext import ReadText 

def get_model(ckpt = None, jit = True, input_dim = 5, output_dim = 3, ddp = False):
    # from torch_cmspepr.gravnet_model import GravnetModel
    from gravnet_model import GravnetModel
    #model = GravnetModelWithNoiseFilter(input_dim=9, output_dim=6, k=50, signal_threshold=.05)

    ckpt = ReadText("Grav_ILC_setting.txt")["Output Model File"] if ckpt is None else ckpt 
    print(f"Loading model from {ckpt=}")

    if jit:
        model = torch.jit.load(ckpt, map_location=torch.device('cpu'))

    else:
        print(f'{input_dim=}')
        model=GravnetModel(input_dim=input_dim,output_dim=output_dim)#,k=50)
        if not ddp:
            model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])
        else:
            from collections import OrderedDict
            checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
            # from collections import OrderedDict
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = k.replace("module.", "") if k.startswith("module.") else k
            #     new_state_dict[name] = v
            # model.load_state_dict(new_state_dict)

            # model.module.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])

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