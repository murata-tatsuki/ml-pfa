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
            model.module.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])

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
