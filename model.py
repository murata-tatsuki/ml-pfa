import torch
from tools.readtext import ReadText 

def get_model(ckpt = None, jit = True, input_dim = 5, output_dim = 3):
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
        model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])        

    return model
