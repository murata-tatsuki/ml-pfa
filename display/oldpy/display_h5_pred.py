import os, os.path as osp
import numpy as np
import awkward as ak
from collections import OrderedDict
import plotly.graph_objects as go

from colorwheel import ColorWheel

import tools.load_awkward # local loader
import sys

class Event:

    status_to_str = OrderedDict()
    status_to_str[31] = 'Endpoint'
    status_to_str[30] = 'CreatedInSimulation'
    status_to_str[29] = 'Backscatter'
    status_to_str[28] = 'VertexIsNotEndpointOfParent'
    status_to_str[27] = 'DecayedInTracker'
    status_to_str[26] = 'DecayedInCalorimeter'
    status_to_str[25] = 'LeftDetector'
    status_to_str[24] = 'Stopped'
    status_to_str[23] = 'Overlay'


    # ak_feat: edep, x, y, z, time, track, charge, px, py, pz (atcalo)
    # ak_label: hitid, mcid, pdg, charge, mass, px, py, pz (of mcp), status
    # ak_pred: beta, x, y
    @classmethod
    def from_awk(cls, ak_feat, ak_label, ak_pred, i):
        feat = ak_feat[i]
        label = ak_label[i]
        pred = ak_pred[i]
        inst = cls()
        inst.x = feat[:,1]
        inst.y = feat[:,2]
        inst.z = feat[:,3]
        inst.energy = feat[:,0]
        #inst.time = feat[:,4]
        #inst.track = feat[:,5]
        #inst.charge = feat[:,6]
        #inst.px = feat[:,7]
        #inst.py = feat[:,8]
        #inst.pz = feat[:,9]
        inst.truth_cluster_idx = label
        #inst.pdgid = label[:,2]
        #inst.status = ak.values_astype(label[:,8],np.uint32)
        #inst.mcpx = label[:,5]
        #inst.mcpy = label[:,6]
        #inst.mcpz = label[:,7]

        inst.beta = pred[:,0]
        inst.predx = pred[:,1]
        inst.predy = pred[:,2]
        
        return inst

    def __init__(self):
        pass

    def __getitem__(self, where):
        new = Event()
        new.x = self.x[where]
        new.y = self.y[where]
        new.z = self.z[where]
        new.energy = self.energy[where]
        #new.time = self.time[where]
        #new.track = self.track[where]
        #new.charge = self.charge[where]
        #new.px = self.px[where]
        #new.py = self.py[where]
        #new.pz = self.pz[where]
        new.truth_cluster_idx = self.truth_cluster_idx[where]
        #new.pdgid = self.pdgid[where]
        #new.status = self.status[where]
        #new.mcpx = self.mcpx[where]
        #new.mcpy = self.mcpy[where]
        #new.mcpz = self.mcpz[where]

        new.beta = self.beta[where]
        new.predx = self.predx[where]
        new.predy = self.predy[where]

    def __len__(self):
        return len(self.x)

    @property
    def status_str(self):
        out = []
        for i in range(len(self)):
            stati = []
            for bit, status in self.status_to_str.items():
                if (self.status[i] >> bit) & 1:
                    stati.append(status)
            out.append('  ' + '<br>  '.join(stati))
        return out



def plot_event(e: Event):

    pdata = []
    colorwheel = ColorWheel()

    for cluster_idx in np.unique(e.truth_cluster_idx):
        sel = e.truth_cluster_idx == cluster_idx
        color = colorwheel(cluster_idx)

        pdata.append(go.Scatter3d(
            x = e.z[sel], y=e.x[sel], z=e.y[sel],
            mode='markers', 
            marker=dict(
                line=dict(width=0),
                size=3.,
                color=color,
                symbol='circle'
            ),
            text=[
                f'beta={beta:.3f}<br>predx={predx:.3f}<br>predy={predy:.3f}<br>energy={energy:.3f}'
                f'<br>clusterindex={cluster_idx}'
                for beta, predx, predy, energy
                in zip(e.beta[sel], e.predx[sel], e.predy[sel], e.energy[sel])
            ],
            hovertemplate=(
                f'x=%{{y:0.2f}}<br>y=%{{z:0.2f}}<br>z=%{{x:0.2f}}'
                f'<br>%{{text}}'                
                # f'<br>E_bound={e.truth_e_bound_by_id(cluster_idx):.3f}'
                # f'<br>sum(E_hit)={e.energy[sel].sum():.3f}'
                f'<br>'
            ),
            name = f'cluster_{cluster_idx}',
            # opacity=1.
        ))
    return pdata

def plot_event_pred(e: Event):

    pdata = []
    colorwheel = ColorWheel()
    beta_th = 0.2

    for cluster_idx in np.unique(e.truth_cluster_idx):
        sel = e.truth_cluster_idx == cluster_idx
        color = colorwheel(cluster_idx)

        for tr in (1, 0):
            sel2 = sel & (e.beta > beta_th)
            pdata.append(go.Scatter(
                x = e.predx[sel2], y=e.predy[sel2],
                mode='markers', 
                marker=dict(
                    line=dict(width=0),
                    size=3.,
                    color=color,
                    symbol='x' if tr else 'circle'
                ),
                text=[
                    f'beta={beta}'
                    f'<br>x={x:.3f}<br>y={y:.3f}<br>y={y:.3f}<br>energy={energy:.3f}'
                    f'<br>clusterindex={cluster_idx}'
                    for beta, x, y, z, energy
                    in zip(e.beta[sel2], e.x[sel2], e.y[sel2], e.z[sel2], e.energy[sel2])
                ],
                hovertemplate=(
                    f'x=%{{x:0.2f}}<br>y=%{{y:0.2f}}'
                    f'<br>%{{text}}'                
                    # f'<br>E_bound={e.truth_e_bound_by_id(cluster_idx):.3f}'
                    # f'<br>sum(E_hit)={e.energy[sel].sum():.3f}'
                    f'<br>'
                ),
                name = f'cluster_{cluster_idx}',
                # opacity=1.
            ))
    return pdata


def single_pdata_to_file(
    outfile, pdata, mode='w', title=None, width=800, height=None, include_plotlyjs='cdn'
    ):
    import plotly.graph_objects as go

    scene = dict(xaxis_title='z (cm)', yaxis_title='x (cm)', zaxis_title='y (cm)', aspectmode='cube')
    if height is None: height = width
    fig = go.Figure(data=pdata, **(dict(layout_title_text=title) if title else {}))
    fig.update_layout(width=width, height=height, scene=scene)
    fig_html = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)

    print('Writing to', outfile)
    os.makedirs(osp.dirname(osp.abspath(outfile)), exist_ok=True)
    with open(outfile, mode) as f:
        f.write(fig_html)

def single_pdata_to_file_pred(
    outfile, pdata, mode='w', title=None, width=800, height=None, include_plotlyjs='cdn'
    ):
    import plotly.graph_objects as go

    scene = dict(xaxis_title='z (cm)', yaxis_title='x (cm)')
    if height is None: height = width
    fig = go.Figure(data=pdata, **(dict(layout_title_text=title) if title else {}))
    fig.update_layout(width=width, height=height, scene=scene)
    fig_html = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)

    print('Writing to', outfile)
    os.makedirs(osp.dirname(osp.abspath(outfile)), exist_ok=True)
    with open(outfile, mode) as f:
        f.write(fig_html)


print("Usage: python display_h5.py inputfile outputhtml nstart nend")

if(len(sys.argv) < 5): sys.exit()

awkfile=sys.argv[1]
outhtml=sys.argv[2]

ak_feat, ak_label, ak_pred = tools.load_awkward.load_awkward2(awkfile)

mode = 'w'
nstart = int(sys.argv[3])
nend = int(sys.argv[4])

print("inputfile:", awkfile, "outputhtml:",outhtml, "nstart:", nstart, "nend:", nend) 

for i in range(nstart,nend):
    e = Event.from_awk(ak_feat, ak_label, ak_pred, i)
    single_pdata_to_file(outhtml, plot_event(e), include_plotlyjs=True, mode=mode)
    mode = 'a'
    single_pdata_to_file_pred(outhtml, plot_event_pred(e), include_plotlyjs=True, mode=mode)
