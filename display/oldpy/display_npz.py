import os, os.path as osp
import numpy as np
from collections import OrderedDict
import plotly.graph_objects as go

from colorwheel import ColorWheel



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

    @classmethod
    def from_npz(cls, npz_file):
        d = np.load(npz_file)
        inst = cls()
        inst.x = d['recHitFeatures'][:,1]
        inst.y = d['recHitFeatures'][:,2]
        inst.z = d['recHitFeatures'][:,3]
        inst.energy = d['recHitFeatures'][:,0]
        inst.time = d['recHitFeatures'][:,4]
        inst.truth_cluster_idx = d['recHitTruthClusterIdx']
        inst.pdgid = d['recHitPDG']
        inst.status = d['recHitStatus'].astype(np.uint32)
        return inst

    def __init__(self):
        pass

    def __getitem__(self, where):
        new = Event()
        new.x = self.x[where]
        new.y = self.y[where]
        new.z = self.z[where]
        new.energy = self.energy[where]
        new.time = self.time[where]
        new.truth_cluster_idx = self.truth_cluster_idx[where]
        new.pdgid = self.pdgid[where]
        new.status = self.status[where]

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
                ),
            text=[
                f'e={e:.3f}<br>t={t:.3f}<br>status=[<br>{s}<br>]'
                for e, t, s
                in zip(e.energy[sel], e.time[sel], e.status_str)
                ],
            hovertemplate=(
                f'x=%{{y:0.2f}}<br>y=%{{z:0.2f}}<br>z=%{{x:0.2f}}'
                f'<br>%{{text}}'
                f'<br>clusterindex={cluster_idx}'
                f'<br>pdgid={int(e.pdgid[0])}'
                # f'<br>E_bound={e.truth_e_bound_by_id(cluster_idx):.3f}'
                # f'<br>sum(E_hit)={e.energy[sel].sum():.3f}'
                f'<br>'
                ),
            name = f'cluster_{cluster_idx}',
            # opacity=1.
            ))
    return pdata


def single_pdata_to_file(
    outfile, pdata, mode='w', title=None, width=600, height=None, include_plotlyjs='cdn'
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



npz_files = [
    "npz_retagged_doublePG_1_com/npz_retagged_0_20_1.npz",
    "npz_retagged_doublePG_1_com/npz_retagged_0_21_1.npz",
    "npz_retagged_doublePG_1_com/npz_retagged_0_22_1.npz"
    ]

mode = 'w'
for npz_file in npz_files:
    e = Event.from_npz(npz_file)
    single_pdata_to_file('test.html', plot_event(e), include_plotlyjs=True, mode=mode)
    mode = 'a'