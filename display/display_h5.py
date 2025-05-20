import os, os.path as osp
import numpy as np
import awkward as ak
from collections import OrderedDict
import plotly.graph_objects as go

from colorwheel import ColorWheel

import tools.load_awkward # local loader
import sys
import argparse
import math


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
    def from_awk(cls, ak_feat, ak_label, ak_pred, ak_energy, i):
        feat = ak_feat[i]
        label = ak_label[i]
        pred = ak_pred[i]
        print(ak.to_numpy(pred).shape)
        print(ak.to_numpy(label).shape)
        np_label = ak.to_numpy(label)
        
        inst = cls()
        inst.x = feat[:,1]
        inst.y = feat[:,2]
        inst.z = feat[:,3]
        inst.energy = feat[:,0]
        inst.time = feat[:,4]
        inst.track = feat[:,5]
        inst.charge = feat[:,6]
        inst.px = feat[:,7]
        inst.py = feat[:,8]
        inst.pz = feat[:,9]
        inst.truth_cluster_idx = label[:,1]
        inst.pdgid = label[:,2]
        inst.status = ak.values_astype(label[:,8],np.uint32)
        inst.mass = label[:,4]
        inst.mcpx = label[:,5]
        inst.mcpy = label[:,6]
        inst.mcpz = label[:,7]

        inst.beta = pred[:,0]
        inst.clusid = pred[:,1]
        inst.predx = pred[:,2]
        inst.predy = pred[:,3]
        inst.predcoord = pred[:,2:-3]
        inst.pred_e_cond = pred[:,-3]
        inst.pred_e_calo = pred[:,-2]
        # inst.mcen = pred[:,-1]
        mcen = np.sqrt(np_label[:,4]**2+np_label[:,5]**2+np_label[:,6]**2+np_label[:,7]**2)
        inst.mcen = mcen


        # np.set_printoptions(threshold=10000)
        # aa = np.unique(np.stack([ak.to_numpy(label[:,2]), mcen], axis=1),axis=0)
        # print(aa)
        
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
        new.track = self.track[where]
        new.charge = self.charge[where]
        new.px = self.px[where]
        new.py = self.py[where]
        new.pz = self.pz[where]
        new.truth_cluster_idx = self.truth_cluster_idx[where]
        new.pdgid = self.pdgid[where]
        new.status = self.status[where]
        new.mass = self.mass[where]
        new.mcpx = self.mcpx[where]
        new.mcpy = self.mcpy[where]
        new.mcpz = self.mcpz[where]

        new.beta = self.beta[where]
        new.clusid = self.clusid[where]
        new.predx = self.predx[where]
        new.predy = self.predy[where]
        new.predcoord = self.predcoord[where]
        new.pred_e_cond = self.pred_e_cond[where]
        new.pred_e_calo = self.pred_e_calo[where]
        new.mcen = self.mcen[where]
        # new.mcen = math.sqrt(self.mass[where]**2 + self.mcpx[where]**2 + self.mcpy[where]**2 + self.mcpz[where]**2)

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

        for tr in (1, 0):
            sel2 = sel & (e.track == tr)
            pdata.append(go.Scatter3d(
                x = e.z[sel2], y=e.x[sel2], z=e.y[sel2],
                mode='markers', 
                marker=dict(
                    line=dict(width=0),
                    size=3.,
                    color=color,
                    symbol='x' if tr else 'circle'
                ),
                text=[
                    f'e={e:.3f}<br>t={t:.3f}<br>tr={tr}<br>p=({px:.3f},{py:.3f},{pz:.3f})<br>status=[<br>{s}<br>]'
                    f'<br>clusterindex={cluster_idx}'
                    f'<br>pdgid={int(pdgid)}'
                    f'<br>mcen={mcen:.3f}'
                    for e, t, tr, px, py, pz, s, pdgid, mcen
                    # in zip(e.energy[sel2], e.time[sel2], e.track[sel2], e.px[sel2], e.py[sel2], e.pz[sel2], e.status_str, e.pdgid[sel2], math.sqrt(e.mass[sel2]**2+e.mcpx[sel2]**2+e.mcpy[sel2]**2+e.mcpz[sel2]**2))
                    in zip(e.energy[sel2], e.time[sel2], e.track[sel2], e.px[sel2], e.py[sel2], e.pz[sel2], e.status_str, e.pdgid[sel2], e.mcen[sel2])
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

def plot_event_pred(e: Event, dx=0, dy=1, use_cluster_color=False):

    pdata = []
    colorwheel = ColorWheel()
    beta_th = 0.2
    clusters = e.clusid if use_cluster_color else e.truth_cluster_idx

    for cluster_idx in np.unique(clusters):
        sel = clusters == cluster_idx
        color = colorwheel(cluster_idx)

        for tr in (1, 0):
            sel2 = [sel[i] and (e.track[i] == tr) for i in range(len(sel))]
            pdata.append(go.Scatter(
                # x = e.predx[sel2], y=e.predy[sel2],
                x = e.predcoord[:,dx][sel2], y=e.predcoord[:,dy][sel2],
                mode='markers', 
                marker=dict(
                    line=dict(width=0),
                    size=8.,
                    color=color,
                    symbol='x' if tr else 'circle'
                ),
                text=[
                    f'beta={beta}'
                    f'<br>x={x:.3f}<br>y={y:.3f}<br>z={z:.3f}<br>energy={energy:.3f}'
                    f'<br>clusterindex={cluster_idx}<br>is_track={track}'
                    f'<br>pdgid={int(pdgid)}'
                    f'<br>mcen={mcen:.3f}<br>Epred={pred_e_cond:.3f}<br>Epred_calo={pred_e_calo:.3f}'
                    for beta, x, y, z, energy, track, pdgid, mcen, pred_e_cond, pred_e_calo
                    in zip(e.beta[sel2], e.x[sel2], e.y[sel2], e.z[sel2], e.energy[sel2], e.track[sel2], e.pdgid[sel2], e.mcen[sel2], e.pred_e_cond[sel2], e.pred_e_calo[sel2])
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


def main():
    
    print("Usage: python display_h5.py inputfile outputhtml nstart nend input_dim output_dim")

    if(len(sys.argv) < 5): sys.exit()
    

    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('outputhtml')
    parser.add_argument('nstart', type=int)
    parser.add_argument('nend', type=int)
    # parser.add_argument('timingCut')
    parser.add_argument('virtual_space_coordinate_dim', type=int)
    parser.add_argument('--pandora', action='store_true', help='Use PandoraPFA result')
    parser.add_argument('--energy-regression', action='store_true', help='Turn on energy regression term on loss function and output')
    parser.add_argument('--energy-regression-cluster', action='store_true', help='Turn on energy regression term on loss function and output (regression for neutral particle)')
    parser.add_argument('-e','--momentum', action='store_true', help='Add momentum to GNN input')
    parser.add_argument('-ea','--momentum-amp', action='store_true', help='Add absoute momentum to GNN input')
    # parser.add_argument('--mctpe', action='store_true', help='Use MC truth momentum and energy for virtual hits')
    parser.add_argument('-eb','--energy-branch', action='store_true', help='Change GNN model to bypass energy')
    # parser.add_argument('--beta-d-scan', action='store_true', help='Turn on beta and diameter scan')
    # parser.add_argument('--tbeta', type=float, default=0.6)
    # parser.add_argument('--td', type=float, default=0.5)
    # parser.add_argument('--device', type=str, default='cpu', help='Specify calculation device')

    args = parser.parse_args()
    
    awkfile=sys.argv[1]
    outhtml=sys.argv[2]

    ak_feat, ak_label, ak_pred, ak_energy, ak_pandora = tools.load_awkward.load_awkward2(awkfile)

    mode = 'w'
    nstart = int(sys.argv[3])
    nend = int(sys.argv[4])
    virtual_space_coordinate_dim = int(sys.argv[5])

    print("inputfile:", awkfile, "outputhtml:",outhtml, "nstart:", nstart, "nend:", nend) 

    for i in range(nstart,nend):
        e = Event.from_awk(ak_feat, ak_label, ak_pred, ak_energy, i)
        single_pdata_to_file(outhtml, plot_event(e), include_plotlyjs=True, mode=mode)
        mode = 'a'
        for ii in range(0,math.ceil(virtual_space_coordinate_dim/2)):
            dx = ii * 2
            dy = ii * 2 + 1
            title=f'virtual coordinate {dx}, {dy}'
            single_pdata_to_file_pred(outhtml, plot_event_pred(e, dx=dx, dy=dy), title=title, include_plotlyjs=True, mode=mode)
            single_pdata_to_file_pred(outhtml, plot_event_pred(e, dx=dx, dy=dy, use_cluster_color=True), title=title, include_plotlyjs=True, mode=mode)


    
if __name__=='__main__':
    main()
    

