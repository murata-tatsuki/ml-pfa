#from torch_geometric.data import Data

import numpy as np
#import tqdm
#from time import strftime
#import os, os.path as osp
import uuid

#import tools.load_awkward as la
#import awkward as ak

#from dataset import ilc_dataset
#import objectcondensation as oc

from event import Event
from colorwheel import ColorWheel, HighlightColorwheel, ColorwheelWithProps
from stats import Stats

import event_view_plot as plt_3d 
import warnings
warnings.filterwarnings("ignore")


def pca_down(cluster_space_coords: np.array, n_components: int = 3):
    from sklearn.decomposition import PCA
    dim = cluster_space_coords.shape[1]
    if dim <= n_components: return cluster_space_coords
    pca = PCA(n_components)
    out = pca.fit_transform(cluster_space_coords)
    assert out.shape == (cluster_space_coords.shape[0], n_components)
    return out

def show_Stats(event,clustering,matches,noise_index=0):
    stats = Stats()
    tag = np.unique(event.y)
    for i in tag:
        stats.add('event_y',i)
    
    return stats

def get_hit_matched_vs_unmatched_noStat(event:Event, clustering, matches, noise_index=0):
    for matches_i in matches:
        rate_pred=[]
        matched_pred = np.array([])
        matched_truth = np.where(event.y == matches_i[0])
        for matches_pred_id in matches_i[1]:
            matched_pred = np.append(matched_pred,np.count_nonzero(clustering[matched_truth] == matches_pred_id))   
        for matched_pred_i in matched_pred:
            rate_pred=np.append(rate_pred,matched_pred_i/len(matched_truth[0]))
    if np.max(rate_pred) < 0.9:
        plt_3d.event_display(event, clustering)
        print("If you want to quit, enter 'y'.")
        x = input('>> ')
        if x == "y" : return "y"
    return "z"

def get_hit_matched_vs_unmatched_noStat_eachEvent(event:Event, clustering, matches, noise_index=0):
    for matches_i in matches:
        rate_pred=[]
        matched_pred = np.array([])
        matched_truth = np.where(event.y == matches_i[0])
        for matches_pred_id in matches_i[1]:
            matched_pred = np.append(matched_pred,np.count_nonzero(clustering[matched_truth] == matches_pred_id))   
        for matched_pred_i in matched_pred:
            rate_pred=np.append(rate_pred,matched_pred_i/len(matched_truth[0]))

    plt_3d.event_display(event, clustering)
    print("If you want to quit, enter 'y'.")
    x = input('>> ')
    if x == "y" : return "y"
    return "z"

def get_hit_matched_vs_unmatched(event:Event, clustering, matches, noise_index=0):
    stats = Stats()
    #if len(set(event.y))>2 : return stats

    #print(f"{event.y=}")
    #print(f"{clustering=}")

    # event-wise matching
    matched_truth_sum = 0
    matched_pred_sum = 0
    matched_truth_sum = len(event.y)

    for matches_i in matches:
        rate_pred=[]
        matched_pred = np.array([])
        matched_truth = np.where(event.y == matches_i[0])
        for matches_pred_id in matches_i[1]:
            matched_pred = np.append(matched_pred,np.count_nonzero(clustering[matched_truth] == matches_pred_id))
        for matched_pred_i in matched_pred:
            #print(f"{matched_pred_i=}")
            matched_pred_sum += matched_pred_i
    #print(f"{matched_pred_sum=}")
    #print(f"{matched_truth_sum=}")
    stats.add('matched_pred_ev',matched_pred_sum)
    stats.add('matched_truth_ev',matched_truth_sum)
    stats.add('rate_ev', 0 if matched_truth_sum == 0 else matched_pred_sum/matched_truth_sum)

    # cluster-wise matching
    for matches_i in matches:
        rate_pred=[]
        matched_pred = np.array([])
        matched_truth = np.where(event.y == matches_i[0])
        for matches_pred_id in matches_i[1]:
            matched_pred = np.append(matched_pred,np.count_nonzero(clustering[matched_truth] == matches_pred_id))
            
        stats.add('num_pred',len(matched_truth[0]))
        for matched_pred_i in matched_pred:
            rate_pred=np.append(rate_pred,matched_pred_i/len(matched_truth[0]))
        stats.add('rate_pred',np.max(rate_pred))

        #print("====")
        #print(f"{event.y=}")
        #print(f"{clustering=}")
        #print(f"{matches_i[0]=}")
        #print(f"{matches_i[1]=}")
        #print("num_pred=", len(matched_truth[0]))
        #print("rate_pred=", np.max(rate_pred))
    return stats

def get_hit_matched_vs_unmatched_energy(event:Event, clustering, matches, noise_index=0):
    stats = Stats()
    #if len(set(event.y))>2 : return stats
    for matches_i in matches:
        rate_pred_energy=[]
        matched_pred_energy = np.array([])
        matched_truth = np.where(event.y == matches_i[0])
        for matches_pred_id in matches_i[1]:
            matched_pred_energy = np.append(matched_pred_energy,np.sum(event.x[:,0][np.where(clustering[matched_truth] == matches_pred_id)]))
        stats.add('num_pred_energy',np.sum(event.x[:,0][np.where(clustering[matched_truth] == matches_pred_id)]))   
        for matched_pred_i in matched_pred_energy:
            rate_pred=np.append(rate_pred_energy,matched_pred_energy/np.sum(event.x[:,0]))
        stats.add('rate_pred_energy',np.max(rate_pred_energy))
    return stats

def get_matched_vs_unmatched(event: Event , clustering, matches, noise_index=0):
    matched_truth = []
    matched_pred = []
    for truth_ids, pred_ids in matches:
        matched_truth.extend(truth_ids)
        matched_pred.extend(pred_ids)
    all_truth_ids = set(np.unique(event.y))
    all_pred_ids = set(np.unique(clustering))
    all_truth_ids.discard(noise_index)
    all_pred_ids.discard(noise_index)
    unmatched_truth = np.array(list(all_truth_ids - set(matched_truth)))
    unmatched_pred = np.array(list(all_pred_ids - set(matched_pred)))
    
    select_matched_truth = np.in1d(event.y, matched_truth)
    select_matched_pred = np.in1d(clustering, matched_pred)
    select_unmatched_truth = np.in1d(event.y, unmatched_truth)
    select_unmatched_pred = np.in1d(clustering, unmatched_pred)

    nhits = (event.y != noise_index).sum()
    total_truth_energy = event.energy[event.y != noise_index].sum()
    total_pred_energy = event.energy[clustering != noise_index].sum()

    stats = Stats()
    #stats.add('showers_truth', all_truth_ids)
    #stats.add('n_showers_truth', len(all_truth_ids))
    stats.add('n_showers_pred', len(all_pred_ids))
    stats.add('n_showers_unmatched_truth', len(unmatched_truth))
    stats.add('n_showers_unmatched_pred', len(unmatched_pred))

    all_truth_pdgids = np.array([event.truth_pdgid_by_id(id) for id in all_truth_ids])
    unmatched_truth_pdgids = np.array([event.truth_pdgid_by_id(id) for id in unmatched_truth])

    count_type_fns = {
        'em' : lambda pdgids: np.in1d(np.abs(pdgids), np.array([11, 22, 111])).sum(),
        'mip' : lambda pdgids: np.in1d(np.abs(pdgids), np.array([13])).sum(),
        'had' : lambda pdgids: (~np.in1d(np.abs(pdgids), np.array([11, 22, 111, 13]))).sum(),
        }

    # print('allcat unmatched/total:', len(unmatched_truth), len(all_truth_ids))
    for i_cat, cat in enumerate(['em', 'had', 'mip']):
        count_type = count_type_fns[cat]
        n_truth_showers_this_cat = count_type(all_truth_pdgids)
        n_unmatched_truth_showers_this_cat = count_type(unmatched_truth_pdgids)
        if n_truth_showers_this_cat == 0: continue
        # print(cat, unmatched_truth_pdgids, n_unmatched_truth_showers_this_cat, n_truth_showers_this_cat)
        stats.add(f'n_showers_truth_{cat}', n_truth_showers_this_cat)
        stats.add(f'n_showers_unmatched_truth_{cat}', n_unmatched_truth_showers_this_cat)

    stats.add('nhits_matched_truth', select_matched_truth.sum())
    stats.add('nhits_matched_pred', select_matched_pred.sum())
    stats.add('nhits_unmatched_truth', select_unmatched_truth.sum())
    stats.add('nhits_unmatched_pred', select_unmatched_pred.sum())
    stats.add('hitenergy_matched_truth', event.energy[select_matched_truth].sum())
    stats.add('hitenergy_matched_pred', event.energy[select_matched_pred].sum())
    stats.add('hitenergy_unmatched_truth', event.energy[select_unmatched_truth].sum())
    stats.add('hitenergy_unmatched_pred', event.energy[select_unmatched_pred].sum())
    stats.add('fraction_nhits_matched_truth', select_matched_truth.sum()/nhits)
    stats.add('fraction_nhits_matched_pred', select_matched_pred.sum()/nhits)
    stats.add('fraction_nhits_unmatched_truth', select_unmatched_truth.sum()/nhits)
    stats.add('fraction_nhits_unmatched_pred', select_unmatched_pred.sum()/nhits)
    stats.add('fraction_hitenergy_matched_truth', event.energy[select_matched_truth].sum()/total_truth_energy)
    stats.add('fraction_hitenergy_matched_pred', event.energy[select_matched_pred].sum()/total_pred_energy)
    stats.add('fraction_hitenergy_unmatched_truth', event.energy[select_unmatched_truth].sum()/total_truth_energy)
    stats.add('fraction_hitenergy_unmatched_pred', event.energy[select_unmatched_pred].sum()/total_pred_energy)
    return stats





def get_matched_vs_unmatched_charged_neutral(event: Event , clustering, matches, noise_index=0):
    true_charged_mask, pred_charged_mask = get_mask_charged_neutral(event, clustering, matches, noise_index)
    eA, eB, eC, eD = get_energy_ABCD(event, true_charged_mask, pred_charged_mask)

    eAAB = eA/(eA+eB)
    eAAC = eA/(eA+eC)
    eDCD = eD/(eC+eD)
    eDBD = eD/(eB+eD)

    # memo20230804: change definition of true charged: try also matching with PDG
    # check track bit

    # A: true charged & pred charged
    # B: true charged & pred neutral
    # C: true neutral & pred charged
    # D: true neutral & pred neutral

    # A/(A+B) <- pred_charged_over_all_true_charged
    # A/(A+C) <- true_charged_over_all_pred_charged
    # D/(C+D) <- pred_neutral_over_all_true_neutral
    # D/(B+D) <- true_neutral_over_all_pred_neutral

    stats = Stats()
    stats.add('pred_charged_over_all_true_charged', eA/(eA+eB))
    stats.add('true_charged_over_all_pred_charged', eA/(eA+eC))
    stats.add('pred_neutral_over_all_true_neutral', eD/(eC+eD))
    stats.add('true_neutral_over_all_pred_neutral', eD/(eB+eD))
    return stats

def get_matched_particle(event: Event , clustering, matches, noise_index=0):
    matched_truth = []
    matched_pred = []
    for truth_ids, pred_ids in matches:
        matched_truth.extend(truth_ids)
        matched_pred.extend(pred_ids)
    all_truth_ids = set(np.unique(event.y[:,0]))
    all_pred_ids = set(np.unique(clustering))
    all_truth_ids.discard(noise_index)
    all_pred_ids.discard(noise_index)
    #truth_charged_index = event.y[:,0][event.y[:,1]==1]
    #pred_charged_index = clustering[event.y[:,1]==1]

    print(f"{matches=}")

    print(f"{len(all_truth_ids)=}")
    print(f"{len(all_pred_ids)=}")
    print(f"{len(event.y[:,0])=}")
    print(f"{len(clustering)=}")
    #print(f"{all_truth_ids=}")
    #print(f"{all_pred_ids=}")
    #print(f"{event.y[:,0]=}")
    #print(f"{clustering=}")

    stats = Stats()
    #stats.add('pred_charged_over_all_true_charged', eA/(eA+eB))
    return stats

def is_np_array(thing):
    return hasattr(thing, 'shape') # Seems more reliable than checking against np.array


def get_category(truth_ids):
    truth_ids = np.abs(truth_ids)
    exclusively_these_ids = lambda test_ids: np.all(np.in1d(truth_ids, np.array(test_ids)))
    any_of_these_ids = lambda test_ids: np.any(np.in1d(truth_ids, np.array(test_ids)))
    if exclusively_these_ids([11, 22, 111]):
        cat = 0 # EM
    elif exclusively_these_ids([13]):
        cat = 2 # MIP
    elif any_of_these_ids([11, 22, 111, 13]):
        # If any particle is EM or MIP, but not *all*, the category must be mixed
        cat = 3 # MIX
    else:
        # The truth ids must be exclusively hadronic
        cat = 1 # HAD
    return cat


def signal_to_noise_confusion_matrix(event, clustering, norm=False):
    # Turn all signal (index > 0) into simply True:
    yp = clustering.astype(bool)
    yt = event.y.astype(bool)
    confmat = np.array([
        [((yt == 0) & (yp == 0)).sum(), ((yt == 1) & (yp == 0)).sum()],
        [((yt == 0) & (yp == 1)).sum(), ((yt == 1) & (yp == 1)).sum()]
        ])
    confmat_hitenergy = np.array([
        [event.energy[(yt == 0) & (yp == 0)].sum(), event.energy[(yt == 1) & (yp == 0)].sum()],
        [event.energy[(yt == 0) & (yp == 1)].sum(), event.energy[(yt == 1) & (yp == 1)].sum()]
        ])
    if norm:
        confmat = confmat / confmat.sum() # Cannot do in place unless casting
        confmat_hitenergy /= confmat_hitenergy.sum()
    return np.stack((confmat, confmat_hitenergy))



def ids_to_selection(ids, clustering):
    np.isin(clustering, ids)


def statistics_per_match(event: Event, clustering, matches):
    stats = Stats()
    for truth_ids, pred_ids in matches:
        ebound_truth = 0.
        sel_truth_hits = np.zeros_like(event.y, dtype=bool)
        for truth_id in truth_ids:
            sel = event.y==truth_id
            index = sel.argmax()
            sel_truth_hits[sel] = True
            ebound_truth += event.truth_e_bound[index]

        stats.add('ebound_truth', ebound_truth)
        stats.add('eta_truth', np.average(event.etahit[sel_truth_hits], weights=event.energy[sel_truth_hits]))

        sel_pred_hits = np.zeros_like(event.y, dtype=bool)
        for pred_id in pred_ids: sel_pred_hits[clustering==pred_id] = True

        stats.add(
            'energy_iou',
            event.energy[sel_truth_hits & sel_pred_hits].sum() / event.energy[sel_truth_hits | sel_pred_hits].sum()
            )
        stats.add('category', get_category(np.unique(event.truth_pdgid[sel_truth_hits])))
        stats.add('nhits_pred', sel_pred_hits.sum())
        stats.add('esum_pred', event.energy[sel_pred_hits].sum())
        stats.add('nhits_truth', sel_truth_hits.sum())
        stats.add('esum_truth', event.energy[sel_truth_hits].sum())
        stats.add('n_pred', len(pred_ids))
        stats.add('n_truth', len(truth_ids))
        #Evaluation
        #stats.add('n_true_cluster_hit',len(matches))
    return stats


def base_colorwheel():
    colorwheel = ColorWheel()
    colorwheel.assign(-1, '#bfbfbf')
    colorwheel.assign(0, '#bfbfbf')
    return colorwheel


# ____________________________________________-
# Plotly stuff

def cube_pdata(xmin, xmax, ymin, ymax, zmin, zmax):
    import plotly.graph_objects as go
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    unit_cube_edges = np.array([
        [[0,0,0], [1,0,0]],
        [[1,0,0], [1,1,0]],
        [[0,1,0], [1,1,0]],
        [[0,0,0], [0,1,0]],
        # 
        [[0,0,1], [1,0,1]],
        [[1,0,1], [1,1,1]],
        [[0,1,1], [1,1,1]],
        [[0,0,1], [0,1,1]],
        #
        [[0,0,0], [0,0,1]],
        [[1,0,0], [1,0,1]],
        [[0,1,0], [0,1,1]],
        [[1,1,0], [1,1,1]],
        ])
    offset = np.array([[xmin, ymin, zmin], [xmin, ymin, zmin]])
    scale = np.array([[dx, dy, dz], [dx, dy, dz]])
    pdata = []
    for unit_edge in unit_cube_edges:
        edge = offset + unit_edge*scale
        pdata.append(go.Scatter3d(
            x=edge[:,2], y=edge[:,0], z=edge[:,1],
            # text=['a', 'b'],
            # textposition='bottom left',
            mode='lines+markers+text',
            marker=dict(size=0),
            line=dict(
                color='black',
                width=6
                ),
            ))
    return pdata

def compile_plotly_data(
    event: Event, clustering: np.array=None, colorwheel=None
    ):
    import plotly.graph_objects as go
    if colorwheel is None: colorwheel = base_colorwheel()
    if clustering is None: clustering = event.y

    pdata = []
    energy_scale = 20./np.average(event.energy)

    for cluster_index in np.unique(clustering):
        is_noise = cluster_index == 0
        size_scale = .7 if is_noise else 1.
        sel_cluster = (clustering == cluster_index)

        if isinstance(colorwheel, ColorwheelWithProps):
            if not cluster_index in colorwheel:
                colorwheel.assign(cluster_index, alpha=.6)
            props = colorwheel(cluster_index)
            color = props['color']
            opacity = props['alpha']
        else:
            color = colorwheel(cluster_index)
            opacity = .6 if is_noise else 1.

        pdata.append(go.Scatter3d(
            x = event.zhit[sel_cluster], y=event.xhit[sel_cluster], z=event.yhit[sel_cluster],
            mode='markers', 
            marker=dict(
                line=dict(width=0),
                size=size_scale*np.maximum(0., np.minimum(3., np.log(energy_scale*event.energy[sel_cluster]))),
                color=color,
                ),
            text=[f'e={e:.3f}<br>t={t:.3f}' for e, t in zip(event.energy[sel_cluster], event.time[sel_cluster])],
            hovertemplate=(
                f'x=%{{y:0.2f}}<br>y=%{{z:0.2f}}<br>z=%{{x:0.2f}}'
                f'<br>%{{text}}'
                f'<br>clusterindex={cluster_index}'
                f'<br>pdgid={int(event.truth_pdgid_by_id(cluster_index))}'
                f'<br>E_bound={event.truth_e_bound_by_id(cluster_index):.3f}'
                f'<br>sum(E_hit)={event.energy[sel_cluster].sum():.3f}'
                f'<br>'
                ),
            name = f'cluster_{cluster_index}',
            opacity=opacity
            ))
    # pdata.extend(cube_pdata(
    #     min(event.xhit), max(event.xhit),
    #     min(event.yhit), max(event.yhit),
    #     min(event.zhit), max(event.zhit)
    #     ))
    return pdata

def compile_plotly_data_clusterspace(
    event: Event, prediction: Prediction, clustering: np.array=None, colorwheel=None
    ):
    import plotly.graph_objects as go
    if colorwheel is None: colorwheel = base_colorwheel()
    if clustering is None: clustering = event.y
    clustering = clustering[prediction.pass_noise_filter]

    pdata = []
    print(prediction.pred_cluster_space_coords.shape)
    coords = pca_down(prediction.pred_cluster_space_coords)
    print(coords.shape)
    assert coords.shape == (clustering.shape[0], 3)

    for cluster_index in np.unique(clustering):
        sel_cluster = (clustering == cluster_index)
        pdata.append(go.Scatter3d(
            x = coords[sel_cluster,0], y=coords[sel_cluster,1], z=coords[sel_cluster,2],
            mode='markers', 
            marker=dict(
                line=dict(width=0),
                size=1.,
                color=colorwheel(int(cluster_index)),
                ),
            hovertemplate=(
                f'x=%{{x:0.2f}}<br>y=%{{y:0.2f}}<br>z=%{{z:0.2f}}'
                ),
            name = f'cluster_{cluster_index}'
            ))
    return pdata


def _make_parent_dirs_and_format(outfile, touch=False):
    import os, os.path as osp
    from time import strftime
    outfile = strftime(outfile)
    outdir = osp.dirname(osp.abspath(outfile))
    if not osp.isdir(outdir): os.makedirs(outdir)
    if touch:
        with open(outfile, 'w'):
            pass
    return outfile

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

#     import re
#     mode_bar_plugin = """\"modeBarButtons\": [[{
#     name: "save camera",
#     click: function(gd) {
#       var scene = gd._fullLayout.scene._scene;
#       scene.saveCamera(gd.layout);     
#     }
#   }, 
#     "toImage"
#   ]],"""
#     print(re.search(r'\}\],\s*\{', fig_html))
#     fig_html = re.sub(r'\}\],\s*\{', '}]\n{' + mode_bar_plugin + '\n', fig_html)

    outfile = _make_parent_dirs_and_format(outfile)

def side_by_side_pdata_to_file(
    outfile, pdata1, pdata2,
    title1=None, title2=None, width=600, height=None, include_plotlyjs='cdn',
    mode='w', legend=True
    ):
    import plotly.graph_objects as go
    scene = dict(xaxis_title='z (cm)', yaxis_title='x (cm)', zaxis_title='y (cm)', aspectmode='cube', xaxis_mirror=True)
    if height is None: height = width
    fig1 = go.Figure(data=pdata1, **(dict(layout_title_text=title1) if title1 else {}))
    fig1.update_layout(width=width, height=height, scene=scene, showlegend=legend)
    fig2 = go.Figure(data=pdata2, **(dict(layout_title_text=title2) if title2 else {}))
    fig2.update_layout(width=width, height=height, scene=scene, showlegend=legend)
    fig1_html = fig1.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
    fig2_html = fig2.to_html(full_html=False, include_plotlyjs=False)
    divid1 = fig1_html.split('<div id="',1)[1].split('"',1)[0]
    divid2 = fig2_html.split('<div id="',1)[1].split('"',1)[0]
    id1 = str(uuid.uuid4())[:6]
    id2 = str(uuid.uuid4())[:6]
    # Compile html: Sync camera angles in javascript
    html = (
        f'<div style="width: 47%; display: inline-block">\n{fig1_html}\n</div>'
        f'\n<div style="width: 47%; display: inline-block">\n{fig2_html}\n</div>'
        f'\n<script>'
        f'\nvar graphdiv_{id1} = document.getElementById("{divid1}");'
        f'\nvar graphdiv_{id2} = document.getElementById("{divid2}");'
        f'\nvar isUnderRelayout_{id1} = false'
        f'\ngraphdiv_{id1}.on("plotly_relayout", () => {{'
        f'\n    // console.log("relayout", isUnderRelayout_{id1})'
        f'\n    if (!isUnderRelayout_{id1}) {{'
        f'\n        Plotly.relayout(graphdiv_{id2}, {{"scene.camera": graphdiv_{id1}.layout.scene.camera}})'
        f'\n        .then(() => {{ isUnderRelayout_{id1} = false }}  )'
        f'\n        }}'
        f'\n    isUnderRelayout_{id1} = true;'
        f'\n    }})'
        f'\nvar isUnderRelayout_{id2} = false'
        f'\ngraphdiv_{id2}.on("plotly_relayout", () => {{'
        f'\n    // console.log("relayout", isUnderRelayout_{id2})'
        f'\n    if (!isUnderRelayout_{id2}) {{'
        f'\n        Plotly.relayout(graphdiv_{id1}, {{"scene.camera": graphdiv_{id2}.layout.scene.camera}})'
        f'\n        .then(() => {{ isUnderRelayout_{id2} = false }}  )'
        f'\n        }}'
        f'\n    isUnderRelayout_{id2} = true;'
        f'\n    }})'
        f'\n</script>'
        )
    outfile = _make_parent_dirs_and_format(outfile)
    with open(outfile, mode) as f:
        f.write(html)
