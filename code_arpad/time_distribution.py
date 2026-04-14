import awkward as ak
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.append('/home/aschaeff/ml-pfa')
import tools.load_awkward as la
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--grouped', action='store_true',
                    help='Un seul PNG avec tous les événements groupés')
args = parser.parse_args()

data_dir = '/data/suehara/mldata/pfa/murata/ntau_lessSample/train'
h5_files = sorted([
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir) if f.endswith('.h5')
])

bundle = la.load_awkward2(h5_files[0])
feat, label = bundle[0], bundle[1]

basename = os.path.splitext(os.path.basename(h5_files[0]))[0]
os.makedirs('plots_par_event', exist_ok=True)

n_events_to_plot = 5  # commence par 5, pas tous

pdg_map = {211:  ('π±',  'tab:blue'),
           111:  ('π0',  'tab:cyan'),
           321:  ('K±',  'tab:orange'),
           2212: ('p',   'tab:green'),
           22:   ('γ',   'tab:red'),
           11:   ('e±',  'tab:purple'),
           2112: ('n',   'tab:brown'),
           311:  ('K0',  'tab:pink'),
           }

if args.grouped:
    # figure unique créée AVANT la boucle
    fig, axes = plt.subplots(n_events_to_plot, 3, figsize=(15, 7 * n_events_to_plot))
    fig.suptitle(f'Fichier {basename}', fontsize=14)
else:
    pass  # fig créée DANS la boucle 


for i in range(n_events_to_plot):
    f = ak.to_numpy(feat[i])     # (N_hits, 10)
    l = ak.to_numpy(label[i])    # (N_hits, 9)

    mcids = l[:, 1]
    mask  = mcids != -1          # exclure le bruit

    time = f[mask, 4]
    pdg  = np.abs(l[mask, 2]).astype(int)
    E    = f[mask, 0]
    n_hits = mask.sum()

    if not args.grouped:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Événement {i} — {n_hits} hits — {basename}')

    # Graphe 1 : distribution temporelle globale
    axes[i, 0].hist(time, bins=50, color='steelblue', edgecolor='none')
    axes[i, 0].set_xlabel('temps (ns)')
    axes[i, 0].set_ylabel('hits')
    axes[i, 0].set_title('Distribution temporelle')

    # Graphe 2 : par type de particule
    for code, (name, color) in pdg_map.items():
        m = pdg == code
        if m.sum() > 0:
            axes[i, 1].hist(time[m], bins=40, alpha=0.5,
                            label=f'{name} ({m.sum()})', color=color)
    axes[i, 1].set_xlabel('temps (ns)')
    axes[i, 1].set_title('Temps par espèce')
    axes[i, 1].legend(fontsize=8)

    # Graphe 3 : temps vs énergie
    axes[i, 2].scatter(E, time, s=2, alpha=0.4, color='steelblue')
    axes[i, 2].set_xlabel('énergie déposée (GeV)')
    axes[i, 2].set_ylabel('temps (ns)')
    axes[i, 2].set_title('Temps vs énergie')
    axes[i, 2].set_xscale('log')
    if not args.grouped:
        plt.tight_layout()
        plt.savefig(f'plots_par_event/{basename}_event{i:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        
if args.grouped:
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.savefig(f'plots_par_event/{basename}_events_grouped.png', dpi=150, bbox_inches='tight')
    plt.close()

print('Terminé.')   