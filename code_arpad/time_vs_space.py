import awkward as ak
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
from itertools import combinations
sys.path.append('/home/aschaeff/ml-pfa')
import tools.load_awkward as la

data_dir = '/data/suehara/mldata/pfa/murata/ntau_lessSample/train'
h5_files = sorted([
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir) if f.endswith('.h5')
])

bundle = la.load_awkward2(h5_files[0])
feat, label = bundle[0], bundle[1]
basename = os.path.splitext(os.path.basename(h5_files[0]))[0]
os.makedirs('plots_par_event', exist_ok=True)

n_events_to_plot = 1

# Pour accumuler sur tous les événements
all_spatial_dist = []
all_time_diff    = []
all_pdg_pair     = []   # label de la paire ex: "π-K"

pdg_names = {211: 'π', 321: 'K', 2212: 'p', 22: 'γ', 11: 'e', 2112: 'n',13 : 'μ', 311: 'K0'}

for i in range(n_events_to_plot):
    f = ak.to_numpy(feat[i])
    l = ak.to_numpy(label[i])

    mcids = l[:, 1]
    mask  = mcids != -1

    E     = f[mask, 0]
    x_pos = f[mask, 1]
    y_pos = f[mask, 2]
    z_pos = f[mask, 3]
    time  = f[mask, 4]
    pdg   = np.abs(l[mask, 2]).astype(int)
    mcid  = l[mask, 1].astype(int)

    # Calcul barycentre et temps moyen par particule
    particules = {}
    for uid in np.unique(mcid):
        m = mcid == uid
        E_part = E[m]
        w = E_part / E_part.sum()   # poids = fraction d'énergie

        bary_x = np.sum(w * x_pos[m])
        bary_y = np.sum(w * y_pos[m])
        bary_z = np.sum(w * z_pos[m])
        t_mean = np.mean(time[m])
        pdg_id = np.bincount(pdg[m]).argmax()   # PDG majoritaire

        particules[uid] = {
            'bary': np.array([bary_x, bary_y, bary_z]),
            't':    t_mean,
            'pdg':  pdg_id,
            'n_hits': m.sum()
        }

    # Toutes les paires de particules
    uids = list(particules.keys())
    for uid_a, uid_b in combinations(uids, 2):
        pa = particules[uid_a]
        pb = particules[uid_b]

        dist_spatiale = np.linalg.norm(pa['bary'] - pb['bary'])
        diff_temps    = abs(pa['t'] - pb['t'])

        name_a = pdg_names.get(pa['pdg'], f"pdg{pa['pdg']}")
        name_b = pdg_names.get(pb['pdg'], f"pdg{pb['pdg']}")
        label_paire = '-'.join(sorted([name_a, name_b]))

        all_spatial_dist.append(dist_spatiale)
        all_time_diff.append(diff_temps)
        all_pdg_pair.append(label_paire)

    print(f'Event {i} : {len(uids)} particules, {len(list(combinations(uids,2)))} paires')

all_spatial_dist = np.array(all_spatial_dist)
all_time_diff    = np.array(all_time_diff)
all_pdg_pair     = np.array(all_pdg_pair)

# ── Figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Séparation spatiale vs temporelle — {n_events_to_plot} événements — {basename}')

# Graphe 1 : scatter toutes paires, coloré par type
paire_types = np.unique(all_pdg_pair)
colors = plt.cm.tab10(np.linspace(0, 1, len(paire_types)))

for paire, color in zip(paire_types, colors):
    m = all_pdg_pair == paire
    axes[0].scatter(all_spatial_dist[m], all_time_diff[m],
                    s=10, alpha=0.6, label=f'{paire} ({m.sum()})', color=color)

axes[0].set_xlabel('distance spatiale entre barycentres (mm)')
axes[0].set_ylabel('différence temporelle moyenne (ns)')
axes[0].set_title('Toutes paires de particules')
axes[0].legend(fontsize=8, markerscale=2)

# Zone d'intérêt : overlap spatial (distance < seuil)
seuil_mm = 1000   # à ajuster 
axes[0].axvline(x=seuil_mm, color='red', linestyle='--', linewidth=1,
                label=f'seuil overlap = {seuil_mm} mm')

# Graphe 2 : zoom sur les paires en overlap spatial
mask_overlap = all_spatial_dist < seuil_mm
axes[1].set_title(f'Zoom : paires avec distance < {seuil_mm} mm')

if mask_overlap.sum() > 0:
    for paire, color in zip(paire_types, colors):
        m = (all_pdg_pair == paire) & mask_overlap
        if m.sum() > 0:
            axes[1].scatter(all_spatial_dist[m], all_time_diff[m],
                            s=20, alpha=0.7, label=f'{paire} ({m.sum()})', color=color)
    axes[1].set_xlabel('distance spatiale (mm)')
    axes[1].set_ylabel('différence temporelle (ns)')
    axes[1].legend(fontsize=8, markerscale=2)
else:
    axes[1].text(0.5, 0.5, f'aucune paire\navec distance < {seuil_mm} mm',
                 transform=axes[1].transAxes, ha='center', va='center',
                 fontsize=11, color='gray')

plt.subplots_adjust(hspace=0.4, wspace=0.35)
plt.savefig(f'plots_par_event/{basename}_time_vs_space_{n_events_to_plot}events.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Terminé — plot sauvegardé.')