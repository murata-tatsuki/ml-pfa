import awkward as ak
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
from itertools import combinations
from scipy.spatial import cKDTree
sys.path.append('/home/aschaeff/ml-pfa')
import tools.load_awkward as la

data_dir = '/data/suehara/mldata/pfa/murata/data/tc/tc_nnqq/train'
h5_files = sorted([
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir) if f.endswith('.h5')
])

bundle = la.load_awkward2(h5_files[0])
feat, label = bundle[0], bundle[1]
basename = os.path.splitext(os.path.basename(h5_files[0]))[0]
os.makedirs('plots_par_event', exist_ok=True)

n_events_to_plot = 1
seuil_mm = 50   # distance minimale inter-hits pour considérer un overlap réel

pdg_names = {211: 'π', 321: 'K', 2212: 'p', 22: 'γ', 11: 'e', 2112: 'n'}

# Accumulation sur tous les événements
all_overlap_dist = []   # distance minimale inter-hits entre les deux particules
all_time_diff    = []   # différence de temps moyen
all_pdg_pair     = []   # label ex: "π-K"

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

    # Construire le dictionnaire des particules
    particules = {}
    for uid in np.unique(mcid):
        m = mcid == uid
        if m.sum() < 2:      # ignorer les particules avec un seul hit
            continue
        particules[uid] = {
            'hits': np.stack([x_pos[m], y_pos[m], z_pos[m]], axis=1),
            't':    np.mean(time[m]),
            'pdg':  int(np.bincount(pdg[m]).argmax()),
        }

    uids = list(particules.keys())
    n_paires_total   = 0
    n_paires_overlap = 0

    for uid_a, uid_b in combinations(uids, 2):
        n_paires_total += 1
        pa = particules[uid_a]
        pb = particules[uid_b]

        # Distance minimale entre les deux nuages de hits
        tree = cKDTree(pb['hits'])
        dist_min, _ = tree.query(pa['hits'], k=1)
        overlap = dist_min.min()

        if overlap > seuil_mm:
            continue   # pas d'overlap réel, on ignore cette paire

        n_paires_overlap += 1
        diff_temps = abs(pa['t'] - pb['t'])

        name_a = pdg_names.get(pa['pdg'], f"pdg{pa['pdg']}")
        name_b = pdg_names.get(pb['pdg'], f"pdg{pb['pdg']}")
        label_paire = '-'.join(sorted([name_a, name_b]))

        all_overlap_dist.append(overlap)
        all_time_diff.append(diff_temps)
        all_pdg_pair.append(label_paire)

    print(f'Event {i} : {len(uids)} particules — '
          f'{n_paires_total} paires total — '
          f'{n_paires_overlap} paires en overlap (< {seuil_mm} mm)')

if len(all_overlap_dist) == 0:
    print(f'Aucune paire en overlap trouvée avec seuil={seuil_mm} mm.')
    print('Essaie d augmenter seuil_mm.')
    sys.exit()

all_overlap_dist = np.array(all_overlap_dist)
all_time_diff    = np.array(all_time_diff)
all_pdg_pair     = np.array(all_pdg_pair)

print(f'\nTotal paires en overlap : {len(all_overlap_dist)}')
print(f'Différence temporelle — min={all_time_diff.min():.3f} ns  '
      f'max={all_time_diff.max():.3f} ns  '
      f'median={np.median(all_time_diff):.3f} ns')

# ── Figure ─────────────────────────────────────────────────────────
paire_types = np.unique(all_pdg_pair)
colors = plt.cm.tab10(np.linspace(0, 1, len(paire_types)))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f'Paires en overlap spatial (dist < {seuil_mm} mm) — '
    f'{n_events_to_plot} événements — {basename}'
)

# Graphe 1 : scatter overlap vs delta_t, coloré par type de paire
for paire, color in zip(paire_types, colors):
    m = all_pdg_pair == paire
    axes[0].scatter(all_overlap_dist[m], all_time_diff[m],
                    s=20, alpha=0.7, label=f'{paire} ({m.sum()})', color=color)
axes[0].set_xlabel('distance minimale inter-hits (mm)')
axes[0].set_ylabel('différence temporelle moyenne (ns)')
axes[0].set_title('Distance inter-hits vs Δt')
axes[0].legend(fontsize=8, markerscale=1.5)

# Graphe 2 : histogramme de Δt pour les paires en overlap
# C'est le graphe clé : si la distribution est large, le timing est discriminant
axes[1].set_title(f'Distribution de Δt pour paires en overlap')
for paire, color in zip(paire_types, colors):
    m = all_pdg_pair == paire
    if m.sum() > 0:
        axes[1].hist(all_time_diff[m], bins=40, alpha=0.5,
                     label=f'{paire} ({m.sum()})', color=color)
axes[1].set_xlabel('différence temporelle (ns)')
axes[1].set_ylabel('nombre de paires')
axes[1].legend(fontsize=8)

# Ligne verticale : résolution typique attendue (100 ps = 0.1 ns)
axes[1].axvline(x=0.1, color='red', linestyle='--', linewidth=1.2,
                label='σ_t = 100 ps')
axes[1].legend(fontsize=8)

plt.subplots_adjust(wspace=0.35)
plt.savefig(f'plots_par_event/{basename}_time_vs_overlap_{n_events_to_plot}events.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Plot sauvegardé.')