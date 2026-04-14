"""
eval_no_root.py — GravNet evaluation without ROOT
================================================
Replaces save_root.py for cases where ROOT/cppyy is broken.
Produces a .npz file directly readable with numpy.

Usage
-----
    python eval_no_root.py \
        --datapath /data/suehara/mldata/pfa/murata/data/tc/tc_nnqq/train \
        --ckpt     /path/to/checkpoint.pth.tar \
        --outfile  results_timing.npz \
        --input-dim 6 \
        --output-dim 5 \
        --nstart 0 \
        --nend 200 \
        --tbeta 0.9 \
        --td 0.5 \
        --device cpu

Reading the results
-------------------
    import numpy as np
    d = np.load('results_timing.npz', allow_pickle=True)
    # Average efficiency per particle species
    mask_pion = d['pdg'] == 211
    print('Pion efficiency:', d['efficiency'][mask_pion].mean())
    print('Pion purity    :', d['purity'][mask_pion].mean())
"""

import argparse
import sys
import numpy as np
import torch

# ── imports from the repo (to be run from the root of ml-pfa) ──────────────────
from model import get_model
from dataset import ILCDataset
from test_yielder import TestYielder


# ── PDG codes ────────────────────────────────────────────────────────
PDG_NAMES = {
    22:   'photon',
    11:   'electron',
    13:   'muon',
    211:  'pion+',
    321:  'kaon+',
    2212: 'proton',
    2112: 'neutron',
    130:  'K0L',
    310:  'K0S',
}


def run(args):
    # ── Model loading ────────────────────────────────────────────────
    print(f"Model loading from {args.ckpt}")
    model = get_model(
        args.ckpt,
        jit=False,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
    ).to(args.device)

    # ── data loading ─────────────────────────────────────────────────
    print(f"Data loading from {args.datapath}")
    dataset = ILCDataset(
        args.datapath,
        timingCut=args.timing_cut,
        test_mode=True,
        nstart=args.nstart,
        nend=args.nend,
    )

    yielder = TestYielder(
        model=model,
        dataset=dataset,
        device=args.device,
    )

    nmax = None if args.nend == -1 else args.nend - args.nstart


    # ── Evaluation loop ─────────────────────────────────────────────────
    # For each matched MC particle we store:
    #   pdg       : PDG code of the particle
    #   mom       : momentum (GeV/c)
    #   edep      : true energy (sum of MC hits)
    #   edep_reco : total energy of the matched reco cluster
    #   edep_match: MC energy in the matched reco cluster
    #   efficiency: edep_match / edep
    #   purity    : edep_match / edep_reco  (computed from the reco cluster)
    

    rows_mc    = []   # one dict per matched MC particle, to be converted to arrays at the end

    print("Starting evaluation...")
    for i, (event, prediction, clustering, matches, _) in enumerate(
        yielder.iter_matches(
            tbeta=args.tbeta,
            td=args.td,
            nmax=nmax,
        )
    ):
        if i < 5 or i % 50 == 0:
            print(f"  Event {i}...")

        matches12, matches21 = matches   # mc→reco, reco→mc
        all_truth_ids = list(set(np.unique(event.y[:, 0])))

        for tid in all_truth_ids:
            # ── truth ──────────────────────────────
            mask_mc = (event.y[:, 0] == tid)
            feat_mc = event.feat[mask_mc]
            label_mc = event.label[mask_mc]

            edep = float(feat_mc[:, 0].sum())
            if edep <= 0:
                continue

            # PDG and momentum
            my_label = label_mc[0].numpy()
            pdg_raw  = int(my_label[2])
            pdg      = abs(pdg_raw)
            px, py, pz = float(my_label[5]), float(my_label[6]), float(my_label[7])
            mom = float(np.sqrt(px**2 + py**2 + pz**2))

            # mcid == -1 → track without true particle (e.g. noise hits), we skip it
            if int(my_label[1]) == -1:
                continue

            # ── matching mc → cluster reco ──────────────────────────────
            if tid not in matches12:
                continue   # unmatched particle (no matching reco cluster)

            # we loop over all matched reco clusters and keep the one with the highest edep_match (best match)
            best_edep_match = 0.0
            best_edep_reco  = 0.0

            for rid in matches12[tid]:
                mask_cl = (clustering == rid)
                edep_reco_cl   = float(event.feat[mask_cl][:, 0].sum())
                mask_both      = mask_mc & mask_cl
                edep_match_cl  = float(event.feat[mask_both][:, 0].sum())

                if edep_match_cl > best_edep_match:
                    best_edep_match = edep_match_cl
                    best_edep_reco  = edep_reco_cl

            if best_edep_reco <= 0:
                continue

            eff = best_edep_match / edep
            pur = best_edep_match / best_edep_reco

            rows_mc.append({
                'event':      i,
                'pdg':        pdg,
                'mom':        mom,
                'edep':       edep,
                'edep_reco':  best_edep_reco,
                'edep_match': best_edep_match,
                'efficiency': eff,
                'purity':     pur,
            })

    # ── save results ──────────────────────────────────────────────────────────
    if not rows_mc:
        print("BE CAREFUL: No results collected!")
        return

    keys = rows_mc[0].keys()
    out  = {k: np.array([r[k] for r in rows_mc]) for k in keys}

    np.savez(args.outfile, **out)
    print(f"\n{len(rows_mc)} MC particles saved to {args.outfile}.npz")

    # ── Immediate summary ─────────────────────────────────────────────────────
    print("\n── Summary of efficiency / purity by species ──")
    print(f"{'Species':<12} {'N':>6}  {'Avg Eff':>8}  {'Avg Pur':>8}")
    print("-" * 42)

    pdgs = out['pdg']
    effs = out['efficiency']
    purs = out['purity']

    for pdg_code, name in sorted(PDG_NAMES.items(), key=lambda x: x[1]):
        mask = (pdgs == pdg_code)
        n = mask.sum()
        if n == 0:
            continue
        print(f"{name:<12} {n:>6}  {effs[mask].mean():>8.4f}  {purs[mask].mean():>8.4f}")

    print("-" * 42)
    print(f"{'TOTAL':<12} {len(pdgs):>6}  {effs.mean():>8.4f}  {purs.mean():>8.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datapath',    required=True,
                   help='Folder containing .h5 test files')
    p.add_argument('--ckpt',        required=True,
                   help='Trained model checkpoint .pth.tar')
    p.add_argument('--outfile',     default='eval_results',
                   help='Output file name (without .npz extension)')
    p.add_argument('--input-dim',   type=int, default=6,
                   help='Model input dimension (5 without timing, 6 with)')
    p.add_argument('--output-dim',  type=int, default=5,
                   help='Model output dimension')
    p.add_argument('--nstart',      type=int, default=0)
    p.add_argument('--nend',        type=int, default=200,
                   help='Number of events to evaluate (-1 = all)')
    p.add_argument('--tbeta',       type=float, default=0.9,
                   help='Beta threshold for condensation points')
    p.add_argument('--td',          type=float, default=0.5,
                   help='Distance threshold for OC clustering')
    p.add_argument('--timing-cut',  action='store_true',
                   help='Apply timing cut (4-14 ns)')
    p.add_argument('--device',      default='cpu',
                   help='PyTorch device (cpu, cuda, cuda:0...)')
    args = p.parse_args()

    run(args)


if __name__ == '__main__':
    main()